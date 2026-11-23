"""Compaction: retention with a hash-chain checkpoint, verify still passes."""

import json
import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone

from blackbox import Recorder, compact_log
from blackbox.recorder import GENESIS_PREV_HASH, _hash_record
from blackbox.replay import load_records
from blackbox.verify import verify


def _write_log(path, steps=5):
    with Recorder(path) as rec:
        for i in range(steps):
            rec.record("note", {"text": "step %d" % i})


def _write_dated_log(path, timestamps):
    """Write a valid chained log with the given ISO timestamps."""
    prev_hash = GENESIS_PREV_HASH
    last_hash = None
    with open(path, "w", encoding="utf-8") as f:
        for i, ts in enumerate(timestamps):
            record = {"seq": i, "ts": ts, "type": "note",
                      "payload": {"text": "step %d" % i}, "tokens": None,
                      "prev_hash": prev_hash}
            record["hash"] = _hash_record(record)
            f.write(json.dumps(record, sort_keys=True) + "\n")
            prev_hash = record["hash"]
            last_hash = record["hash"]
    return last_hash


def _tamper(path, position, mutate):
    with open(path) as f:
        lines = f.readlines()
    record = json.loads(lines[position])
    mutate(record)
    lines[position] = json.dumps(record, sort_keys=True) + "\n"
    with open(path, "w") as f:
        f.writelines(lines)


class CompactTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "run.jsonl")

    def tearDown(self):
        self.tmp.cleanup()

    def test_keep_n_round_trip_verifies(self):
        _write_log(self.path, steps=5)
        original = load_records(self.path)
        removed_head = original[2]["hash"]  # last removed step
        result = compact_log(self.path, keep_n=2)
        self.assertTrue(result["changed"])
        self.assertEqual(result["compacted"], 3)
        self.assertEqual(result["kept"], 2)
        self.assertEqual(result["head_hash"], removed_head)

        records = load_records(self.path)
        self.assertEqual(len(records), 3)  # checkpoint + 2 kept
        checkpoint = records[0]
        self.assertEqual(checkpoint["type"], "checkpoint")
        self.assertEqual(checkpoint["seq"], 0)
        self.assertEqual(checkpoint["prev_hash"], GENESIS_PREV_HASH)
        self.assertEqual(checkpoint["payload"]["compacted_count"], 3)
        self.assertEqual(checkpoint["payload"]["head_hash"], removed_head)
        self.assertEqual(checkpoint["payload"]["original_events"], 5)
        self.assertEqual([r["payload"]["text"] for r in records[1:]],
                         ["step 3", "step 4"])
        self.assertEqual([r["seq"] for r in records], [0, 1, 2])

        check = verify(self.path)
        self.assertTrue(check["ok"], check)
        self.assertEqual(check["records"], 3)
        self.assertEqual(check["compacted"], 3)

    def test_keep_days_uses_timestamps(self):
        now = datetime.now(timezone.utc)
        stamps = [
            (now - timedelta(days=10)).isoformat().replace("+00:00", "Z"),
            (now - timedelta(days=5)).isoformat().replace("+00:00", "Z"),
            (now - timedelta(hours=1)).isoformat().replace("+00:00", "Z"),
            now.isoformat().replace("+00:00", "Z"),
        ]
        _write_dated_log(self.path, stamps)
        result = compact_log(self.path, keep_days=1)
        self.assertTrue(result["changed"])
        self.assertEqual(result["compacted"], 2)
        self.assertEqual(result["kept"], 2)
        self.assertTrue(verify(self.path)["ok"])

    def test_tamper_after_compaction_detected_at_that_seq(self):
        _write_log(self.path, steps=5)
        compact_log(self.path, keep_n=2)
        _tamper(self.path, 2, lambda r: r["payload"].update(
            {"text": "step 4, edited later"}))
        check = verify(self.path)
        self.assertFalse(check["ok"])
        self.assertEqual(check["first_bad_seq"], 2)

    def test_tampered_checkpoint_detected_at_seq_zero(self):
        _write_log(self.path, steps=5)
        compact_log(self.path, keep_n=2)
        _tamper(self.path, 0, lambda r: r["payload"].update(
            {"compacted_count": 999}))
        check = verify(self.path)
        self.assertFalse(check["ok"])
        self.assertEqual(check["first_bad_seq"], 0)

    def test_tampered_head_hash_detected(self):
        _write_log(self.path, steps=5)
        compact_log(self.path, keep_n=2)
        _tamper(self.path, 0, lambda r: r["payload"].update(
            {"head_hash": "0" * 64}))
        check = verify(self.path)
        self.assertFalse(check["ok"])
        self.assertEqual(check["first_bad_seq"], 0)

    def test_compact_twice_accumulates_count(self):
        _write_log(self.path, steps=5)
        compact_log(self.path, keep_n=3)
        # Recording resumes on the compacted log: new steps chain from
        # the kept records, and close writes a fresh summary.
        with Recorder(self.path) as rec:
            rec.record("note", {"text": "step 5"})
            rec.record("note", {"text": "step 6"})
        result = compact_log(self.path, keep_n=2)
        self.assertEqual(result["compacted"], 3)
        records = load_records(self.path)
        self.assertEqual(records[0]["payload"]["compacted_count"], 5)
        self.assertEqual([r["payload"]["text"] for r in records[1:]],
                         ["step 5", "step 6"])
        self.assertTrue(verify(self.path)["ok"])

    def test_nothing_to_remove_leaves_log_untouched(self):
        _write_log(self.path, steps=3)
        with open(self.path, "rb") as f:
            before = f.read()
        result = compact_log(self.path, keep_n=100)
        self.assertFalse(result["changed"])
        self.assertEqual(result["compacted"], 0)
        with open(self.path, "rb") as f:
            self.assertEqual(f.read(), before)
        self.assertTrue(verify(self.path)["ok"])

    def test_dry_run_reports_without_writing(self):
        _write_log(self.path, steps=5)
        with open(self.path, "rb") as f:
            before = f.read()
        result = compact_log(self.path, keep_n=2, dry_run=True)
        self.assertTrue(result["changed"])
        self.assertEqual(result["compacted"], 3)
        self.assertIsNone(result["checkpoint_hash"])
        with open(self.path, "rb") as f:
            self.assertEqual(f.read(), before)

    def test_compact_requires_a_criterion(self):
        _write_log(self.path, steps=2)
        with self.assertRaises(ValueError):
            compact_log(self.path)

    def test_compact_sqlite_round_trip(self):
        db_path = os.path.join(self.tmp.name, "run.db")
        _write_log(db_path, steps=5)
        result = compact_log(db_path, keep_n=2)
        self.assertTrue(result["changed"])
        self.assertEqual(result["compacted"], 3)
        records = load_records(db_path)
        self.assertEqual(len(records), 3)
        self.assertEqual(records[0]["type"], "checkpoint")
        self.assertEqual([r["payload"]["text"] for r in records[1:]],
                         ["step 3", "step 4"])
        check = verify(db_path)
        self.assertTrue(check["ok"], check)
        self.assertEqual(check["compacted"], 3)

    def test_recorder_resumes_and_verifies_after_compact(self):
        _write_log(self.path, steps=5)
        compact_log(self.path, keep_n=2)
        with Recorder(self.path) as rec:
            rec.record("note", {"text": "step 5"})
        check = verify(self.path)
        self.assertTrue(check["ok"], check)
        self.assertEqual(check["records"], 5)  # checkpoint + 3 + summary


if __name__ == "__main__":
    unittest.main()
