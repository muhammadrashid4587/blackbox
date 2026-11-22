"""SQLite backend: write, verify, tamper, resume, and cross-backend
replay/diff. The JSONL behaviour is covered by the existing tests;
these tests prove the SQLite store keeps the same records and the
same hash chain."""

import io
import json
import os
import sqlite3
import tempfile
import unittest
from contextlib import redirect_stdout

from blackbox import Recorder
from blackbox.replay import diff, load_records, replay
from blackbox.verify import verify


def _write_run(path, steps=5):
    with Recorder(path) as rec:
        for i in range(steps):
            rec.record("note", {"text": "step %d" % i})


class SqliteTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "run.db")

    def tearDown(self):
        self.tmp.cleanup()

    def test_write_and_verify_ok(self):
        _write_run(self.path)
        result = verify(self.path)
        self.assertTrue(result["ok"], result)
        self.assertEqual(result["records"], 6)  # 5 steps + summary
        self.assertIsNone(result["first_bad_seq"])

    def test_tampered_row_caught_at_that_seq(self):
        _write_run(self.path)
        conn = sqlite3.connect(self.path)
        conn.execute(
            "UPDATE records SET payload = ? WHERE seq = ?",
            (json.dumps({"text": "step 2, edited later"}), 2))
        conn.commit()
        conn.close()
        result = verify(self.path)
        self.assertFalse(result["ok"])
        self.assertEqual(result["first_bad_seq"], 2)

    def test_tampered_hash_column_caught(self):
        _write_run(self.path)
        conn = sqlite3.connect(self.path)
        conn.execute("UPDATE records SET hash = ? WHERE seq = ?",
                     ("0" * 64, 1))
        conn.commit()
        conn.close()
        result = verify(self.path)
        self.assertFalse(result["ok"])
        self.assertEqual(result["first_bad_seq"], 1)

    def test_deleted_row_caught(self):
        _write_run(self.path)
        conn = sqlite3.connect(self.path)
        conn.execute("DELETE FROM records WHERE seq = 3")
        conn.commit()
        conn.close()
        result = verify(self.path)
        self.assertFalse(result["ok"])
        # The record after the gap now sits at the wrong position.
        self.assertEqual(result["first_bad_seq"], 3)

    def test_resume_continues_chain(self):
        _write_run(self.path, steps=3)
        _write_run(self.path, steps=2)  # resume: 2 more steps
        result = verify(self.path)
        self.assertTrue(result["ok"], result)
        records = load_records(self.path)
        self.assertEqual([r["seq"] for r in records], [0, 1, 2, 3, 4, 5])
        # The first run's summary was dropped on resume and replaced by
        # the second run's summary, so the store has exactly one summary.
        self.assertEqual(records[-1]["type"], "summary")
        self.assertEqual(records[-1]["payload"]["events"], 5)

    def test_replay_reads_sqlite(self):
        _write_run(self.path)
        buf = io.StringIO()
        with redirect_stdout(buf):
            records = replay(self.path)
        out = buf.getvalue()
        self.assertEqual(len(records), 6)  # 5 steps + summary
        for needle in ("step 0", "step 4", "summary"):
            self.assertIn(needle, out)

    def test_diff_across_backends(self):
        # The same steps recorded to JSONL and to SQLite diff identical:
        # the chain and the records are backend-independent.
        jsonl = os.path.join(self.tmp.name, "run.jsonl")
        _write_run(jsonl)
        _write_run(self.path)
        buf = io.StringIO()
        with redirect_stdout(buf):
            result = diff(jsonl, self.path)
        self.assertEqual(result["changed"], [])
        self.assertEqual(result["added"], [])
        self.assertEqual(result["removed"], [])
        self.assertIn("IDENTICAL", buf.getvalue())

    def test_durability_pragmas(self):
        # synchronous is a per-connection setting, so check it on the
        # store's own live connection, not a fresh one.
        from blackbox.sqlite_store import SQLiteStore
        store = SQLiteStore(self.path)
        journal = store.conn.execute("PRAGMA journal_mode").fetchone()[0]
        synchronous = store.conn.execute("PRAGMA synchronous").fetchone()[0]
        store.close()
        self.assertEqual(journal, "wal")
        self.assertEqual(synchronous, 2)  # FULL

    def test_missing_db(self):
        result = verify(os.path.join(self.tmp.name, "nope.db"))
        self.assertFalse(result["ok"])
        self.assertIn("not found", result["error"])

    def test_not_a_database(self):
        with open(self.path, "w") as f:
            f.write("this is not a database\n")
        result = verify(self.path)
        self.assertFalse(result["ok"])
        self.assertIn("not a blackbox database", result["error"])

    def test_record_after_close_raises(self):
        rec = Recorder(self.path)
        rec.record("note", {"text": "x"})
        rec.close()
        with self.assertRaises(RuntimeError):
            rec.record("note", {"text": "too late"})

    def test_summary_totals(self):
        with Recorder(self.path, task="t", model="m") as rec:
            rec.record("model_call", {"model": "m", "prompt": "a"}, tokens=100)
            rec.record("model_call", {"model": "m", "prompt": "b"},
                       tokens={"input": 50, "output": 70})
            rec.record("note", {"text": "no tokens here"})
        records = load_records(self.path)
        summary = records[-1]
        self.assertEqual(summary["type"], "summary")
        self.assertEqual(summary["payload"]["events"], 3)
        self.assertEqual(summary["payload"]["total_tokens"], 220)
        self.assertEqual(summary["payload"]["task"], "t")


if __name__ == "__main__":
    unittest.main()
