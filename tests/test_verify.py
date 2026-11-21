"""Verify: tampering, truncation and torn writes are caught at the
exact record."""

import json
import os
import tempfile
import unittest

from blackbox import Recorder
from blackbox.verify import verify


def _write_log(path, steps=5):
    with Recorder(path) as rec:
        for i in range(steps):
            rec.record("note", {"text": "step %d" % i})


class VerifyTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "run.jsonl")

    def tearDown(self):
        self.tmp.cleanup()

    def test_clean_log_ok(self):
        _write_log(self.path)
        result = verify(self.path)
        self.assertTrue(result["ok"], result)
        self.assertEqual(result["records"], 6)  # 5 steps + summary
        self.assertIsNone(result["first_bad_seq"])

    def test_tampered_middle_record_caught_at_that_seq(self):
        _write_log(self.path)
        with open(self.path) as f:
            lines = f.readlines()
        record = json.loads(lines[2])
        record["payload"]["text"] = "step 2, edited later"
        lines[2] = json.dumps(record, sort_keys=True) + "\n"
        with open(self.path, "w") as f:
            f.writelines(lines)
        result = verify(self.path)
        self.assertFalse(result["ok"])
        self.assertEqual(result["first_bad_seq"], 2)

    def test_tampered_hash_field_caught(self):
        _write_log(self.path)
        with open(self.path) as f:
            lines = f.readlines()
        record = json.loads(lines[1])
        record["hash"] = "0" * 64
        lines[1] = json.dumps(record, sort_keys=True) + "\n"
        with open(self.path, "w") as f:
            f.writelines(lines)
        result = verify(self.path)
        self.assertFalse(result["ok"])
        self.assertEqual(result["first_bad_seq"], 1)

    def test_truncated_log_caught_by_summary_count(self):
        _write_log(self.path)
        with open(self.path) as f:
            lines = f.readlines()
        # Delete a middle record and keep the summary: the record after
        # the gap now sits at the wrong position, so its seq no longer
        # matches, and the summary's event count is off too.
        del lines[3]
        with open(self.path, "w") as f:
            f.writelines(lines)
        result = verify(self.path)
        self.assertFalse(result["ok"])
        # The chain breaks at the record after the deletion (seq 4 was
        # at position 3, and its prev_hash points at the deleted record).
        self.assertEqual(result["first_bad_seq"], 3)

    def test_torn_final_line_caught(self):
        _write_log(self.path)
        with open(self.path, "rb") as f:
            data = f.read()
        # Chop the file mid-line, as a crash during a write would.
        cut = data.rfind(b"\n", 0, len(data) - 10)
        with open(self.path, "wb") as f:
            f.write(data[:cut + 5])
        result = verify(self.path)
        self.assertFalse(result["ok"])
        self.assertIsNotNone(result["first_bad_seq"])

    def test_missing_file(self):
        result = verify(os.path.join(self.tmp.name, "nope.jsonl"))
        self.assertFalse(result["ok"])
        self.assertIn("not found", result["error"])


if __name__ == "__main__":
    unittest.main()
