"""Recorder: round trip, chain linkage, resume, summary totals."""

import json
import os
import tempfile
import unittest

from blackbox import Recorder
from blackbox.verify import verify


class RecorderTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "run.jsonl")

    def tearDown(self):
        self.tmp.cleanup()

    def test_round_trip_verifies(self):
        with Recorder(self.path, task="t", model="m") as rec:
            rec.record("model_call", {"model": "m", "prompt": "hi"}, tokens=10)
            rec.record("tool_call", {"tool": "search", "args": {"q": "x"}})
            rec.record("tool_result", {"tool": "search", "result": "ok"})
        result = verify(self.path)
        self.assertTrue(result["ok"], result)
        self.assertEqual(result["records"], 4)  # 3 steps + summary

    def test_hash_chain_links(self):
        with Recorder(self.path) as rec:
            rec.record("note", {"text": "one"})
            rec.record("note", {"text": "two"})
        with open(self.path) as f:
            records = [json.loads(line) for line in f]
        self.assertEqual(records[0]["prev_hash"], "0" * 64)
        self.assertEqual(records[1]["prev_hash"], records[0]["hash"])
        self.assertEqual(records[2]["prev_hash"], records[1]["hash"])
        self.assertEqual([r["seq"] for r in records], [0, 1, 2])

    def test_summary_totals(self):
        with Recorder(self.path, task="t", model="m") as rec:
            rec.record("model_call", {"model": "m", "prompt": "a"}, tokens=100)
            rec.record("model_call", {"model": "m", "prompt": "b"},
                       tokens={"input": 50, "output": 70})
            rec.record("note", {"text": "no tokens here"})
        with open(self.path) as f:
            records = [json.loads(line) for line in f]
        summary = records[-1]
        self.assertEqual(summary["type"], "summary")
        self.assertEqual(summary["payload"]["events"], 3)
        self.assertEqual(summary["payload"]["total_tokens"], 220)
        self.assertEqual(summary["payload"]["task"], "t")

    def test_record_after_close_raises(self):
        rec = Recorder(self.path)
        rec.record("note", {"text": "x"})
        rec.close()
        with self.assertRaises(RuntimeError):
            rec.record("note", {"text": "too late"})

    def test_resume_continues_chain(self):
        with Recorder(self.path) as rec:
            rec.record("note", {"text": "first run"})
        with Recorder(self.path) as rec:
            rec.record("note", {"text": "second run"})
        result = verify(self.path)
        self.assertTrue(result["ok"], result)
        with open(self.path) as f:
            records = [json.loads(line) for line in f]
        # The first run's summary was dropped on resume and replaced by
        # the second run's summary, so the log has exactly one summary.
        self.assertEqual([r["seq"] for r in records], [0, 1, 2])
        self.assertEqual(records[-1]["type"], "summary")
        self.assertEqual(records[-1]["payload"]["events"], 2)


if __name__ == "__main__":
    unittest.main()
