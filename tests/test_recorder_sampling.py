"""Recorder sampling: keep/drop choice, honest summary, chain intact.

With sample_rate < 1.0 each step is kept with that probability.
Dropped steps are never written and never hashed — the chain covers
exactly what is in the log — but the summary records the rate and
how many steps were dropped, so a sampled log says what it skipped.
"""

import json
import os
import tempfile
import unittest

from blackbox import Recorder
from blackbox.verify import verify


class SamplingTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "run.jsonl")

    def tearDown(self):
        self.tmp.cleanup()

    def _read_records(self, path=None):
        with open(path or self.path) as f:
            return [json.loads(line) for line in f if line.strip()]

    def test_invalid_rates_rejected(self):
        for bad in (0, -0.5, 1.5):
            with self.assertRaises(ValueError):
                Recorder(self.path, sample_rate=bad)

    def test_default_rate_keeps_everything(self):
        with Recorder(self.path) as rec:
            seqs = [rec.record("note", {"text": "step %d" % i})
                    for i in range(5)]
        self.assertEqual(seqs, [0, 1, 2, 3, 4])
        records = self._read_records()
        self.assertEqual(len(records), 6)  # 5 steps + summary
        summary = records[-1]["payload"]
        self.assertNotIn("sample_rate", summary)
        self.assertNotIn("sampled_out", summary)

    def test_dropped_step_returns_none(self):
        # Random(0) draws 0.844, 0.758, 0.421, ... so rate 0.5 drops
        # the first two steps and keeps the third as seq 0.
        with Recorder(self.path, sample_rate=0.5, sample_seed=0) as rec:
            seqs = [rec.record("note", {"text": "step %d" % i})
                    for i in range(3)]
        self.assertEqual(seqs, [None, None, 0])
        records = self._read_records()
        self.assertEqual(records[0]["payload"]["text"], "step 2")
        self.assertEqual(records[0]["seq"], 0)

    def test_seeded_sampling_is_deterministic(self):
        def run(path):
            with Recorder(path, sample_rate=0.5, sample_seed=7) as rec:
                return [rec.record("note", {"text": "step %d" % i})
                        for i in range(20)]

        seqs_a = run(os.path.join(self.tmp.name, "a.jsonl"))
        seqs_b = run(os.path.join(self.tmp.name, "b.jsonl"))
        self.assertEqual(seqs_a, seqs_b)
        kept = [s for s in seqs_a if s is not None]
        dropped = 20 - len(kept)
        self.assertGreater(len(kept), 0)
        self.assertGreater(dropped, 0)

    def test_sampled_log_verifies_and_summary_is_honest(self):
        with Recorder(self.path, sample_rate=0.5, sample_seed=7,
                      task="sampled") as rec:
            kept_tokens = 0
            for i in range(20):
                seq = rec.record("model_call",
                                 {"model": "m", "prompt": "p%d" % i},
                                 tokens=10)
                if seq is not None:
                    kept_tokens += 10
        result = verify(self.path)
        self.assertTrue(result["ok"], result)
        records = self._read_records()
        summary = records[-1]["payload"]
        kept = [r for r in records if r["type"] != "summary"]
        self.assertEqual(summary["events"], len(kept))
        self.assertEqual(summary["sample_rate"], 0.5)
        self.assertEqual(summary["sampled_out"], 20 - len(kept))
        self.assertEqual(summary["total_tokens"], kept_tokens)
        # Seq numbers are contiguous over the kept steps only.
        self.assertEqual([r["seq"] for r in kept],
                         list(range(len(kept))))

    def test_sampling_works_on_sqlite_backend(self):
        path = os.path.join(self.tmp.name, "run.db")
        with Recorder(path, sample_rate=0.5, sample_seed=7) as rec:
            seqs = [rec.record("note", {"text": "step %d" % i})
                    for i in range(10)]
        result = verify(path)
        self.assertTrue(result["ok"], result)
        kept = [s for s in seqs if s is not None]
        self.assertEqual(result["records"], len(kept) + 1)

    def test_resume_after_sampling_keeps_chain(self):
        with Recorder(self.path, sample_rate=0.5, sample_seed=7) as rec:
            for i in range(10):
                rec.record("note", {"text": "step %d" % i})
        with Recorder(self.path) as rec:
            rec.record("note", {"text": "resumed"})
        result = verify(self.path)
        self.assertTrue(result["ok"], result)
        records = self._read_records()
        self.assertEqual(records[-2]["payload"]["text"], "resumed")
        self.assertEqual(records[-1]["type"], "summary")


if __name__ == "__main__":
    unittest.main()
