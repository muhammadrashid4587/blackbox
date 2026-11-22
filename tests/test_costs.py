"""Cost tracking per model: pricing table, summary totals, resume."""

import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout

from blackbox import Recorder, cost_for, price_for, register_pricing
from blackbox.replay import load_records, replay
from blackbox.verify import verify


class PricingTest(unittest.TestCase):
    def test_known_model(self):
        # gpt-5: $1.25 in / $10.00 out per 1M tokens.
        self.assertEqual(cost_for("gpt-5", 1_000_000, 500_000), 6.25)
        self.assertEqual(price_for("gpt-5"), (1.25, 10.00))

    def test_unknown_model_costs_none_never_a_guess(self):
        self.assertIsNone(cost_for("some-future-model", 1000, 1000))
        self.assertIsNone(price_for("some-future-model"))

    def test_name_normalization(self):
        # Dots, case, and a snapshot date all resolve to the table entry.
        self.assertEqual(cost_for("GPT-5", 1_000_000, 0), 1.25)
        self.assertEqual(cost_for("gpt-5-2025-08-07", 0, 1_000_000), 10.0)
        self.assertEqual(cost_for("claude-sonnet-4.5", 1_000_000, 0), 3.0)
        self.assertEqual(cost_for("claude-sonnet-4-5", 0, 1_000_000), 15.0)

    def test_register_pricing_overrides_and_adds(self):
        register_pricing("gpt-5", 2.00, 20.00)
        try:
            self.assertEqual(cost_for("gpt-5", 1_000_000, 0), 2.00)
        finally:
            register_pricing("gpt-5", 1.25, 10.00)
        register_pricing("my-local-model", 0.10, 0.20)
        self.assertEqual(cost_for("my-local-model", 1_000_000, 1_000_000), 0.30)

    def test_negative_tokens_rejected(self):
        with self.assertRaises(ValueError):
            cost_for("gpt-5", -1, 0)
        with self.assertRaises(ValueError):
            register_pricing("gpt-5", -1.0, 1.0)


class RecorderCostTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "run.jsonl")

    def tearDown(self):
        self.tmp.cleanup()

    def _summary(self, path=None):
        with open(path or self.path) as f:
            records = [json.loads(line) for line in f]
        return records[-1]["payload"]

    def test_summary_cost_by_model(self):
        with Recorder(self.path, task="t", model="gpt-5") as rec:
            rec.record("model_call", {"model": "gpt-5", "prompt": "a"},
                       tokens={"input": 1_000_000, "output": 500_000})
            rec.record("model_call", {"model": "claude-haiku-4-5",
                                      "prompt": "b"},
                       tokens={"input": 2_000_000, "output": 0})
        summary = self._summary()
        # gpt-5: 1.25 + 5.00 = 6.25; haiku: 2.00. Total 8.25.
        self.assertEqual(summary["cost_usd"], 8.25)
        self.assertEqual(summary["cost_by_model"]["gpt-5"]["cost_usd"], 6.25)
        self.assertEqual(summary["cost_by_model"]["gpt-5"]["input_tokens"],
                         1_000_000)
        self.assertEqual(summary["cost_by_model"]["gpt-5"]["output_tokens"],
                         500_000)
        self.assertEqual(summary["cost_by_model"]["claude-haiku-4-5"]["cost_usd"],
                         2.00)

    def test_model_falls_back_to_recorder_default(self):
        with Recorder(self.path, model="gpt-5-mini") as rec:
            rec.record("model_call", {"prompt": "no model in payload"},
                       tokens={"input": 1_000_000, "output": 1_000_000})
        summary = self._summary()
        # mini: 0.25 + 2.00 = 2.25.
        self.assertEqual(summary["cost_usd"], 2.25)
        self.assertIn("gpt-5-mini", summary["cost_by_model"])

    def test_unsplit_tokens_counted_but_not_priced(self):
        with Recorder(self.path, model="gpt-5") as rec:
            rec.record("model_call", {"model": "gpt-5", "prompt": "a"},
                       tokens=500)
        summary = self._summary()
        entry = summary["cost_by_model"]["gpt-5"]
        self.assertEqual(entry["tokens"], 500)
        self.assertIsNone(entry["cost_usd"])
        self.assertEqual(summary["cost_usd"], 0.0)

    def test_unknown_model_costs_none(self):
        with Recorder(self.path) as rec:
            rec.record("model_call", {"model": "demo-model", "prompt": "a"},
                       tokens={"input": 100, "output": 100})
        summary = self._summary()
        self.assertIsNone(summary["cost_by_model"]["demo-model"]["cost_usd"])
        self.assertEqual(summary["cost_usd"], 0.0)

    def test_non_model_events_do_not_accrue_cost(self):
        with Recorder(self.path, model="gpt-5") as rec:
            rec.record("tool_call", {"tool": "search", "args": {}},
                       tokens={"input": 1_000_000, "output": 1_000_000})
        summary = self._summary()
        self.assertEqual(summary["cost_by_model"], {})
        self.assertEqual(summary["cost_usd"], 0.0)

    def test_resume_rebuilds_cost(self):
        with Recorder(self.path) as rec:
            rec.record("model_call", {"model": "gpt-5", "prompt": "a"},
                       tokens={"input": 1_000_000, "output": 0})
        with Recorder(self.path) as rec:
            rec.record("model_call", {"model": "gpt-5", "prompt": "b"},
                       tokens={"input": 0, "output": 1_000_000})
        summary = self._summary()
        self.assertEqual(summary["cost_usd"], 11.25)  # 1.25 + 10.00
        self.assertEqual(summary["cost_by_model"]["gpt-5"]["input_tokens"],
                         1_000_000)
        self.assertEqual(summary["cost_by_model"]["gpt-5"]["output_tokens"],
                         1_000_000)
        result = verify(self.path)
        self.assertTrue(result["ok"], result)

    def test_cost_on_sqlite_backend(self):
        db = os.path.join(self.tmp.name, "run.db")
        with Recorder(db) as rec:
            rec.record("model_call", {"model": "gemini-2-5-flash",
                                      "prompt": "a"},
                       tokens={"input": 1_000_000, "output": 1_000_000})
        with open(db, "rb") as f:
            self.assertTrue(f.read(16).startswith(b"SQLite format 3"))
        result = verify(db)
        self.assertTrue(result["ok"], result)
        records = load_records(db)
        summary = records[-1]["payload"]
        # flash: 0.30 + 2.50 = 2.80.
        self.assertEqual(summary["cost_usd"], 2.80)

    def test_replay_prints_cost(self):
        with Recorder(self.path) as rec:
            rec.record("model_call", {"model": "gpt-5-nano", "prompt": "a"},
                       tokens={"input": 1_000_000, "output": 1_000_000})
        out = io.StringIO()
        with redirect_stdout(out):
            replay(self.path)
        text = out.getvalue()
        # nano: 0.05 + 0.40 = 0.45.
        self.assertIn("cost: $0.450000", text)
        self.assertIn("gpt-5-nano", text)

    def test_summary_still_verifies_with_cost_fields(self):
        with Recorder(self.path) as rec:
            rec.record("model_call", {"model": "gpt-5", "prompt": "a"},
                       tokens={"input": 100, "output": 50})
        result = verify(self.path)
        self.assertTrue(result["ok"], result)
        self.assertEqual(result["records"], 2)  # 1 step + summary


if __name__ == "__main__":
    unittest.main()
