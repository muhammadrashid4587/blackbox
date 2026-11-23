"""Session aggregates: per-session totals are computed from the
records, read-only, and the CLI prints them as a report or JSON."""

import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout

from blackbox import Recorder
from blackbox.__main__ import main
from blackbox.aggregates import aggregates_log, session_aggregates


def _record(seq, event_type, payload, tokens=None, ts="2026-11-22T10:00:00Z"):
    return {"seq": seq, "ts": ts, "type": event_type, "payload": payload,
            "tokens": tokens, "prev_hash": "0" * 64, "hash": "0" * 64}


class SessionAggregatesTest(unittest.TestCase):
    def setUp(self):
        self.records = [
            _record(0, "model_call", {"model": "gpt-5", "prompt": "plan"},
                    {"input": 1000, "output": 500}, "2026-11-22T10:00:00Z"),
            _record(1, "tool_call", {"tool": "search", "args": {"q": "x"}},
                    None, "2026-11-22T10:01:00Z"),
            _record(2, "tool_result", {"tool": "search", "result": "ok"},
                    None, "2026-11-22T10:01:30Z"),
            _record(3, "tool_result", {"tool": "fetch",
                                       "result": "error: 404"},
                    None, "2026-11-22T10:04:00Z"),
            _record(4, "model_call", {"model": "unknown-model",
                                      "prompt": "p"},
                    {"input": 9000, "output": 9000}, "2026-11-22T10:04:30Z"),
            _record(5, "model_call", {"model": "gpt-5", "prompt": "small"},
                    50, "2026-11-22T10:05:00Z"),
            _record(6, "summary", {"events": 6, "total_tokens": 19550},
                    None, "2026-11-22T10:06:00Z"),
        ]

    def test_steps_exclude_summary_and_duration_is_first_to_last(self):
        agg = session_aggregates(self.records)
        self.assertEqual(agg["steps"], 6)
        self.assertEqual(agg["started"], "2026-11-22T10:00:00Z")
        self.assertEqual(agg["ended"], "2026-11-22T10:05:00Z")
        self.assertEqual(agg["duration_seconds"], 300.0)

    def test_by_type_and_by_tool_counts(self):
        agg = session_aggregates(self.records)
        self.assertEqual(agg["by_type"],
                         {"model_call": 3, "tool_call": 1,
                          "tool_result": 2})
        self.assertEqual(agg["by_tool"], {"search": 1})

    def test_by_model_costs_priced_calls_and_never_guesses_unpriced(self):
        agg = session_aggregates(self.records)
        gpt5 = agg["by_model"]["gpt-5"]
        self.assertEqual(gpt5["calls"], 2)
        self.assertEqual(gpt5["input_tokens"], 1000)
        self.assertEqual(gpt5["output_tokens"], 500)
        self.assertEqual(gpt5["tokens"], 50)
        # 1000 in at $1.25/1M + 500 out at $10/1M = $0.00625; the
        # plain-int 50 tokens are counted but not priced.
        self.assertEqual(gpt5["cost_usd"], 0.00625)
        unknown = agg["by_model"]["unknown-model"]
        self.assertEqual(unknown["cost_usd"], None)
        self.assertEqual(agg["total_tokens"], 19550)
        self.assertEqual(agg["total_cost_usd"], 0.00625)
        # Unpriced: the unknown model, and the gpt-5 call with a plain
        # int token count (no input/output split to price).
        self.assertEqual(agg["unpriced_model_calls"], 2)

    def test_errors_list_the_error_seqs(self):
        agg = session_aggregates(self.records)
        self.assertEqual(agg["errors"], 1)
        self.assertEqual(agg["error_seqs"], [3])

    def test_slowest_step_is_the_longest_gap_to_the_next_step(self):
        agg = session_aggregates(self.records)
        # seq 2 waits 150s for seq 3; seq 3 waits 30s; seq 0 waits 60s.
        self.assertEqual(agg["slowest_step"],
                         {"seq": 2, "type": "tool_result",
                          "latency_seconds": 150.0})

    def test_checkpoint_counts_compacted_records_and_excludes_them(self):
        records = ([_record(0, "checkpoint",
                            {"compacted_count": 40,
                             "head_hash": "ab" * 32},
                            None, "2026-11-22T09:00:00Z")]
                   + self.records)
        agg = session_aggregates(records)
        self.assertEqual(agg["steps"], 6)
        self.assertEqual(agg["compacted_count"], 40)

    def test_sampled_out_comes_from_the_summary(self):
        records = self.records[:-1] + [
            _record(6, "summary", {"events": 6, "total_tokens": 19550,
                                   "sampled_out": 3},
                    None, "2026-11-22T10:06:00Z")]
        agg = session_aggregates(records)
        self.assertEqual(agg["sampled_out"], 3)

    def test_empty_log_has_no_duration_or_slowest_step(self):
        agg = session_aggregates([])
        self.assertEqual(agg["steps"], 0)
        self.assertIsNone(agg["started"])
        self.assertIsNone(agg["duration_seconds"])
        self.assertIsNone(agg["slowest_step"])
        self.assertEqual(agg["total_cost_usd"], 0.0)


class AggregatesLogTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "run.jsonl")
        with Recorder(self.path, task="test") as rec:
            rec.record("model_call", {"model": "gpt-5", "prompt": "plan"},
                       tokens={"input": 1000, "output": 500})
            rec.record("tool_call", {"tool": "search",
                                     "args": {"query": "x"}})
            rec.record("tool_result", {"tool": "search",
                                       "result": "error: timeout"})

    def tearDown(self):
        self.tmp.cleanup()

    def test_report_names_steps_tokens_cost_and_error(self):
        out = io.StringIO()
        with redirect_stdout(out):
            agg = aggregates_log(self.path)
        text = out.getvalue()
        self.assertEqual(agg["steps"], 3)
        self.assertIn("session: 3 steps", text)
        self.assertIn("tokens: 1500", text)
        self.assertIn("cost: $0.006250", text)
        self.assertIn("errors: 1", text)

    def test_json_output_round_trips(self):
        out = io.StringIO()
        with redirect_stdout(out):
            aggregates_log(self.path, as_json=True)
        data = json.loads(out.getvalue())
        self.assertEqual(data["steps"], 3)
        self.assertEqual(data["by_model"]["gpt-5"]["cost_usd"], 0.00625)

    def test_cli_aggregates_command(self):
        out = io.StringIO()
        with redirect_stdout(out):
            self.assertEqual(main(["aggregates", self.path]), 0)
        self.assertIn("session: 3 steps", out.getvalue())

    def test_cli_aggregates_json(self):
        out = io.StringIO()
        with redirect_stdout(out):
            self.assertEqual(
                main(["aggregates", self.path, "--json"]), 0)
        data = json.loads(out.getvalue())
        self.assertEqual(data["steps"], 3)

    def test_works_on_sqlite_store(self):
        db_path = os.path.join(self.tmp.name, "run.db")
        with Recorder(db_path, task="test") as rec:
            rec.record("model_call", {"model": "gpt-5", "prompt": "plan"},
                       tokens={"input": 1000, "output": 500})
        out = io.StringIO()
        with redirect_stdout(out):
            agg = aggregates_log(db_path)
        self.assertEqual(agg["steps"], 1)
        self.assertEqual(agg["total_cost_usd"], 0.00625)


if __name__ == "__main__":
    unittest.main()
