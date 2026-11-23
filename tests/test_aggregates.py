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
from blackbox.aggregates import (aggregates_log, cost_totals,
                                  cost_totals_log, session_aggregates)


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


class CostTotalsTest(unittest.TestCase):
    def setUp(self):
        # seq 0: an orphan tool_call with no model call before it.
        # seq 1 -> seq 2: gpt-5 calls search. seq 4 -> seq 5: an
        # unknown model calls fetch (unpriced). seq 6: a plain-int
        # gpt-5 call that invokes no tool. seq 7 -> seqs 8, 9: one
        # gpt-5-mini call invoking two tools at once.
        self.records = [
            _record(0, "tool_call", {"tool": "lookup", "args": {}},
                    None, "2026-11-22T10:00:00Z"),
            _record(1, "model_call", {"model": "gpt-5", "prompt": "plan"},
                    {"input": 1000, "output": 500},
                    "2026-11-22T10:01:00Z"),
            _record(2, "tool_call", {"tool": "search",
                                     "args": {"q": "x"}},
                    None, "2026-11-22T10:01:30Z"),
            _record(3, "tool_result", {"tool": "search", "result": "ok"},
                    None, "2026-11-22T10:02:00Z"),
            _record(4, "model_call", {"model": "unknown-model",
                                      "prompt": "p"},
                    {"input": 9000, "output": 9000},
                    "2026-11-22T10:03:00Z"),
            _record(5, "tool_call", {"tool": "fetch",
                                     "args": {"url": "u"}},
                    None, "2026-11-22T10:03:30Z"),
            _record(6, "model_call", {"model": "gpt-5", "prompt": "small"},
                    50, "2026-11-22T10:04:00Z"),
            _record(7, "model_call", {"model": "gpt-5-mini",
                                      "prompt": "both"},
                    {"input": 4000, "output": 1000},
                    "2026-11-22T10:05:00Z"),
            _record(8, "tool_call", {"tool": "search",
                                     "args": {"q": "y"}},
                    None, "2026-11-22T10:05:30Z"),
            _record(9, "tool_call", {"tool": "fetch",
                                     "args": {"url": "v"}},
                    None, "2026-11-22T10:06:00Z"),
            _record(10, "summary", {"events": 10, "total_tokens": 24550},
                    None, "2026-11-22T10:07:00Z"),
        ]

    def test_per_model_reuses_model_usage_and_never_guesses(self):
        totals = cost_totals(self.records)
        gpt5 = totals["per_model"]["gpt-5"]
        self.assertEqual(gpt5["calls"], 2)
        self.assertEqual(gpt5["input_tokens"], 1000)
        self.assertEqual(gpt5["output_tokens"], 500)
        self.assertEqual(gpt5["tokens"], 50)
        self.assertEqual(gpt5["cost_usd"], 0.00625)
        mini = totals["per_model"]["gpt-5-mini"]
        # 4000 in at $0.25/1M + 1000 out at $2.00/1M = $0.003.
        self.assertEqual(mini["cost_usd"], 0.003)
        unknown = totals["per_model"]["unknown-model"]
        self.assertEqual(unknown["cost_usd"], None)

    def test_total_and_unpriced_agree_with_session_aggregates(self):
        totals = cost_totals(self.records)
        agg = session_aggregates(self.records)
        self.assertEqual(totals["total_cost_usd"],
                         agg["total_cost_usd"])
        self.assertEqual(totals["unpriced_model_calls"],
                         agg["unpriced_model_calls"])
        # Priced: gpt-5 $0.00625 + gpt-5-mini $0.003. Unpriced: the
        # unknown model and the plain-int gpt-5 call.
        self.assertEqual(totals["total_cost_usd"], 0.00925)
        self.assertEqual(totals["unpriced_model_calls"], 2)

    def test_per_tool_charges_the_invoking_model_call(self):
        totals = cost_totals(self.records)
        search = totals["per_tool"]["search"]
        # Called twice: once by the gpt-5 call ($0.00625), once by
        # the gpt-5-mini call ($0.003).
        self.assertEqual(search["calls"], 2)
        self.assertEqual(search["input_tokens"], 5000)
        self.assertEqual(search["output_tokens"], 1500)
        self.assertEqual(search["cost_usd"], 0.00925)
        self.assertEqual(search["unpriced_calls"], 0)

    def test_per_tool_partial_cost_reports_the_unpriced_share(self):
        totals = cost_totals(self.records)
        fetch = totals["per_tool"]["fetch"]
        # Once by the unknown model (unpriced), once by gpt-5-mini.
        self.assertEqual(fetch["calls"], 2)
        self.assertEqual(fetch["input_tokens"], 13000)
        self.assertEqual(fetch["output_tokens"], 10000)
        self.assertEqual(fetch["cost_usd"], 0.003)
        self.assertEqual(fetch["unpriced_calls"], 1)

    def test_orphan_tool_call_is_counted_but_unpriced(self):
        totals = cost_totals(self.records)
        lookup = totals["per_tool"]["lookup"]
        self.assertEqual(lookup,
                         {"calls": 1, "input_tokens": 0,
                          "output_tokens": 0, "tokens": 0,
                          "cost_usd": None, "unpriced_calls": 0})

    def test_one_call_invoking_two_tools_counts_in_full_under_both(self):
        totals = cost_totals(self.records)
        # The seq 7 call is charged in full to search and to fetch,
        # so the per-tool costs sum to more than the session total.
        per_tool_total = sum(
            (entry["cost_usd"] or 0.0)
            for entry in totals["per_tool"].values())
        self.assertGreater(per_tool_total, totals["total_cost_usd"])

    def test_default_model_prices_a_modelless_call(self):
        records = [_record(0, "model_call", {"prompt": "p"},
                           {"input": 100, "output": 50}),
                   _record(1, "tool_call", {"tool": "search",
                                            "args": {}})]
        totals = cost_totals(records, default_model="gpt-5-mini")
        # 100 in at $0.25/1M + 50 out at $2.00/1M = $0.000125.
        self.assertEqual(
            totals["per_model"]["gpt-5-mini"]["cost_usd"], 0.000125)
        self.assertEqual(totals["total_cost_usd"], 0.000125)
        self.assertEqual(totals["unpriced_model_calls"], 0)
        self.assertEqual(totals["per_tool"]["search"]["cost_usd"],
                         0.000125)

    def test_empty_log_has_empty_breakdowns(self):
        totals = cost_totals([])
        self.assertEqual(totals["per_model"], {})
        self.assertEqual(totals["per_tool"], {})
        self.assertEqual(totals["total_cost_usd"], 0.0)
        self.assertEqual(totals["unpriced_model_calls"], 0)


class CostTotalsLogTest(unittest.TestCase):
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

    def test_report_names_total_per_model_and_per_tool(self):
        out = io.StringIO()
        with redirect_stdout(out):
            totals = cost_totals_log(self.path)
        text = out.getvalue()
        self.assertEqual(totals["total_cost_usd"], 0.00625)
        self.assertEqual(totals["per_tool"]["search"]["cost_usd"],
                         0.00625)
        self.assertIn("cost totals: $0.006250 priced "
                      "(0 unpriced model calls)", text)
        self.assertIn("gpt-5: 1 calls, 1000 in / 500 out / 0 tokens "
                      "-> $0.006250", text)
        self.assertIn("search: 1 calls, 1000 in / 500 out / 0 tokens "
                      "-> $0.006250", text)

    def test_json_output_round_trips(self):
        out = io.StringIO()
        with redirect_stdout(out):
            cost_totals_log(self.path, as_json=True)
        data = json.loads(out.getvalue())
        self.assertEqual(data["per_model"]["gpt-5"]["cost_usd"], 0.00625)
        self.assertEqual(data["per_tool"]["search"]["cost_usd"], 0.00625)

    def test_cli_costs_command(self):
        out = io.StringIO()
        with redirect_stdout(out):
            self.assertEqual(main(["costs", self.path]), 0)
        self.assertIn("cost totals: $0.006250 priced", out.getvalue())

    def test_cli_costs_json(self):
        out = io.StringIO()
        with redirect_stdout(out):
            self.assertEqual(main(["costs", self.path, "--json"]), 0)
        data = json.loads(out.getvalue())
        self.assertEqual(data["total_cost_usd"], 0.00625)


if __name__ == "__main__":
    unittest.main()
