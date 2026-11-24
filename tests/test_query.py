"""Query: filters combine with AND, cost is per priced model call,
the summary is never a match unless asked for, --json and --count
print machine-readable output."""

import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from datetime import datetime, timedelta, timezone

from blackbox import Recorder
from blackbox.__main__ import main
from blackbox.query import _is_error, _record_cost, query_log, query_records

NOW = datetime(2026, 11, 22, 10, 6, tzinfo=timezone.utc)


def _record(seq, event_type, payload, tokens=None, ts="2026-11-22T10:00:00Z"):
    return {"seq": seq, "ts": ts, "type": event_type, "payload": payload,
            "tokens": tokens, "prev_hash": "0" * 64, "hash": "0" * 64}


class QueryRecordsTest(unittest.TestCase):
    def setUp(self):
        self.records = [
            _record(0, "model_call", {"model": "gpt-5", "prompt": "plan"},
                    {"input": 1000, "output": 500}, "2026-11-22T10:00:00Z"),
            _record(1, "tool_call", {"tool": "search", "args": {"q": "x"}},
                    None, "2026-11-22T10:01:00Z"),
            _record(2, "tool_result", {"tool": "search", "result": "ok"},
                    None, "2026-11-22T10:02:00Z"),
            _record(3, "tool_result", {"tool": "fetch", "result": "error: 404"},
                    None, "2026-11-22T10:03:00Z"),
            _record(4, "model_call", {"model": "unknown-model", "prompt": "p"},
                    {"input": 9000, "output": 9000}, "2026-11-22T10:04:00Z"),
            _record(5, "model_call", {"model": "gpt-5", "prompt": "small"},
                    50, "2026-11-22T10:05:00Z"),
            _record(6, "summary", {"events": 6, "total_tokens": 19550},
                    None, "2026-11-22T10:06:00Z"),
        ]

    def test_no_filters_matches_every_step_but_not_summary(self):
        matched = query_records(self.records)
        self.assertEqual([r["seq"] for r in matched], [0, 1, 2, 3, 4, 5])

    def test_type_filter(self):
        matched = query_records(self.records, types=["tool_call"])
        self.assertEqual([r["seq"] for r in matched], [1])

    def test_type_summary_includes_summary(self):
        matched = query_records(self.records, types=["summary"])
        self.assertEqual([r["seq"] for r in matched], [6])

    def test_model_filter_case_insensitive_and_only_model_calls(self):
        matched = query_records(self.records, model="GPT-5")
        self.assertEqual([r["seq"] for r in matched], [0, 5])

    def test_tool_filter_matches_call_and_result(self):
        matched = query_records(self.records, tool="SEARCH")
        self.assertEqual([r["seq"] for r in matched], [1, 2])

    def test_min_tokens(self):
        matched = query_records(self.records, min_tokens=1000)
        self.assertEqual([r["seq"] for r in matched], [0, 4])

    def test_min_cost_only_priced_model_calls(self):
        # gpt-5 at 1000 in / 500 out = $0.00625; the unknown model and
        # the plain-int token count have no cost and never match.
        matched = query_records(self.records, min_cost=0.005)
        self.assertEqual([r["seq"] for r in matched], [0])
        self.assertEqual(query_records(self.records, min_cost=0.01), [])

    def test_errors_only(self):
        matched = query_records(self.records, errors_only=True)
        self.assertEqual([r["seq"] for r in matched], [3])

    def test_since_until(self):
        matched = query_records(self.records, since="2026-11-22T10:02:00Z",
                                until="2026-11-22T10:04:00Z")
        self.assertEqual([r["seq"] for r in matched], [2, 3, 4])

    def test_last_keeps_only_recent_steps(self):
        # now 10:06, last 5m -> bound 10:01, so seqs 1-5; the
        # summary (10:06) is still excluded unless asked for.
        matched = query_records(self.records, last="5m", now=NOW)
        self.assertEqual([r["seq"] for r in matched], [1, 2, 3, 4, 5])
        matched = query_records(self.records, last="1m", now=NOW)
        self.assertEqual([r["seq"] for r in matched], [5])

    def test_last_accepts_timedelta(self):
        matched = query_records(self.records, last=timedelta(minutes=5),
                                now=NOW)
        self.assertEqual([r["seq"] for r in matched], [1, 2, 3, 4, 5])

    def test_last_rejects_combination_with_since_or_until(self):
        for kwargs in ({"since": "2026-11-22T10:00:00Z"},
                       {"until": "2026-11-22T10:06:00Z"}):
            with self.assertRaises(ValueError):
                query_records(self.records, last="5m", now=NOW, **kwargs)

    def test_last_rejects_bad_duration(self):
        for bad in ("10", "1.5m", "0m", "", "next week"):
            with self.assertRaises(ValueError):
                query_records(self.records, last=bad, now=NOW)

    def test_last_fractional_seconds_compare_as_datetimes(self):
        # A record stamped half a second after a whole-second bound
        # matches; a plain string comparison would get this wrong
        # ("." sorts before "Z").
        records = [_record(0, "tool_call", {"tool": "search"}, None,
                           "2026-11-22T10:05:00.500Z")]
        matched = query_records(records, last="1m",
                                now=datetime(2026, 11, 22, 10, 6,
                                             tzinfo=timezone.utc))
        self.assertEqual([r["seq"] for r in matched], [0])

    def test_last_excludes_summary_unless_type_summary(self):
        matched = query_records(self.records, last="1m", now=NOW,
                                types=["summary"])
        self.assertEqual([r["seq"] for r in matched], [6])
        matched = query_records(self.records, last="1h", now=NOW)
        self.assertNotIn(6, [r["seq"] for r in matched])

    def test_last_combines_with_other_filters(self):
        # seq 0 is gpt-5 but older than the window; seq 4 is in the
        # window but an unknown model — only seq 5 passes both.
        matched = query_records(self.records, last="5m", now=NOW,
                                model="gpt-5")
        self.assertEqual([r["seq"] for r in matched], [5])

    def test_last_ignores_records_with_unusable_timestamps(self):
        records = [_record(0, "tool_call", {"tool": "search"}, None,
                           "not-a-timestamp"),
                   _record(1, "tool_call", {"tool": "search"}, None,
                           "2026-11-22T10:05:00")]
        matched = query_records(records, last="1d", now=NOW)
        self.assertEqual(matched, [])

    def test_filters_combine_with_and(self):
        matched = query_records(self.records, model="gpt-5", min_tokens=100)
        self.assertEqual([r["seq"] for r in matched], [0])

    def test_record_cost(self):
        self.assertAlmostEqual(_record_cost(self.records[0]), 0.00625)
        self.assertIsNone(_record_cost(self.records[4]))  # unknown model
        self.assertIsNone(_record_cost(self.records[5]))  # int tokens
        self.assertIsNone(_record_cost(self.records[1]))  # not a model call

    def test_is_error(self):
        self.assertTrue(_is_error(_record(0, "tool_result",
                                          {"tool": "t", "result": "Error: boom"})))
        self.assertTrue(_is_error(_record(0, "note", {"error": "boom"})))
        self.assertTrue(_is_error(_record(0, "note", {"status": "error"})))
        self.assertFalse(_is_error(_record(0, "tool_result",
                                           {"tool": "t", "result": "ok"})))


class QueryLogTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "run.jsonl")
        with Recorder(self.path) as rec:
            rec.record("model_call", {"model": "gpt-5", "prompt": "plan"},
                       tokens={"input": 1000, "output": 500})
            rec.record("tool_call", {"tool": "search", "args": {"q": "x"}})
            rec.record("tool_result", {"tool": "search", "result": "error: 404"})
            rec.record("model_call", {"model": "gpt-5", "prompt": "small"},
                       tokens=50)

    def tearDown(self):
        self.tmp.cleanup()

    def test_query_log_prints_matches_and_footer(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            matched = query_log(self.path, model="gpt-5")
        self.assertEqual(len(matched), 2)
        out = buf.getvalue()
        self.assertIn("plan", out)
        self.assertIn("small", out)
        self.assertNotIn("search", out)
        self.assertIn("matched: 2 of 4 records, 1550 tokens", out)
        self.assertIn("$0.006250 cost", out)
        self.assertIn("(1 unpriced)", out)

    def test_query_log_json_prints_raw_records(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            matched = query_log(self.path, errors_only=True, as_json=True)
        self.assertEqual(len(matched), 1)
        parsed = json.loads(buf.getvalue().strip())
        self.assertEqual(parsed["type"], "tool_result")
        self.assertEqual(parsed["payload"]["tool"], "search")

    def test_query_log_count_only(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            matched = query_log(self.path, tool="search", count_only=True)
        self.assertEqual(len(matched), 2)
        self.assertEqual(buf.getvalue().strip(), "2")

    def test_cli_query(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            code = main(["query", self.path, "--model", "gpt-5",
                         "--min-cost", "0.005"])
        self.assertEqual(code, 0)
        out = buf.getvalue()
        self.assertIn("plan", out)
        self.assertNotIn("small", out)
        self.assertIn("matched: 1 of 4 records", out)

    def test_cli_query_count(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            code = main(["query", self.path, "--errors-only", "--count"])
        self.assertEqual(code, 0)
        self.assertEqual(buf.getvalue().strip(), "1")

    def test_cli_query_last(self):
        # The log was just recorded, so --last 1d matches every step
        # and --last 0s is an error; --last with --since is an error.
        buf = io.StringIO()
        with redirect_stdout(buf):
            code = main(["query", self.path, "--last", "1d", "--count"])
        self.assertEqual(code, 0)
        self.assertEqual(buf.getvalue().strip(), "4")
        with redirect_stdout(io.StringIO()):
            self.assertEqual(
                main(["query", self.path, "--last", "10", "--count"]), 2)
            self.assertEqual(
                main(["query", self.path, "--last", "1d", "--since",
                      "2026-11-22T10:00:00Z", "--count"]), 2)


if __name__ == "__main__":
    unittest.main()
