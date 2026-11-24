"""OpenTelemetry metrics export: per-model calls, tokens, latency,
cost, and per-tool call counts as OTLP Sum metrics."""

import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from datetime import datetime, timezone

from blackbox import Recorder
from blackbox.__main__ import main
from blackbox.costs import cost_for
from blackbox.otel import (export_otlp_metrics, export_otlp_metrics_file,
                           records_to_metrics)

MODEL = "claude-sonnet-4-5"


class OtelMetricsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "run.jsonl")
        with Recorder(self.path, task="t", model="m") as rec:
            rec.record("model_call", {"model": MODEL, "prompt": "hi"},
                       tokens={"input": 100, "output": 50})
            rec.record("model_call", {"model": MODEL, "prompt": "again"},
                       tokens={"input": 200, "output": 25})
            rec.record("model_call", {"model": "mystery-model-9",
                                      "prompt": "?"}, tokens={"input": 10,
                                                             "output": 5})
            rec.record("tool_call", {"tool": "search", "args": {"q": "x"}})
            rec.record("tool_call", {"tool": "search", "args": {"q": "y"}})
            rec.record("tool_call", {"tool": "fetch", "args": {}})
            rec.record("tool_result", {"tool": "search", "result": "ok"})
        with open(self.path) as f:
            self.records = [json.loads(line) for line in f]

    def tearDown(self):
        self.tmp.cleanup()

    def _metrics(self):
        data = export_otlp_metrics(self.path)
        return data["resourceMetrics"][0]["scopeMetrics"][0]["metrics"]

    @staticmethod
    def _points(metric):
        out = {}
        for point in metric["sum"]["dataPoints"]:
            attrs = {a["key"]: a["value"].get("stringValue", "")
                     for a in point["attributes"]}
            key = attrs.get("gen_ai.request.model") or attrs.get("tool.name")
            value = point.get("asDouble")
            if value is None:
                value = int(point["asInt"])
            out[key] = value
        return out

    def _by_name(self):
        return {m["name"]: m for m in self._metrics()}

    def test_model_call_counts_per_model(self):
        by_name = self._by_name()
        self.assertEqual(self._points(by_name["blackbox.model_calls"]),
                         {MODEL: 2, "mystery-model-9": 1})

    def test_token_totals_per_model(self):
        by_name = self._by_name()
        self.assertEqual(
            self._points(by_name["blackbox.model_input_tokens"]),
            {MODEL: 300, "mystery-model-9": 10})
        self.assertEqual(
            self._points(by_name["blackbox.model_output_tokens"]),
            {MODEL: 75, "mystery-model-9": 5})

    def test_cost_only_for_priced_models(self):
        by_name = self._by_name()
        expected = (cost_for(MODEL, 100, 50) + cost_for(MODEL, 200, 25))
        self.assertEqual(self._points(by_name["blackbox.model_cost_usd"]),
                         {MODEL: round(expected, 6)})
        # No cost data point for the unpriced model: never a guess.
        self.assertNotIn("mystery-model-9",
                         self._points(by_name["blackbox.model_cost_usd"]))

    def test_latency_present_and_non_negative(self):
        by_name = self._by_name()
        latency = self._points(by_name["blackbox.model_latency_ms"])
        self.assertIn(MODEL, latency)
        self.assertIn("mystery-model-9", latency)
        for value in latency.values():
            self.assertGreaterEqual(value, 0)

    def test_tool_call_counts_per_tool(self):
        by_name = self._by_name()
        self.assertEqual(self._points(by_name["blackbox.tool_calls"]),
                         {"search": 2, "fetch": 1})

    def test_sums_are_cumulative_and_monotonic(self):
        for metric in self._metrics():
            self.assertEqual(metric["sum"]["aggregationTemporality"], 2)
            if metric["name"] != "blackbox.model_latency_ms":
                self.assertTrue(metric["sum"]["isMonotonic"])

    def test_empty_session_yields_no_metrics(self):
        empty = os.path.join(self.tmp.name, "empty.jsonl")
        open(empty, "w").close()
        self.assertEqual(records_to_metrics([]), [])
        data = export_otlp_metrics(empty)
        metrics = data["resourceMetrics"][0]["scopeMetrics"][0]["metrics"]
        self.assertEqual(metrics, [])

    def test_metrics_file_round_trip(self):
        out = os.path.join(self.tmp.name, "metrics.json")
        data = export_otlp_metrics_file(self.path, out)
        with open(out) as f:
            self.assertEqual(json.load(f), data)
        self.assertEqual(
            data["resourceMetrics"][0]["resource"]["attributes"],
            [{"key": "service.name", "value": {"stringValue": "blackbox"}}])


def _record(seq, event_type, payload, tokens=None, ts="2026-11-22T10:00:00Z"):
    return {"seq": seq, "ts": ts, "type": event_type, "payload": payload,
            "tokens": tokens, "prev_hash": "0" * 64, "hash": "0" * 64}


class OtelMetricsSinceUntilTest(unittest.TestCase):
    """--since / --until (and --last) narrow the metrics export to a
    window.

    Same fixture as the aggregates window tests, plus a step stamped
    with a naive timestamp (seq 8, no timezone) that never matches a
    window. The closed window [10:01:00Z, 10:04:30Z] keeps seqs 1-4:
    the priced gpt-5 call at 10:00 and the small gpt-5 call at 10:05
    fall outside it. Latency is measured between kept steps, and the
    last kept step runs to the summary timestamp, as in the
    unwindowed export.
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "run.jsonl")
        records = [
            _record(0, "model_call", {"model": "gpt-5", "prompt": "plan"},
                    {"input": 1000, "output": 500},
                    "2026-11-22T10:00:00Z"),
            _record(1, "tool_call", {"tool": "search", "args": {"q": "x"}},
                    None, "2026-11-22T10:01:00Z"),
            _record(2, "tool_result", {"tool": "search", "result": "ok"},
                    None, "2026-11-22T10:01:30Z"),
            _record(3, "tool_result", {"tool": "fetch",
                                       "result": "error: 404"},
                    None, "2026-11-22T10:04:00Z"),
            _record(4, "model_call", {"model": "unknown-model",
                                      "prompt": "p"},
                    {"input": 9000, "output": 9000},
                    "2026-11-22T10:04:30Z"),
            _record(5, "model_call", {"model": "gpt-5", "prompt": "small"},
                    50, "2026-11-22T10:05:00Z"),
            _record(6, "summary", {"events": 6, "total_tokens": 19550,
                                   "sampled_out": 2},
                    None, "2026-11-22T10:06:00Z"),
            _record(7, "checkpoint", {"compacted_count": 3,
                                      "head_hash": "ab" * 32},
                    None, "2026-11-22T09:00:00Z"),
            _record(8, "tool_call", {"tool": "search", "args": {}},
                    None, "2026-11-22T10:02:00"),
        ]
        with open(self.path, "w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record) + "\n")

    def tearDown(self):
        self.tmp.cleanup()

    def _by_name(self, **kwargs):
        data = export_otlp_metrics(self.path, **kwargs)
        metrics = data["resourceMetrics"][0]["scopeMetrics"][0]["metrics"]
        return {m["name"]: m for m in metrics}

    @staticmethod
    def _points(metric):
        out = {}
        for point in metric["sum"]["dataPoints"]:
            attrs = {a["key"]: a["value"].get("stringValue", "")
                     for a in point["attributes"]}
            key = attrs.get("gen_ai.request.model") or attrs.get("tool.name")
            value = point.get("asDouble")
            if value is None:
                value = int(point["asInt"])
            out[key] = value
        return out

    def test_window_keeps_only_in_window_steps(self):
        by_name = self._by_name(since="2026-11-22T10:01:00Z",
                                until="2026-11-22T10:04:30Z")
        self.assertEqual(self._points(by_name["blackbox.model_calls"]),
                         {"unknown-model": 1})
        # The naive-timestamp step sits inside the window's span but
        # carries no timezone, so it never matches.
        self.assertEqual(self._points(by_name["blackbox.tool_calls"]),
                         {"search": 1})
        self.assertEqual(
            self._points(by_name["blackbox.model_input_tokens"]),
            {"unknown-model": 9000})
        self.assertEqual(
            self._points(by_name["blackbox.model_output_tokens"]),
            {"unknown-model": 9000})
        # The only model call in the window is unpriced, and the
        # priced gpt-5 call is outside it: no cost metric at all.
        self.assertNotIn("blackbox.model_cost_usd", by_name)
        # The last kept step runs to the summary timestamp: 90 s.
        self.assertEqual(
            self._points(by_name["blackbox.model_latency_ms"]),
            {"unknown-model": 90000.0})

    def test_checkpoint_is_not_a_step(self):
        # A checkpoint is bookkeeping about compacted-away records,
        # not a step: it contributes no counts, and it never ends a
        # step's latency — before this fix its 09:00 timestamp ended
        # the last model call's latency with a negative gap.
        by_name = self._by_name()
        self.assertEqual(self._points(by_name["blackbox.model_calls"]),
                         {"gpt-5": 2, "unknown-model": 1})
        # The gpt-5 call at seq 5 contributes no latency: the next
        # step's timestamp is naive, so it cannot end a latency.
        self.assertEqual(
            self._points(by_name["blackbox.model_latency_ms"]),
            {"gpt-5": 60000.0, "unknown-model": 30000.0})

    def test_window_since_only_and_until_only(self):
        by_name = self._by_name(since="2026-11-22T10:04:00Z")
        self.assertEqual(self._points(by_name["blackbox.model_calls"]),
                         {"unknown-model": 1, "gpt-5": 1})
        self.assertNotIn("blackbox.tool_calls", by_name)
        self.assertEqual(
            self._points(by_name["blackbox.model_latency_ms"]),
            {"unknown-model": 30000.0, "gpt-5": 60000.0})
        by_name = self._by_name(until="2026-11-22T10:01:00Z")
        self.assertEqual(self._points(by_name["blackbox.model_calls"]),
                         {"gpt-5": 1})
        self.assertEqual(self._points(by_name["blackbox.tool_calls"]),
                         {"search": 1})
        self.assertEqual(
            self._points(by_name["blackbox.model_cost_usd"]),
            {"gpt-5": 0.00625})
        self.assertEqual(
            self._points(by_name["blackbox.model_latency_ms"]),
            {"gpt-5": 60000.0})

    def test_window_last_relative(self):
        # --last is a lower bound only: from 10:04:30Z back 3m30s the
        # bound is 10:01:00Z, so seqs 1-5 match, including the step
        # stamped after now.
        now = datetime(2026, 11, 22, 10, 4, 30, tzinfo=timezone.utc)
        by_name = self._by_name(last="3m30s", now=now)
        self.assertEqual(self._points(by_name["blackbox.model_calls"]),
                         {"unknown-model": 1, "gpt-5": 1})
        self.assertEqual(self._points(by_name["blackbox.tool_calls"]),
                         {"search": 1})

    def test_window_compares_as_datetimes(self):
        # A fractional-second bound keeps seq 3 (10:04:00) but drops
        # seq 2 (10:01:30), which a plain string comparison against
        # the whole-second bound would get wrong — and seq 3 alone
        # (a tool_result) yields no metrics at all.
        data = export_otlp_metrics(
            self.path, since="2026-11-22T10:01:30.500Z",
            until="2026-11-22T10:04:00Z")
        metrics = data["resourceMetrics"][0]["scopeMetrics"][0]["metrics"]
        self.assertEqual(metrics, [])

    def test_window_accepts_a_timezone_offset(self):
        # 12:01:00+02:00 is 10:01:00Z, so the window is the same one.
        by_name = self._by_name(since="2026-11-22T12:01:00+02:00",
                                until="2026-11-22T10:04:30Z")
        self.assertEqual(self._points(by_name["blackbox.model_calls"]),
                         {"unknown-model": 1})

    def test_window_rejects_bad_bounds(self):
        for bad in ("tomorrow", "2026-11-22T10:00:00", "", "10m"):
            with self.assertRaises(ValueError):
                export_otlp_metrics(self.path, since=bad)
            with self.assertRaises(ValueError):
                export_otlp_metrics(self.path, until=bad)
        with self.assertRaises(ValueError):
            export_otlp_metrics(self.path,
                                since="2026-11-22T10:05:00Z",
                                until="2026-11-22T10:01:00Z")
        now = datetime(2026, 11, 22, 10, 5, tzinfo=timezone.utc)
        with self.assertRaises(ValueError):
            export_otlp_metrics(self.path, last="5m",
                                since="2026-11-22T10:01:00Z", now=now)
        with self.assertRaises(ValueError):
            export_otlp_metrics(self.path, last="5m",
                                until="2026-11-22T10:04:30Z", now=now)
        with self.assertRaises(ValueError):
            export_otlp_metrics(self.path, last="10", now=now)

    def test_cli_metrics_since_until(self):
        # A wide absolute window matches every step with a timezone;
        # a bad bound, a reversed window, and --last combined with
        # --since are all errors, exit code 2.
        out = io.StringIO()
        with redirect_stdout(out):
            self.assertEqual(
                main(["metrics", self.path, "--since",
                      "2026-11-01T00:00:00Z", "--until",
                      "2027-01-01T00:00:00Z"]), 0)
        data = json.loads(out.getvalue())
        metrics = data["resourceMetrics"][0]["scopeMetrics"][0]["metrics"]
        by_name = {m["name"]: m for m in metrics}
        self.assertEqual(self._points(by_name["blackbox.model_calls"]),
                         {"gpt-5": 2, "unknown-model": 1})
        self.assertEqual(self._points(by_name["blackbox.tool_calls"]),
                         {"search": 1})
        with redirect_stdout(io.StringIO()):
            self.assertEqual(
                main(["metrics", self.path, "--since", "tomorrow"]), 2)
            self.assertEqual(
                main(["metrics", self.path, "--since",
                      "2027-01-01T00:00:00Z", "--until",
                      "2026-01-01T00:00:00Z"]), 2)
            self.assertEqual(
                main(["metrics", self.path, "--last", "1d",
                      "--since", "2026-11-01T00:00:00Z"]), 2)
            self.assertEqual(
                main(["metrics", self.path, "--last", "10"]), 2)

    def test_cli_metrics_last(self):
        # The log was just recorded, so --last 1d matches every step.
        path = os.path.join(self.tmp.name, "fresh.jsonl")
        with Recorder(path, task="test") as rec:
            rec.record("model_call", {"model": "gpt-5", "prompt": "plan"},
                       tokens={"input": 1000, "output": 500})
        out = io.StringIO()
        with redirect_stdout(out):
            self.assertEqual(main(["metrics", path, "--last", "1d"]), 0)
        data = json.loads(out.getvalue())
        metrics = data["resourceMetrics"][0]["scopeMetrics"][0]["metrics"]
        by_name = {m["name"]: m for m in metrics}
        self.assertEqual(self._points(by_name["blackbox.model_calls"]),
                         {"gpt-5": 1})
