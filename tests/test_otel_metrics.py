"""OpenTelemetry metrics export: per-model calls, tokens, latency,
cost, and per-tool call counts as OTLP Sum metrics."""

import json
import os
import tempfile
import unittest

from blackbox import Recorder
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


if __name__ == "__main__":
    unittest.main()
