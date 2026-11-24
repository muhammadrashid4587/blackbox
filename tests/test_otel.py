"""OpenTelemetry export: spans carry the chain, round-trip the steps."""

import json
import os
import tempfile
import unittest

from blackbox import Recorder
from blackbox.otel import (export_otlp, export_otlp_file, records_to_spans,
                           session_span, spans_to_records)


class OtelTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "run.jsonl")
        with Recorder(self.path, task="t", model="m") as rec:
            rec.record("model_call", {"model": "m", "prompt": "hi"},
                       tokens={"input": 10, "output": 5})
            rec.record("tool_call", {"tool": "search", "args": {"q": "x"}})
            rec.record("tool_result", {"tool": "search", "result": "ok"})
        with open(self.path) as f:
            self.records = [json.loads(line) for line in f]

    def tearDown(self):
        self.tmp.cleanup()

    def _spans(self):
        return export_otlp(self.path)["resourceSpans"][0]["scopeSpans"][0]["spans"]

    def test_spans_one_per_step_summary_excluded(self):
        spans = self._spans()
        self.assertEqual(len(spans), 3)  # 3 steps, summary not a span
        self.assertEqual([s["name"] for s in spans],
                         ["model_call m", "tool_call search",
                          "tool_result search"])

    def test_trace_and_span_ids_come_from_hash_chain(self):
        spans = self._spans()
        trace_id = self.records[0]["hash"][:32]
        for span in spans:
            self.assertEqual(span["traceId"], trace_id)
        self.assertEqual(spans[0]["spanId"], self.records[0]["hash"][:16])
        self.assertNotIn("parentSpanId", spans[0])  # root
        self.assertEqual(spans[1]["parentSpanId"],
                         self.records[0]["hash"][:16])
        self.assertEqual(spans[2]["parentSpanId"],
                         self.records[1]["hash"][:16])

    def test_attributes_carry_model_tool_and_tokens(self):
        spans = self._spans()

        def attrs(span):
            out = {}
            for a in span["attributes"]:
                v = a["value"]
                out[a["key"]] = (int(v["intValue"]) if "intValue" in v
                                 else v.get("stringValue"))
            return out

        model = attrs(spans[0])
        self.assertEqual(model["gen_ai.request.model"], "m")
        self.assertEqual(model["gen_ai.usage.input_tokens"], 10)
        self.assertEqual(model["gen_ai.usage.output_tokens"], 5)
        self.assertEqual(model["blackbox.tokens"], 15)
        self.assertEqual(model["blackbox.seq"], 0)
        tool = attrs(spans[1])
        self.assertEqual(tool["tool.name"], "search")
        self.assertEqual(tool["blackbox.type"], "tool_call")

    def test_span_times_cover_time_to_next_step(self):
        spans = self._spans()
        for a, b in zip(spans, spans[1:]):
            self.assertEqual(a["endTimeUnixNano"], b["startTimeUnixNano"])
        for span in spans:
            self.assertLessEqual(int(span["startTimeUnixNano"]),
                                int(span["endTimeUnixNano"]))

    def test_session_span_one_step(self):
        span = session_span(self.records, 1)
        self.assertEqual(span["spanId"], "%016x" % 1)
        self.assertEqual(span["name"], "tool_call")
        self.assertEqual(span["traceId"], self.records[0]["hash"][:32])
        self.assertEqual(span["parentSpanId"],
                         self.records[0]["hash"][:16])
        exported = records_to_spans(self.records)[1]
        self.assertEqual(span["startTimeUnixNano"],
                         exported["startTimeUnixNano"])
        self.assertEqual(span["endTimeUnixNano"],
                         exported["endTimeUnixNano"])
        self.assertEqual(span["attributes"], exported["attributes"])

    def test_session_span_unknown_and_summary_seqs_are_none(self):
        self.assertIsNone(session_span(self.records, 99))
        self.assertIsNone(session_span(self.records, 3))  # the summary

    def test_session_span_unpriced_model_has_no_cost(self):
        # Model "m" has no published price: no cost attribute,
        # never a guessed number.
        span = session_span(self.records, 0)
        keys = [a["key"] for a in span["attributes"]]
        self.assertNotIn("blackbox.cost_usd", keys)

    def test_session_span_missing_timestamp_is_an_error(self):
        records = [dict(r) for r in self.records]
        del records[0]["ts"]
        with self.assertRaises(ValueError):
            session_span(records, 0)

    def test_round_trip_recovers_steps(self):
        spans = self._spans()
        back = spans_to_records(spans)
        steps = self.records[:-1]  # summary excluded from export
        self.assertEqual(len(back), len(steps))
        for original, recovered in zip(steps, back):
            self.assertEqual(recovered["seq"], original["seq"])
            self.assertEqual(recovered["type"], original["type"])
            self.assertEqual(recovered["payload"], original["payload"])
            self.assertEqual(recovered["tokens"], original["tokens"])
            self.assertEqual(recovered["hash"], original["hash"])
            self.assertEqual(recovered["prev_hash"], original["prev_hash"])
            self.assertEqual(recovered["ts"], original["ts"])

    def test_export_file_writes_otlp_json(self):
        out_path = os.path.join(self.tmp.name, "spans.json")
        data = export_otlp_file(self.path, out_path)
        with open(out_path) as f:
            on_disk = json.load(f)
        self.assertEqual(on_disk, data)
        resource = data["resourceSpans"][0]
        self.assertEqual(
            resource["resource"]["attributes"][0]["value"]["stringValue"],
            "blackbox")
        scope = resource["scopeSpans"][0]["scope"]
        self.assertEqual(scope["name"], "blackbox")

    def test_empty_log_exports_no_spans(self):
        empty = os.path.join(self.tmp.name, "empty.jsonl")
        open(empty, "w").close()
        spans = records_to_spans([])
        self.assertEqual(spans, [])
        data = export_otlp(empty)
        self.assertEqual(
            data["resourceSpans"][0]["scopeSpans"][0]["spans"], [])
