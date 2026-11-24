"""OpenTelemetry export: spans carry the chain, round-trip the steps."""

import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from datetime import datetime, timezone

from blackbox import Recorder
from blackbox.__main__ import main
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


def _window_record(seq, event_type, payload, tokens=None,
                   ts="2026-11-22T10:00:00Z"):
    return {"seq": seq, "ts": ts, "type": event_type, "payload": payload,
            "tokens": tokens, "prev_hash": "0" * 64, "hash": "0" * 64}


class OtelSinceUntilTest(unittest.TestCase):
    """--since / --until (and --last) narrow the span export to a
    window, and records_to_spans no longer treats a checkpoint as a
    step.

    Same fixture as the metrics window tests, plus a step stamped
    with a naive timestamp (seq 8, no timezone) that never matches a
    window and gets no span. The closed window [10:01:00Z,
    10:04:30Z] keeps seqs 1-4: the priced gpt-5 call at 10:00 and
    the small gpt-5 call at 10:05 fall outside it. The last kept
    step runs to the summary timestamp, as in the unwindowed export.
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "run.jsonl")
        records = [
            _window_record(0, "model_call",
                           {"model": "gpt-5", "prompt": "plan"},
                           {"input": 1000, "output": 500},
                           "2026-11-22T10:00:00Z"),
            _window_record(1, "tool_call",
                           {"tool": "search", "args": {"q": "x"}},
                           None, "2026-11-22T10:01:00Z"),
            _window_record(2, "tool_result",
                           {"tool": "search", "result": "ok"},
                           None, "2026-11-22T10:01:30Z"),
            _window_record(3, "tool_result",
                           {"tool": "fetch", "result": "error: 404"},
                           None, "2026-11-22T10:04:00Z"),
            _window_record(4, "model_call",
                           {"model": "unknown-model", "prompt": "p"},
                           {"input": 9000, "output": 9000},
                           "2026-11-22T10:04:30Z"),
            _window_record(5, "model_call",
                           {"model": "gpt-5", "prompt": "small"},
                           50, "2026-11-22T10:05:00Z"),
            _window_record(6, "summary",
                           {"events": 6, "total_tokens": 19550,
                            "sampled_out": 2},
                           None, "2026-11-22T10:06:00Z"),
            _window_record(7, "checkpoint",
                           {"compacted_count": 3, "head_hash": "ab" * 32},
                           None, "2026-11-22T09:00:00Z"),
            _window_record(8, "tool_call", {"tool": "search", "args": {}},
                           None, "2026-11-22T10:02:00"),
        ]
        with open(self.path, "w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record) + "\n")

    def tearDown(self):
        self.tmp.cleanup()

    def _spans(self, **kwargs):
        data = export_otlp(self.path, **kwargs)
        return data["resourceSpans"][0]["scopeSpans"][0]["spans"]

    @staticmethod
    def _seqs(spans):
        return [int(a["value"]["intValue"])
                for s in spans for a in s["attributes"]
                if a["key"] == "blackbox.seq"]

    @staticmethod
    def _durations(spans):
        return {int(a["value"]["intValue"]): (
            int(s["endTimeUnixNano"]) - int(s["startTimeUnixNano"])) / 1e9
            for s in spans for a in s["attributes"]
            if a["key"] == "blackbox.seq"}

    def test_window_keeps_only_in_window_steps(self):
        spans = self._spans(since="2026-11-22T10:01:00Z",
                            until="2026-11-22T10:04:30Z")
        self.assertEqual(self._seqs(spans), [1, 2, 3, 4])
        self.assertEqual([s["name"] for s in spans],
                         ["tool_call search", "tool_result search",
                          "tool_result fetch",
                          "model_call unknown-model"])
        # Latency is measured between kept steps, and the last kept
        # step runs to the summary timestamp: 90 s.
        self.assertEqual(self._durations(spans),
                         {1: 30.0, 2: 150.0, 3: 30.0, 4: 90.0})

    def test_checkpoint_is_not_a_span(self):
        # A checkpoint is bookkeeping about compacted-away records,
        # not a step: it gets no span, and it never ends a step's
        # latency — before this fix its 09:00 timestamp ended the
        # last model call's span with a negative gap.
        spans = self._spans()
        self.assertEqual(self._seqs(spans), [0, 1, 2, 3, 4, 5])
        # The naive-timestamp step gets no span either: a span with
        # no honest start time is not a span.
        self.assertNotIn(8, self._seqs(spans))
        # The gpt-5 call at seq 5 contributes no latency: the next
        # step's timestamp is naive, so its span is 1 ms.
        self.assertEqual(self._durations(spans)[5], 0.001)
        self.assertEqual(self._durations(spans)[4], 30.0)
        self.assertEqual(self._durations(spans)[0], 60.0)

    def test_last_summary_timestamp_ends_last_span(self):
        # The summary timestamp comes from the last summary record,
        # not assumed to be the last record: here the last record is
        # a checkpoint at 09:00, and there are two summaries, so the
        # last step runs to the later summary at 10:05.
        path = os.path.join(self.tmp.name, "compacted.jsonl")
        records = [
            _window_record(0, "model_call",
                           {"model": "gpt-5", "prompt": "plan"},
                           {"input": 10, "output": 5},
                           "2026-11-22T10:00:00Z"),
            _window_record(1, "model_call",
                           {"model": "gpt-5", "prompt": "again"},
                           {"input": 10, "output": 5},
                           "2026-11-22T10:01:00Z"),
            _window_record(2, "summary", {"events": 2},
                           None, "2026-11-22T10:02:00Z"),
            _window_record(3, "summary", {"events": 2},
                           None, "2026-11-22T10:05:00Z"),
            _window_record(4, "checkpoint",
                           {"compacted_count": 3, "head_hash": "ab" * 32},
                           None, "2026-11-22T09:00:00Z"),
        ]
        with open(path, "w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record) + "\n")
        spans = export_otlp(path)["resourceSpans"][0]["scopeSpans"][0]["spans"]
        self.assertEqual(self._seqs(spans), [0, 1])
        self.assertEqual(self._durations(spans), {0: 60.0, 1: 240.0})

    def test_unparseable_step_contributes_no_latency(self):
        # A step whose timestamp does not parse gets no span and
        # contributes no latency — the export does not crash, and
        # the step before it ends 1 ms after it starts.
        path = os.path.join(self.tmp.name, "bad-ts.jsonl")
        records = [
            _window_record(0, "model_call",
                           {"model": "gpt-5", "prompt": "plan"},
                           {"input": 10, "output": 5},
                           "2026-11-22T10:00:00Z"),
            _window_record(1, "tool_call",
                           {"tool": "search", "args": {"q": "x"}},
                           None, "not-a-time"),
            _window_record(2, "model_call",
                           {"model": "gpt-5", "prompt": "again"},
                           {"input": 10, "output": 5},
                           "2026-11-22T10:02:00Z"),
            _window_record(3, "summary", {"events": 3},
                           None, "2026-11-22T10:03:00Z"),
        ]
        with open(path, "w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record) + "\n")
        spans = export_otlp(path)["resourceSpans"][0]["scopeSpans"][0]["spans"]
        self.assertEqual(self._seqs(spans), [0, 2])
        self.assertEqual(self._durations(spans), {0: 0.001, 2: 60.0})

    def test_window_since_only_and_until_only(self):
        spans = self._spans(since="2026-11-22T10:04:00Z")
        self.assertEqual(self._seqs(spans), [3, 4, 5])
        spans = self._spans(until="2026-11-22T10:01:00Z")
        self.assertEqual(self._seqs(spans), [0, 1])

    def test_window_last_relative(self):
        # --last is a lower bound only: from 10:04:30Z back 3m30s the
        # bound is 10:01:00Z, so seqs 1-5 match, including the step
        # stamped after now.
        now = datetime(2026, 11, 22, 10, 4, 30, tzinfo=timezone.utc)
        spans = self._spans(last="3m30s", now=now)
        self.assertEqual(self._seqs(spans), [1, 2, 3, 4, 5])

    def test_window_compares_as_datetimes(self):
        # A fractional-second bound keeps seq 3 (10:04:00) but drops
        # seq 2 (10:01:30), which a plain string comparison against
        # the whole-second bound would get wrong — and seq 3 alone
        # (a tool_result) is the only span.
        spans = self._spans(since="2026-11-22T10:01:30.500Z",
                            until="2026-11-22T10:04:00Z")
        self.assertEqual(self._seqs(spans), [3])

    def test_window_accepts_a_timezone_offset(self):
        # 12:01:00+02:00 is 10:01:00Z, so the window is the same one.
        spans = self._spans(since="2026-11-22T12:01:00+02:00",
                            until="2026-11-22T10:04:30Z")
        self.assertEqual(self._seqs(spans), [1, 2, 3, 4])

    def test_window_rejects_bad_bounds(self):
        for bad in ("tomorrow", "2026-11-22T10:00:00", "", "10m"):
            with self.assertRaises(ValueError):
                export_otlp(self.path, since=bad)
            with self.assertRaises(ValueError):
                export_otlp(self.path, until=bad)
        with self.assertRaises(ValueError):
            export_otlp(self.path, since="2026-11-22T10:05:00Z",
                        until="2026-11-22T10:01:00Z")
        now = datetime(2026, 11, 22, 10, 5, tzinfo=timezone.utc)
        with self.assertRaises(ValueError):
            export_otlp(self.path, last="5m",
                        since="2026-11-22T10:01:00Z", now=now)
        with self.assertRaises(ValueError):
            export_otlp(self.path, last="5m",
                        until="2026-11-22T10:04:30Z", now=now)
        with self.assertRaises(ValueError):
            export_otlp(self.path, last="10", now=now)

    def test_cli_otel_since_until(self):
        # A wide absolute window matches every step with a timezone;
        # a bad bound, a reversed window, and --last combined with
        # --since are all errors, exit code 2.
        out = io.StringIO()
        with redirect_stdout(out):
            self.assertEqual(
                main(["otel", self.path, "--since",
                      "2026-11-01T00:00:00Z", "--until",
                      "2027-01-01T00:00:00Z"]), 0)
        data = json.loads(out.getvalue())
        spans = data["resourceSpans"][0]["scopeSpans"][0]["spans"]
        self.assertEqual(self._seqs(spans), [0, 1, 2, 3, 4, 5])
        with redirect_stdout(io.StringIO()):
            self.assertEqual(
                main(["otel", self.path, "--since", "tomorrow"]), 2)
            self.assertEqual(
                main(["otel", self.path, "--since",
                      "2027-01-01T00:00:00Z", "--until",
                      "2026-01-01T00:00:00Z"]), 2)
            self.assertEqual(
                main(["otel", self.path, "--last", "1d",
                      "--since", "2026-11-01T00:00:00Z"]), 2)
            self.assertEqual(
                main(["otel", self.path, "--last", "10"]), 2)

    def test_cli_otel_last(self):
        # The log was just recorded, so --last 1d matches every step.
        path = os.path.join(self.tmp.name, "fresh.jsonl")
        with Recorder(path, task="test") as rec:
            rec.record("model_call", {"model": "gpt-5", "prompt": "plan"},
                       tokens={"input": 1000, "output": 500})
        out = io.StringIO()
        with redirect_stdout(out):
            self.assertEqual(main(["otel", path, "--last", "1d"]), 0)
        data = json.loads(out.getvalue())
        spans = data["resourceSpans"][0]["scopeSpans"][0]["spans"]
        self.assertEqual(len(spans), 1)
        self.assertEqual(spans[0]["name"], "model_call gpt-5")

    def test_cli_export_since_until(self):
        # The same window flags on `blackbox export` (file export):
        # a narrow window keeps only the in-window spans, and a bad
        # bound exits 2.
        out = io.StringIO()
        with redirect_stdout(out):
            self.assertEqual(
                main(["export", self.path, "--since",
                      "2026-11-22T10:01:00Z", "--until",
                      "2026-11-22T10:04:30Z"]), 0)
        data = json.loads(out.getvalue())
        spans = data["resourceSpans"][0]["scopeSpans"][0]["spans"]
        self.assertEqual(self._seqs(spans), [1, 2, 3, 4])
        with redirect_stdout(io.StringIO()):
            self.assertEqual(
                main(["export", self.path, "--since", "tomorrow"]), 2)
            self.assertEqual(
                main(["export", self.path, "--last", "1d",
                      "--until", "2026-11-22T10:04:30Z"]), 2)
