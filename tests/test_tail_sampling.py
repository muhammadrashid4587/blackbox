"""Tail-based sampling: interesting traces kept whole, boring ones
sampled down, and the decision recorded in the export summary."""

import json
import os
import tempfile
import unittest

from blackbox import Recorder
from blackbox.__main__ import main
from blackbox.tail_sampling import (export_tail_sampled_otlp,
                                    tail_sample_records)


def _record(seq, ts, type_, payload=None, tokens=None):
    return {"seq": seq, "ts": ts, "type": type_,
            "payload": payload or {}, "tokens": tokens,
            "prev_hash": "0" * 64 if seq == 0 else "%064x" % seq,
            "hash": "%064x" % (seq + 100)}


def _steps(n, gap_s=0, start="2026-11-22T00:00:00Z"):
    """n plain steps, `gap_s` seconds apart."""
    from datetime import datetime, timedelta
    base = datetime.fromisoformat(start.replace("Z", "+00:00"))
    return [_record(i, (base + timedelta(seconds=gap_s * i)).isoformat()
                    .replace("+00:00", "Z"), "tool_call",
                    {"tool": "search"})
            for i in range(n)]


def _summary(ts="2026-11-22T00:00:01Z"):
    return _record(99, ts, "summary", {"events": 8})


class TailSampleTest(unittest.TestCase):
    def test_error_result_keeps_whole_trace(self):
        records = _steps(4) + [_summary()]
        records[2] = _record(2, records[2]["ts"], "tool_result",
                             {"tool": "search", "error": "timeout"})
        kept, decision = tail_sample_records(records, keep_rate=0.5)
        self.assertEqual(decision["decision"], "kept_all")
        self.assertEqual(decision["reasons"], ["error"])
        self.assertEqual(len(kept), 4)
        self.assertEqual(decision["dropped_spans"], 0)

    def test_error_status_and_error_type_also_trigger(self):
        for payload, type_ in (({"status": "error"}, "tool_result"),
                               ({}, "error")):
            records = _steps(2) + [_summary()]
            records[1] = _record(1, records[1]["ts"], type_, payload)
            _, decision = tail_sample_records(records)
            self.assertEqual(decision["reasons"], ["error"])

    def test_slow_span_keeps_whole_trace(self):
        records = _steps(3, gap_s=6) + [_summary("2026-11-22T00:00:20Z")]
        kept, decision = tail_sample_records(records, keep_rate=0.5)
        self.assertEqual(decision["decision"], "kept_all")
        self.assertEqual(decision["reasons"], ["slow_span"])
        self.assertEqual(decision["max_span_ms"], 8000.0)
        self.assertEqual(len(kept), 3)

    def test_span_at_threshold_is_not_slow(self):
        records = _steps(2, gap_s=5) + [_summary("2026-11-22T00:00:10Z")]
        _, decision = tail_sample_records(records, keep_rate=1.0,
                                          slow_ms=5000)
        self.assertEqual(decision["decision"], "sampled")
        self.assertEqual(decision["reasons"], [])

    def test_high_cost_keeps_whole_trace(self):
        records = [
            _record(0, "2026-11-22T00:00:00Z", "model_call",
                    {"model": "gpt-5"},
                    {"input": 1000000, "output": 0}),
            _record(1, "2026-11-22T00:00:01Z", "tool_call",
                    {"tool": "search"}),
        ] + [_summary("2026-11-22T00:00:02Z")]
        kept, decision = tail_sample_records(records, keep_rate=0.5)
        self.assertEqual(decision["decision"], "kept_all")
        self.assertEqual(decision["reasons"], ["high_cost"])
        self.assertEqual(decision["total_cost_usd"], 1.25)
        self.assertEqual(len(kept), 2)

    def test_unpriced_model_has_no_cost(self):
        records = [
            _record(0, "2026-11-22T00:00:00Z", "model_call",
                    {"model": "unknown-model"},
                    {"input": 1000000, "output": 1000000}),
        ]
        _, decision = tail_sample_records(records, keep_rate=1.0)
        self.assertIsNone(decision["total_cost_usd"])
        self.assertEqual(decision["reasons"], [])

    def test_boring_trace_is_sampled_down(self):
        records = _steps(8) + [_summary()]
        kept, decision = tail_sample_records(records, keep_rate=0.5,
                                             seed=7)
        self.assertEqual(decision["decision"], "sampled")
        self.assertEqual(decision["reasons"], [])
        self.assertEqual(decision["total_spans"], 8)
        self.assertEqual(decision["kept_spans"] + decision["dropped_spans"],
                         8)
        self.assertLess(decision["kept_spans"], 8)
        self.assertGreater(decision["kept_spans"], 0)

    def test_same_seed_keeps_same_spans(self):
        records = _steps(8) + [_summary()]
        first, _ = tail_sample_records(records, keep_rate=0.5, seed=7)
        second, _ = tail_sample_records(records, keep_rate=0.5, seed=7)
        self.assertEqual([r["seq"] for r in first],
                         [r["seq"] for r in second])

    def test_keep_rate_must_be_in_range(self):
        records = _steps(2)
        for bad in (0, 0.0, 1.5, -0.5):
            with self.assertRaises(ValueError):
                tail_sample_records(records, keep_rate=bad)

    def test_empty_session(self):
        kept, decision = tail_sample_records([])
        self.assertEqual(kept, [])
        self.assertEqual(decision["decision"], "sampled")
        self.assertEqual(decision["total_spans"], 0)
        self.assertIsNone(decision["max_span_ms"])
        self.assertIsNone(decision["total_cost_usd"])

    def test_summary_is_not_a_span(self):
        records = _steps(3) + [_summary()]
        kept, decision = tail_sample_records(records, keep_rate=1.0)
        self.assertEqual(decision["total_spans"], 3)
        self.assertNotIn("summary", [r["type"] for r in kept])


class TailSampleExportTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "run.jsonl")
        with Recorder(self.path, task="t", model="m") as rec:
            rec.record("model_call", {"model": "m", "prompt": "hi"},
                       tokens=10)
            rec.record("tool_call", {"tool": "search", "args": {"q": "x"}})
            rec.record("tool_result", {"tool": "search", "result": "ok"})

    def tearDown(self):
        self.tmp.cleanup()

    def _spans(self, data):
        return data["resourceSpans"][0]["scopeSpans"][0]["spans"]

    def test_export_carries_decision_in_summary(self):
        data = export_tail_sampled_otlp(self.path, keep_rate=0.5, seed=7)
        decision = data["tail_sampling"]
        self.assertEqual(decision["decision"], "sampled")
        self.assertEqual(decision["total_spans"], 3)
        self.assertEqual(len(self._spans(data)),
                         decision["kept_spans"])

    def test_export_keeps_error_trace_whole(self):
        error_path = os.path.join(self.tmp.name, "error.jsonl")
        with Recorder(error_path, task="t", model="m") as rec:
            rec.record("tool_call", {"tool": "search", "args": {"q": "x"}})
            rec.record("tool_result",
                       {"tool": "search", "error": "timeout"})
        data = export_tail_sampled_otlp(error_path, keep_rate=0.5, seed=7)
        self.assertEqual(data["tail_sampling"]["decision"], "kept_all")
        self.assertEqual(len(self._spans(data)), 2)

    def test_kept_spans_still_point_at_their_records(self):
        data = export_tail_sampled_otlp(self.path, keep_rate=0.5, seed=7)
        with open(self.path) as f:
            records = [json.loads(line) for line in f]
        hashes = {r["hash"][:16] for r in records}
        for span in self._spans(data):
            self.assertIn(span["spanId"], hashes)

    def test_cli_writes_file_and_reports_decision(self):
        out = os.path.join(self.tmp.name, "spans.json")
        rc = main(["tail-sample", self.path, "--out", out,
                   "--keep-rate", "0.5", "--seed", "7"])
        self.assertEqual(rc, 0)
        with open(out) as f:
            data = json.load(f)
        self.assertIn("tail_sampling", data)
        self.assertEqual(len(self._spans(data)),
                         data["tail_sampling"]["kept_spans"])

    def test_cli_rejects_bad_keep_rate(self):
        out = os.path.join(self.tmp.name, "spans.json")
        rc = main(["tail-sample", self.path, "--out", out,
                   "--keep-rate", "0"])
        self.assertEqual(rc, 2)


if __name__ == "__main__":
    unittest.main()
