"""Tail-based sampling: interesting traces kept whole, boring ones
sampled down, and the decision recorded in the export summary."""

import json
import os
import tempfile
import unittest

from blackbox import Recorder
from blackbox.__main__ import main
from blackbox.otel import export_otlp, spans_to_records
from blackbox.tail_sampling import (export_tail_sampled_otlp,
                                    tail_sample_records)
from blackbox.trace import link_parents


def _causal_session():
    """model -> two parallel search calls -> their results in order
    -> model. Causal parents differ from chain parents at seq 2, 3
    and 4, so closure and parent ids are distinguishable."""
    return [
        _record(0, "2026-11-22T00:00:00Z", "model_call",
                {"model": "m"}),
        _record(1, "2026-11-22T00:00:01Z", "tool_call",
                {"tool": "search", "args": {"q": "a"}}),
        _record(2, "2026-11-22T00:00:02Z", "tool_call",
                {"tool": "search", "args": {"q": "b"}}),
        _record(3, "2026-11-22T00:00:03Z", "tool_result",
                {"tool": "search", "result": "ra"}),
        _record(4, "2026-11-22T00:00:04Z", "tool_result",
                {"tool": "search", "result": "rb"}),
        _record(5, "2026-11-22T00:00:05Z", "model_call",
                {"model": "m"}),
        _summary("2026-11-22T00:00:06Z"),
    ]


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


class CausalTailSampleTest(unittest.TestCase):
    """--causal: a sampled span pulls in its causal ancestors, so
    the kept spans form a connected tree."""

    def test_causal_keeps_ancestors_of_sampled_spans(self):
        records = _causal_session()
        # Seed 17 keeps only seq 3 (the first tool_result) on its own.
        plain, _ = tail_sample_records(records, keep_rate=0.5, seed=17)
        self.assertEqual([r["seq"] for r in plain], [3])
        kept, decision = tail_sample_records(records, keep_rate=0.5,
                                             seed=17, causal=True)
        # seq 3's causal ancestors: seq 1 (its tool_call) and seq 0
        # (the model_call that decided it).
        self.assertEqual([r["seq"] for r in kept], [0, 1, 3])
        self.assertTrue(decision["causal"])
        self.assertEqual(decision["causal_ancestors_kept"], 2)
        self.assertEqual(decision["kept_spans"], 3)
        self.assertEqual(decision["dropped_spans"], 3)

    def test_causal_kept_is_superset_of_plain_for_same_seed(self):
        records = _causal_session()
        for seed in range(10):
            plain, _ = tail_sample_records(records, keep_rate=0.5,
                                           seed=seed)
            kept, _ = tail_sample_records(records, keep_rate=0.5,
                                          seed=seed, causal=True)
            self.assertTrue({r["seq"] for r in plain}
                            <= {r["seq"] for r in kept})

    def test_causal_kept_spans_are_closed_under_ancestors(self):
        records = _causal_session()
        parents = link_parents(records)
        for seed in range(10):
            kept, decision = tail_sample_records(
                records, keep_rate=0.5, seed=seed, causal=True)
            kept_seqs = {r["seq"] for r in kept}
            for record in kept:
                seq = parents.get(record["seq"])
                while seq is not None:
                    self.assertIn(seq, kept_seqs)
                    seq = parents.get(seq)
            # The decision counts exactly the spans the RNG did not
            # choose but closure added.
            plain, _ = tail_sample_records(records, keep_rate=0.5,
                                           seed=seed)
            self.assertEqual(decision["causal_ancestors_kept"],
                             len(kept_seqs) - len(plain))

    def test_causal_off_records_no_ancestors(self):
        records = _causal_session()
        _, decision = tail_sample_records(records, keep_rate=0.5,
                                          seed=17)
        self.assertFalse(decision["causal"])
        self.assertEqual(decision["causal_ancestors_kept"], 0)

    def test_causal_kept_all_adds_no_ancestors(self):
        records = _causal_session()
        records[3] = _record(3, records[3]["ts"], "tool_result",
                             {"tool": "search", "error": "timeout"})
        kept, decision = tail_sample_records(records, keep_rate=0.5,
                                             seed=17, causal=True)
        self.assertEqual(decision["decision"], "kept_all")
        self.assertEqual(len(kept), 6)
        self.assertEqual(decision["causal_ancestors_kept"], 0)


class CausalTailSampleExportTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "run.jsonl")
        with Recorder(self.path, task="t", model="m") as rec:
            rec.record("model_call", {"model": "m", "prompt": "plan"},
                       tokens={"input": 10, "output": 5})
            rec.record("tool_call", {"tool": "search", "args": {"q": "a"}})
            rec.record("tool_call", {"tool": "search", "args": {"q": "b"}})
            rec.record("tool_result", {"tool": "search", "result": "ra"})
            rec.record("tool_result", {"tool": "search", "result": "rb"})
            rec.record("model_call", {"model": "m", "prompt": "summarise"},
                       tokens={"input": 4, "output": 2})

    def tearDown(self):
        self.tmp.cleanup()

    def _spans(self, data):
        return data["resourceSpans"][0]["scopeSpans"][0]["spans"]

    def _attrs(self, span):
        out = {}
        for a in span["attributes"]:
            v = a["value"]
            if "intValue" in v:
                out[a["key"]] = int(v["intValue"])
            elif "boolValue" in v:
                out[a["key"]] = v["boolValue"]
            else:
                out[a["key"]] = v.get("stringValue")
        return out

    def test_causal_export_spans_are_connected(self):
        data = export_tail_sampled_otlp(self.path, keep_rate=0.5,
                                        seed=17, causal=True)
        spans = self._spans(data)
        self.assertEqual(len(spans),
                         data["tail_sampling"]["kept_spans"])
        self.assertEqual(data["tail_sampling"]["causal_ancestors_kept"],
                         2)
        span_ids = {s["spanId"] for s in spans}
        for span in spans:
            # Every non-root kept span's parent is a span in the
            # export: the sampled trace stays connected.
            if "parentSpanId" in span:
                self.assertIn(span["parentSpanId"], span_ids)
            self.assertIs(self._attrs(span).get("blackbox.causal_parent"),
                          True)

    def test_causal_export_parents_match_full_log_tree(self):
        data = export_tail_sampled_otlp(self.path, keep_rate=0.5,
                                        seed=17, causal=True)
        spans = self._spans(data)
        with open(self.path) as f:
            records = [json.loads(line) for line in f]
        parents = link_parents(records)
        by_seq = {r["seq"]: r for r in records
                  if r["type"] != "summary"}
        for span in spans:
            seq = self._attrs(span)["blackbox.seq"]
            parent_seq = parents[seq]
            if parent_seq is None:
                self.assertNotIn("parentSpanId", span)
            else:
                self.assertEqual(
                    span["parentSpanId"],
                    by_seq[parent_seq]["hash"][:16])

    def test_plain_export_can_dangle_but_causal_does_not(self):
        # Same seed, no --causal: only seq 3 is kept, and its chain
        # parent (seq 2's hash) is a span that is not in the export.
        plain = export_tail_sampled_otlp(self.path, keep_rate=0.5,
                                         seed=17)
        plain_spans = self._spans(plain)
        self.assertEqual(len(plain_spans), 1)
        plain_ids = {s["spanId"] for s in plain_spans}
        self.assertNotIn(plain_spans[0]["parentSpanId"], plain_ids)
        causal = export_tail_sampled_otlp(self.path, keep_rate=0.5,
                                          seed=17, causal=True)
        causal_ids = {s["spanId"] for s in self._spans(causal)}
        for span in self._spans(causal):
            if "parentSpanId" in span:
                self.assertIn(span["parentSpanId"], causal_ids)

    def test_cli_causal_flag(self):
        out = os.path.join(self.tmp.name, "spans.json")
        rc = main(["tail-sample", self.path, "--causal", "--out", out,
                   "--keep-rate", "0.5", "--seed", "17"])
        self.assertEqual(rc, 0)
        with open(out) as f:
            data = json.load(f)
        self.assertTrue(data["tail_sampling"]["causal"])
        self.assertEqual(data["tail_sampling"]["causal_ancestors_kept"],
                         2)
        span_ids = {s["spanId"] for s in self._spans(data)}
        for span in self._spans(data):
            if "parentSpanId" in span:
                self.assertIn(span["parentSpanId"], span_ids)


class ParentGapTest(unittest.TestCase):
    """Plain sampling: a kept span whose parent was dropped is
    marked, and the decision counts the gaps."""

    def _attrs(self, span):
        out = {}
        for a in span["attributes"]:
            v = a["value"]
            if "intValue" in v:
                out[a["key"]] = int(v["intValue"])
            elif "boolValue" in v:
                out[a["key"]] = v["boolValue"]
            else:
                out[a["key"]] = v.get("stringValue")
        return out

    def _spans(self, data):
        return data["resourceSpans"][0]["scopeSpans"][0]["spans"]

    def test_plain_sample_counts_parent_gaps(self):
        records = _causal_session()
        # Seed 17 keeps only seq 3, whose chain parent (seq 2) was
        # dropped: exactly one gap.
        _, decision = tail_sample_records(records, keep_rate=0.5,
                                          seed=17)
        self.assertEqual(decision["parent_gaps"], 1)
        self.assertEqual(decision["kept_spans"], 1)

    def test_kept_span_flags_dropped_parent(self):
        # Build the export from a real log file, as the CLI does.
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "run.jsonl")
            with Recorder(path, task="t", model="m") as rec:
                rec.record("model_call", {"model": "m", "prompt": "plan"},
                           tokens={"input": 10, "output": 5})
                rec.record("tool_call",
                           {"tool": "search", "args": {"q": "a"}})
                rec.record("tool_call",
                           {"tool": "search", "args": {"q": "b"}})
                rec.record("tool_result",
                           {"tool": "search", "result": "ra"})
                rec.record("tool_result",
                           {"tool": "search", "result": "rb"})
                rec.record("model_call",
                           {"model": "m", "prompt": "summarise"},
                           tokens={"input": 4, "output": 2})
            data = export_tail_sampled_otlp(path, keep_rate=0.5,
                                            seed=17)
            spans = self._spans(data)
            self.assertEqual(len(spans), 1)
            self.assertEqual(data["tail_sampling"]["parent_gaps"], 1)
            attrs = self._attrs(spans[0])
            # The gap is flagged, and the parentSpanId still names
            # the real dropped parent (seq 2's hash), not a rewrite.
            self.assertIs(attrs.get("blackbox.parent_sampled_out"),
                          True)
            with open(path) as f:
                log = [json.loads(line) for line in f]
            dropped = [r for r in log
                       if r["seq"] == 2][0]
            self.assertEqual(spans[0]["parentSpanId"],
                             dropped["hash"][:16])
            # Round-tripping still works: the flag is metadata, not
            # record data.
            back = spans_to_records(spans)
            self.assertEqual(back[0]["seq"], 3)

    def test_causal_sample_has_no_parent_gaps(self):
        records = _causal_session()
        _, decision = tail_sample_records(records, keep_rate=0.5,
                                          seed=17, causal=True)
        self.assertEqual(decision["parent_gaps"], 0)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "run.jsonl")
            with Recorder(path, task="t", model="m") as rec:
                rec.record("model_call", {"model": "m", "prompt": "plan"},
                           tokens={"input": 10, "output": 5})
                rec.record("tool_call",
                           {"tool": "search", "args": {"q": "a"}})
                rec.record("tool_call",
                           {"tool": "search", "args": {"q": "b"}})
                rec.record("tool_result",
                           {"tool": "search", "result": "ra"})
                rec.record("tool_result",
                           {"tool": "search", "result": "rb"})
                rec.record("model_call",
                           {"model": "m", "prompt": "summarise"},
                           tokens={"input": 4, "output": 2})
            data = export_tail_sampled_otlp(path, keep_rate=0.5,
                                            seed=17, causal=True)
            self.assertEqual(data["tail_sampling"]["parent_gaps"], 0)
            spans = self._spans(data)
        for span in spans:
            self.assertNotIn("blackbox.parent_sampled_out",
                             self._attrs(span))

    def test_kept_all_has_no_parent_gaps(self):
        records = _steps(4) + [_summary()]
        records[2] = _record(2, records[2]["ts"], "tool_result",
                             {"tool": "search", "error": "timeout"})
        _, decision = tail_sample_records(records, keep_rate=0.5)
        self.assertEqual(decision["decision"], "kept_all")
        self.assertEqual(decision["parent_gaps"], 0)

    def test_full_export_has_no_gap_flags(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "run.jsonl")
            with Recorder(path, task="t", model="m") as rec:
                rec.record("model_call", {"model": "m", "prompt": "hi"},
                           tokens=10)
                rec.record("tool_call",
                           {"tool": "search", "args": {"q": "x"}})
                rec.record("tool_result",
                           {"tool": "search", "result": "ok"})
            data = export_otlp(path)
        for span in self._spans(data):
            self.assertNotIn("blackbox.parent_sampled_out",
                             self._attrs(span))

    def test_cli_reports_parent_gaps(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "run.jsonl")
            with Recorder(path, task="t", model="m") as rec:
                rec.record("model_call", {"model": "m", "prompt": "plan"},
                           tokens={"input": 10, "output": 5})
                rec.record("tool_call",
                           {"tool": "search", "args": {"q": "a"}})
                rec.record("tool_call",
                           {"tool": "search", "args": {"q": "b"}})
                rec.record("tool_result",
                           {"tool": "search", "result": "ra"})
                rec.record("tool_result",
                           {"tool": "search", "result": "rb"})
                rec.record("model_call",
                           {"model": "m", "prompt": "summarise"},
                           tokens={"input": 4, "output": 2})
            out = os.path.join(tmp, "spans.json")
            rc = main(["tail-sample", path, "--out", out,
                       "--keep-rate", "0.5", "--seed", "17"])
            self.assertEqual(rc, 0)
            with open(out) as f:
                data = json.load(f)
            self.assertEqual(data["tail_sampling"]["parent_gaps"], 1)


if __name__ == "__main__":
    unittest.main()
