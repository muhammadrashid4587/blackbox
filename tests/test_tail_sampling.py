"""Tail-based sampling: interesting traces kept whole, boring ones
sampled down, and the decision recorded in the export summary."""

import json
import os
import tempfile
import unittest
from datetime import datetime, timezone

from blackbox import Recorder
from blackbox.__main__ import main
from blackbox.otel import export_otlp, spans_to_records
from blackbox.sqlite_store import SQLiteStore
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


class TailSampleWindowTest(unittest.TestCase):
    """--since / --until (and --last) narrow the tail-sampled export
    to a window, and the window is applied before the tail decision.

    Same shape as the otel window fixture, plus an error step (seq 5)
    just outside the window: with the window it does not keep the
    trace whole, without it (or with a window that includes it) it
    does. Steps are 1 s apart, so no span is slow and the slow-span
    trigger never fires — the tests are about the window, not the
    trigger. The checkpoint (seq 7) is not a step, and the
    naive-timestamp step (seq 8) never matches a window.
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "run.jsonl")
        with open(self.path, "w", encoding="utf-8") as f:
            for record in _window_fixture():
                f.write(json.dumps(record) + "\n")

    def tearDown(self):
        self.tmp.cleanup()

    def _spans(self, data):
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
        data = export_tail_sampled_otlp(
            self.path, keep_rate=1.0,
            since="2026-11-22T10:01:00Z",
            until="2026-11-22T10:01:03Z")
        decision = data["tail_sampling"]
        self.assertEqual(decision["total_spans"], 4)
        self.assertEqual(decision["kept_spans"], 4)
        self.assertEqual(decision["reasons"], [])
        self.assertEqual(self._seqs(self._spans(data)), [1, 2, 3, 4])
        # Latency is measured between kept steps, and the last kept
        # step runs to the summary timestamp: 2 s.
        self.assertEqual(self._durations(self._spans(data)),
                         {1: 1.0, 2: 1.0, 3: 1.0, 4: 2.0})

    def test_checkpoint_is_not_a_step(self):
        # Unwindowed, every step is kept at rate 1.0: the checkpoint
        # is not counted in total_spans and gets no span. The
        # naive-timestamp step is still counted by the decision (it
        # gets no span) — only the window drops it.
        data = export_tail_sampled_otlp(self.path, keep_rate=1.0)
        decision = data["tail_sampling"]
        self.assertEqual(decision["total_spans"], 7)
        self.assertEqual(self._seqs(self._spans(data)),
                         [0, 1, 2, 3, 4, 5])

    def test_window_changes_the_trigger(self):
        # The error step is at 10:01:04, outside the window: the
        # windowed trace is boring and gets sampled down.
        data = export_tail_sampled_otlp(
            self.path, keep_rate=0.5, seed=7,
            since="2026-11-22T10:01:00Z",
            until="2026-11-22T10:01:03Z")
        decision = data["tail_sampling"]
        self.assertEqual(decision["decision"], "sampled")
        self.assertEqual(decision["reasons"], [])
        self.assertEqual(decision["total_spans"], 4)
        # A window that includes the error keeps the trace whole.
        data = export_tail_sampled_otlp(
            self.path, keep_rate=0.5, seed=7,
            since="2026-11-22T10:01:00Z",
            until="2026-11-22T10:01:04Z")
        decision = data["tail_sampling"]
        self.assertEqual(decision["decision"], "kept_all")
        self.assertEqual(decision["reasons"], ["error"])
        self.assertEqual(decision["total_spans"], 5)
        self.assertEqual(self._seqs(self._spans(data)),
                         [1, 2, 3, 4, 5])

    def test_window_since_only_and_until_only(self):
        data = export_tail_sampled_otlp(
            self.path, keep_rate=1.0,
            since="2026-11-22T10:01:02Z")
        self.assertEqual(data["tail_sampling"]["total_spans"], 3)
        self.assertEqual(data["tail_sampling"]["decision"], "kept_all")
        data = export_tail_sampled_otlp(
            self.path, keep_rate=1.0,
            until="2026-11-22T10:01:00Z")
        self.assertEqual(data["tail_sampling"]["total_spans"], 2)
        self.assertEqual(data["tail_sampling"]["decision"], "sampled")
        self.assertEqual(self._seqs(self._spans(data)), [0, 1])

    def test_window_last_relative(self):
        # --last is a lower bound only: from 10:01:03Z back 3 s the
        # bound is 10:01:00Z, so seqs 1-5 match, including the error
        # step stamped after now — and it keeps the trace whole.
        now = datetime(2026, 11, 22, 10, 1, 3, tzinfo=timezone.utc)
        data = export_tail_sampled_otlp(self.path, keep_rate=0.5,
                                        seed=7, last="3s", now=now)
        decision = data["tail_sampling"]
        self.assertEqual(decision["total_spans"], 5)
        self.assertEqual(decision["decision"], "kept_all")
        self.assertEqual(decision["reasons"], ["error"])

    def test_window_compares_as_datetimes(self):
        # A fractional-second bound keeps seqs 2-3: seq 1 (10:01:00)
        # falls before it, and the closed until keeps seq 3
        # (10:01:02).
        data = export_tail_sampled_otlp(
            self.path, keep_rate=1.0,
            since="2026-11-22T10:01:00.500Z",
            until="2026-11-22T10:01:02Z")
        self.assertEqual(self._seqs(self._spans(data)), [2, 3])

    def test_window_accepts_a_timezone_offset(self):
        # 12:01:00+02:00 is 10:01:00Z, so the window is the same one.
        data = export_tail_sampled_otlp(
            self.path, keep_rate=1.0,
            since="2026-11-22T12:01:00+02:00",
            until="2026-11-22T10:01:03Z")
        self.assertEqual(self._seqs(self._spans(data)), [1, 2, 3, 4])

    def test_window_rejects_bad_bounds(self):
        for bad in ("tomorrow", "2026-11-22T10:00:00", "", "10m"):
            with self.assertRaises(ValueError):
                export_tail_sampled_otlp(self.path, since=bad)
            with self.assertRaises(ValueError):
                export_tail_sampled_otlp(self.path, until=bad)
        with self.assertRaises(ValueError):
            export_tail_sampled_otlp(
                self.path, since="2026-11-22T10:01:04Z",
                until="2026-11-22T10:01:00Z")
        now = datetime(2026, 11, 22, 10, 1, 3, tzinfo=timezone.utc)
        with self.assertRaises(ValueError):
            export_tail_sampled_otlp(
                self.path, last="3s",
                since="2026-11-22T10:01:00Z", now=now)
        with self.assertRaises(ValueError):
            export_tail_sampled_otlp(
                self.path, last="3s",
                until="2026-11-22T10:01:03Z", now=now)
        with self.assertRaises(ValueError):
            export_tail_sampled_otlp(self.path, last="10", now=now)

    def test_window_works_on_sqlite_store(self):
        db_path = os.path.join(self.tmp.name, "run.db")
        store = SQLiteStore(db_path)
        store.replace_all(_window_fixture())
        store.close()
        data = export_tail_sampled_otlp(
            db_path, keep_rate=1.0,
            since="2026-11-22T10:01:00Z",
            until="2026-11-22T10:01:03Z")
        decision = data["tail_sampling"]
        self.assertEqual(decision["total_spans"], 4)
        self.assertEqual(decision["reasons"], [])
        self.assertEqual(self._seqs(self._spans(data)), [1, 2, 3, 4])

    def test_cli_tail_sample_since_until(self):
        # A wide absolute window matches every step with a timezone;
        # a bad bound, a reversed window, and --last combined with
        # --since are all errors, exit code 2.
        out = os.path.join(self.tmp.name, "spans.json")
        rc = main(["tail-sample", self.path, "--out", out,
                   "--since", "2026-11-01T00:00:00Z",
                   "--until", "2027-01-01T00:00:00Z",
                   "--keep-rate", "1.0"])
        self.assertEqual(rc, 0)
        with open(out) as f:
            data = json.load(f)
        self.assertEqual(self._seqs(self._spans(data)),
                         [0, 1, 2, 3, 4, 5])
        for args in (
                ["--since", "tomorrow"],
                ["--since", "2027-01-01T00:00:00Z",
                 "--until", "2026-11-01T00:00:00Z"],
                ["--last", "1d",
                 "--since", "2026-11-01T00:00:00Z"],
                ["--last", "10"]):
            self.assertEqual(
                main(["tail-sample", self.path, "--out", out] + args),
                2)
        # The log was just recorded, so --last 1d matches every step.
        live = os.path.join(self.tmp.name, "live.jsonl")
        with Recorder(live, task="t", model="m") as rec:
            rec.record("model_call", {"model": "m", "prompt": "hi"},
                       tokens=10)
            rec.record("tool_call", {"tool": "search", "args": {"q": "x"}})
        rc = main(["tail-sample", live, "--out", out, "--last", "1d",
                   "--keep-rate", "1.0"])
        self.assertEqual(rc, 0)
        with open(out) as f:
            data = json.load(f)
        self.assertEqual(data["tail_sampling"]["total_spans"], 2)


def _window_fixture():
    """Steps 1 s apart, an error step at 10:01:04, a summary, a
    checkpoint, and a naive-timestamp step that never matches a
    window. The closed window [10:01:00Z, 10:01:03Z] keeps seqs
    1-4."""
    return [
        _record(0, "2026-11-22T10:00:59Z", "model_call",
                {"model": "gpt-5", "prompt": "plan"},
                {"input": 1000, "output": 500}),
        _record(1, "2026-11-22T10:01:00Z", "tool_call",
                {"tool": "search", "args": {"q": "x"}}),
        _record(2, "2026-11-22T10:01:01Z", "tool_result",
                {"tool": "search", "result": "ok"}),
        _record(3, "2026-11-22T10:01:02Z", "tool_result",
                {"tool": "fetch", "result": "ok"}),
        _record(4, "2026-11-22T10:01:03Z", "model_call",
                {"model": "unknown-model", "prompt": "p"},
                {"input": 9000, "output": 9000}),
        _record(5, "2026-11-22T10:01:04Z", "tool_result",
                {"tool": "search", "error": "timeout"}),
        _record(6, "2026-11-22T10:01:05Z", "summary",
                {"events": 6, "total_tokens": 19550,
                 "sampled_out": 2}),
        _record(7, "2026-11-22T09:00:00Z", "checkpoint",
                {"compacted_count": 3, "head_hash": "ab" * 32}),
        _record(8, "2026-11-22T10:01:01", "tool_call",
                {"tool": "search", "args": {}}),
    ]


if __name__ == "__main__":
    unittest.main()
