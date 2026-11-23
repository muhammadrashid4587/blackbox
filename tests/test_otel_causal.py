"""OTLP export with --causal: parent span ids from the causal trace
tree (trace.link_parents), not the hash chain, on .jsonl and .db
stores. The default export keeps chain parents."""

import json
import os
import tempfile
import unittest

from blackbox import Recorder
from blackbox.__main__ import main
from blackbox.otel import export_otlp, export_otlp_file, records_to_spans
from blackbox.replay import load_records
from blackbox.trace import build_trace, link_parents


def _record_session(path):
    """model -> two parallel search calls -> their results in order
    -> model. Causal parents differ from chain parents at seq 2, 3
    and 4, so the two exports are distinguishable."""
    with Recorder(path, task="t", model="m") as rec:
        rec.record("model_call", {"model": "m", "prompt": "plan"},
                   tokens={"input": 10, "output": 5})
        rec.record("tool_call", {"tool": "search", "args": {"q": "a"}})
        rec.record("tool_call", {"tool": "search", "args": {"q": "b"}})
        rec.record("tool_result", {"tool": "search", "result": "ra"})
        rec.record("tool_result", {"tool": "search", "result": "rb"})
        rec.record("model_call", {"model": "m", "prompt": "summarise"},
                   tokens={"input": 4, "output": 2})
    with open(path) as f:
        return [json.loads(line) for line in f] if path.endswith(".jsonl") \
            else load_records(path)


def _attrs(span):
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


class CausalOtelTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "run.jsonl")
        self.records = _record_session(self.path)
        self.steps = self.records[:-1]  # summary excluded

    def tearDown(self):
        self.tmp.cleanup()

    def _spans(self, **kw):
        return export_otlp(self.path, **kw)["resourceSpans"][0][
            "scopeSpans"][0]["spans"]

    def test_causal_parents_match_link_parents(self):
        spans = self._spans(causal=True)
        parents = link_parents(self.records)
        by_seq = {r["seq"]: r for r in self.steps}
        for span in spans:
            seq = _attrs(span)["blackbox.seq"]
            parent_seq = parents[seq]
            if parent_seq is None:
                self.assertNotIn("parentSpanId", span)
            else:
                self.assertEqual(
                    span["parentSpanId"],
                    by_seq[parent_seq]["hash"][:16])

    def test_causal_parents_match_trace_tree(self):
        spans = self._spans(causal=True)
        by_seq = {r["seq"]: r for r in self.steps}
        span_by_seq = {_attrs(s)["blackbox.seq"]: s for s in spans}
        # Walk the trace tree; each node's parent span id must be
        # the span id of its parent node.
        def walk(node, parent_seq):
            span = span_by_seq[node["seq"]]
            if parent_seq is None:
                self.assertNotIn("parentSpanId", span)
            else:
                self.assertEqual(
                    span["parentSpanId"],
                    by_seq[parent_seq]["hash"][:16])
            for child in node["children"]:
                walk(child, node["seq"])
        for root in build_trace(self.records):
            walk(root, None)

    def test_parallel_results_pair_fifo_not_chain_order(self):
        spans = self._spans(causal=True)
        by_seq = {r["seq"]: r for r in self.steps}
        span_by_seq = {_attrs(s)["blackbox.seq"]: s for s in spans}
        # seq 3 (first result) hangs off seq 1 (first call), seq 4
        # off seq 2 — the chain would put them off seq 2 and 3.
        self.assertEqual(span_by_seq[3]["parentSpanId"],
                         by_seq[1]["hash"][:16])
        self.assertEqual(span_by_seq[4]["parentSpanId"],
                         by_seq[2]["hash"][:16])
        # seq 2 (second call) hangs off the model_call, not the
        # first call.
        self.assertEqual(span_by_seq[2]["parentSpanId"],
                         by_seq[0]["hash"][:16])

    def test_causal_spans_carry_causal_parent_attribute(self):
        spans = self._spans(causal=True)
        for span in spans:
            self.assertIs(_attrs(span).get("blackbox.causal_parent"), True)

    def test_default_export_keeps_chain_parents(self):
        spans = self._spans()
        for span, record in zip(spans, self.steps):
            if record["prev_hash"] == "0" * 64:
                self.assertNotIn("parentSpanId", span)
            else:
                self.assertEqual(span["parentSpanId"],
                                 record["prev_hash"][:16])
            self.assertNotIn("blackbox.causal_parent", _attrs(span))
        # And the causal export really differs: seq 2's parent is
        # the model_call, not the first tool_call.
        causal = self._spans(causal=True)
        self.assertNotEqual(spans[2]["parentSpanId"],
                            causal[2]["parentSpanId"])

    def test_records_to_spans_causal_flag(self):
        spans = records_to_spans(self.records, causal=True)
        self.assertEqual(len(spans), 6)
        self.assertNotIn("parentSpanId", spans[0])
        by_seq = {r["seq"]: r for r in self.steps}
        self.assertEqual(spans[3]["parentSpanId"],
                         by_seq[1]["hash"][:16])

    def test_export_file_writes_causal_parents(self):
        out_path = os.path.join(self.tmp.name, "spans.json")
        data = export_otlp_file(self.path, out_path, causal=True)
        with open(out_path) as f:
            self.assertEqual(json.load(f), data)
        spans = data["resourceSpans"][0]["scopeSpans"][0]["spans"]
        by_seq = {r["seq"]: r for r in self.steps}
        self.assertEqual(spans[4]["parentSpanId"],
                         by_seq[2]["hash"][:16])

    def test_cli_otel_causal(self):
        out_path = os.path.join(self.tmp.name, "cli.json")
        self.assertEqual(main(["otel", self.path, "--causal",
                               "--out", out_path]), 0)
        with open(out_path) as f:
            data = json.load(f)
        spans = data["resourceSpans"][0]["scopeSpans"][0]["spans"]
        by_seq = {r["seq"]: r for r in self.steps}
        self.assertEqual(spans[3]["parentSpanId"],
                         by_seq[1]["hash"][:16])

    def test_cli_export_causal(self):
        out_path = os.path.join(self.tmp.name, "exp.json")
        self.assertEqual(main(["export", self.path, "--causal",
                               "--out", out_path]), 0)
        with open(out_path) as f:
            data = json.load(f)
        spans = data["resourceSpans"][0]["scopeSpans"][0]["spans"]
        by_seq = {r["seq"]: r for r in self.steps}
        self.assertEqual(spans[2]["parentSpanId"],
                         by_seq[0]["hash"][:16])

    def test_causal_export_on_db_store(self):
        db_path = os.path.join(self.tmp.name, "run.db")
        records = _record_session(db_path)
        steps = records[:-1]
        parents = link_parents(records)
        by_seq = {r["seq"]: r for r in steps}
        spans = export_otlp(db_path, causal=True)["resourceSpans"][0][
            "scopeSpans"][0]["spans"]
        self.assertEqual(len(spans), 6)
        for span in spans:
            seq = _attrs(span)["blackbox.seq"]
            parent_seq = parents[seq]
            if parent_seq is None:
                self.assertNotIn("parentSpanId", span)
            else:
                self.assertEqual(span["parentSpanId"],
                                 by_seq[parent_seq]["hash"][:16])

    def test_empty_log_causal_exports_no_spans(self):
        empty = os.path.join(self.tmp.name, "empty.jsonl")
        open(empty, "w").close()
        self.assertEqual(records_to_spans([], causal=True), [])
        data = export_otlp(empty, causal=True)
        self.assertEqual(
            data["resourceSpans"][0]["scopeSpans"][0]["spans"], [])


if __name__ == "__main__":
    unittest.main()
