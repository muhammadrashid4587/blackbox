"""`blackbox span`: one step as an OTLP span, the same span the
blackbox://session/span/{seq} MCP resource template serves — readable
without an MCP client. The report form and --json agree with the full
`blackbox otel` export; an unknown step, the summary, or a step with
no usable timestamp is an error, never an invented span."""

import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout

from blackbox import Recorder
from blackbox.__main__ import main
from blackbox.otel import records_to_spans, session_span
from blackbox.replay import load_records


class SpanCliTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "run.jsonl")
        with Recorder(self.path, task="t", model="gpt-5") as rec:
            rec.record("model_call", {"model": "gpt-5", "prompt": "hi"},
                       tokens={"input": 1200, "output": 340})
            rec.record("tool_call", {"tool": "search",
                                     "args": {"query": "mcp"}})
            rec.record("tool_result", {"tool": "search",
                                       "result": "3 results"})
        self.records = load_records(self.path)

    def tearDown(self):
        self.tmp.cleanup()

    def _run(self, *argv):
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = main(["span", self.path] + [str(a) for a in argv])
        return code, out.getvalue(), err.getvalue()

    def test_report_names_step_kind_and_seq_span_id(self):
        code, out, err = self._run(0)
        self.assertEqual(code, 0)
        self.assertEqual(err, "")
        lines = out.splitlines()
        self.assertEqual(lines[0], "model_call (span 0000000000000000)")
        self.assertEqual(lines[1],
                         "  trace: %s" % self.records[0]["hash"][:32])
        self.assertNotIn("parent", out)  # first step is the root
        self.assertIn("  start: ", out)
        self.assertIn("  end: ", out)
        self.assertIn("  duration: ", out)
        self.assertIn("  blackbox.seq: 0", out)
        self.assertIn("  gen_ai.request.model: gpt-5", out)
        self.assertIn("  gen_ai.usage.input_tokens: 1200", out)
        # gpt-5, 1200 in + 340 out = $0.0049.
        self.assertIn("  blackbox.cost_usd: 0.0049", out)

    def test_report_second_step_has_chain_parent(self):
        code, out, err = self._run(1)
        self.assertEqual(code, 0)
        lines = out.splitlines()
        self.assertEqual(lines[0], "tool_call (span 0000000000000001)")
        self.assertIn("  parent: %s" % self.records[0]["hash"][:16], out)
        self.assertIn("  tool.name: search", out)

    def test_json_matches_session_span(self):
        code, out, err = self._run(0, "--json")
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(out), session_span(self.records, 0))

    def test_span_agrees_with_full_export(self):
        # Same trace id, times, parent and attributes as the full
        # `blackbox otel` export — only the span id and name differ,
        # by design (see session_span).
        code, out, err = self._run(1, "--json")
        self.assertEqual(code, 0)
        span = json.loads(out)
        exported = records_to_spans(self.records)[1]
        self.assertEqual(span["traceId"], exported["traceId"])
        self.assertEqual(span["startTimeUnixNano"],
                         exported["startTimeUnixNano"])
        self.assertEqual(span["endTimeUnixNano"],
                         exported["endTimeUnixNano"])
        self.assertEqual(span["parentSpanId"], exported["parentSpanId"])
        self.assertEqual(span["attributes"], exported["attributes"])

    def test_last_step_without_summary_ends_1ms_after_start(self):
        # No summary yet: the last step's span ends 1 ms after it
        # starts, the same minimum-duration fallback as the full
        # export — no end time is invented from a later step. The
        # setUp log has a summary (the Recorder writes one on
        # close), so this test writes its own summary-free log.
        path = os.path.join(self.tmp.name, "no-summary.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for seq, ts in enumerate(("2026-11-22T10:00:00Z",
                                      "2026-11-22T10:01:00Z",
                                      "2026-11-22T10:02:00Z")):
                f.write(json.dumps({
                    "seq": seq, "ts": ts, "type": "tool_call",
                    "payload": {"tool": "search"},
                    "tokens": None, "prev_hash": "0" * 64,
                    "hash": "%064x" % (seq + 1)}) + "\n")
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = main(["span", path, "2", "--json"])
        self.assertEqual(code, 0)
        span = json.loads(out.getvalue())
        self.assertEqual(int(span["endTimeUnixNano"])
                         - int(span["startTimeUnixNano"]), 1_000_000)

    def test_last_step_ends_at_summary(self):
        with Recorder(self.path, task="t", model="gpt-5"):
            pass  # closing writes the summary record
        records = load_records(self.path)
        summary = records[-1]
        self.assertEqual(summary["type"], "summary")
        code, out, err = self._run(2, "--json")
        self.assertEqual(code, 0)
        span = json.loads(out)
        self.assertEqual(span["endTimeUnixNano"],
                         _nanos(summary["ts"]))

    def test_unpriced_call_has_no_cost(self):
        path = os.path.join(self.tmp.name, "unpriced.jsonl")
        with Recorder(path, task="t", model="mystery-model") as rec:
            rec.record("model_call",
                       {"model": "mystery-model", "prompt": "hi"},
                       tokens={"input": 100, "output": 50})
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = main(["span", path, "0", "--json"])
        self.assertEqual(code, 0)
        span = json.loads(out.getvalue())
        keys = [a["key"] for a in span["attributes"]]
        self.assertNotIn("blackbox.cost_usd", keys)

    def test_unknown_seq_is_an_error(self):
        code, out, err = self._run(99)
        self.assertEqual(code, 2)
        self.assertEqual(out, "")
        self.assertIn("no step 99", err)

    def test_summary_seq_is_an_error(self):
        with Recorder(self.path, task="t", model="gpt-5"):
            pass  # closing writes the summary record (seq 3)
        code, out, err = self._run(3)
        self.assertEqual(code, 2)
        self.assertEqual(out, "")
        self.assertIn("no step 3", err)

    def test_non_integer_seq_is_an_error(self):
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            with self.assertRaises(SystemExit) as caught:
                main(["span", self.path, "abc"])
        self.assertEqual(caught.exception.code, 2)

    def test_missing_file_is_an_error(self):
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = main(["span", os.path.join(self.tmp.name, "nope.jsonl"),
                         "0"])
        self.assertEqual(code, 1)
        self.assertEqual(out.getvalue(), "")
        self.assertIn("file not found", err.getvalue())

    def test_step_with_bad_timestamp_is_an_error(self):
        path = os.path.join(self.tmp.name, "bad-ts.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            f.write(json.dumps({
                "seq": 0, "ts": "not-a-time", "type": "model_call",
                "payload": {"model": "gpt-5", "prompt": "hi"},
                "tokens": {"input": 1, "output": 1},
                "prev_hash": "0" * 64, "hash": "a" * 64}) + "\n")
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = main(["span", path, "0"])
        self.assertEqual(code, 2)
        self.assertEqual(out.getvalue(), "")
        self.assertIn("no usable timestamp", err.getvalue())


def _nanos(ts):
    from datetime import datetime, timezone
    dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
    delta = dt - epoch
    seconds = delta.days * 86400 + delta.seconds
    return str(seconds * 1_000_000_000 + delta.microseconds * 1000)


if __name__ == "__main__":
    unittest.main()
