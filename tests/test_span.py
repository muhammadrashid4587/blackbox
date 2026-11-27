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


class SpanWindowCliTest(unittest.TestCase):
    """Window flags for `blackbox span`: --before / --after around
    a step, --since / --until / --last narrowing the session first
    — the same window the blackbox_span MCP tool returns, through
    the shared span_window, so the CLI and the tool always agree."""

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

    def _fixed_log(self):
        # Three steps at fixed timestamps, so --since / --until /
        # --last windows are deterministic.
        path = os.path.join(self.tmp.name, "fixed.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for seq, ts in enumerate(("2026-11-22T10:00:00Z",
                                      "2026-11-22T10:01:00Z",
                                      "2026-11-22T10:02:00Z")):
                f.write(json.dumps({
                    "seq": seq, "ts": ts, "type": "tool_call",
                    "payload": {"tool": "search"},
                    "tokens": None, "prev_hash": "0" * 64,
                    "hash": "%064x" % (seq + 1)}) + "\n")
        return path

    def _run_fixed(self, path, *argv):
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = main(["span", path] + [str(a) for a in argv])
        return code, out.getvalue(), err.getvalue()

    def test_report_window_around_seq(self):
        code, out, err = self._run(1, "--before", 1, "--after", 1)
        self.assertEqual(code, 0)
        self.assertEqual(err, "")
        headers = [line for line in out.splitlines()
                   if line.endswith(")") and "(span " in line]
        self.assertEqual(headers, [
            "model_call (span 0000000000000000)",
            "tool_call (span 0000000000000001)",
            "tool_result (span 0000000000000002)",
        ])

    def test_window_clamps_at_session_ends(self):
        code, out, err = self._run(0, "--before", 5)
        self.assertEqual(code, 0)
        self.assertIn("model_call (span 0000000000000000)", out)
        self.assertNotIn("tool_call", out)
        code, out, err = self._run(2, "--after", 5)
        self.assertEqual(code, 0)
        self.assertIn("tool_result (span 0000000000000002)", out)
        self.assertNotIn("tool_call", out)

    def test_json_window_matches_span_window(self):
        from blackbox.otel import span_window
        code, out, err = self._run(1, "--before", 1, "--after", 1,
                                   "--json")
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(out),
                         span_window(self.records, 1, before=1, after=1))

    def test_json_window_shape(self):
        code, out, err = self._run(1, "--before", 1, "--after", 1,
                                   "--json")
        self.assertEqual(code, 0)
        window = json.loads(out)
        self.assertEqual(window["seq"], 1)
        self.assertEqual((window["before"], window["after"]), (1, 1))
        self.assertEqual(window["count"], 3)
        self.assertEqual(window["total"], 3)
        self.assertEqual([int(s["spanId"], 16)
                          for s in window["spans"]], [0, 1, 2])

    def test_since_narrows_the_session(self):
        path = self._fixed_log()
        # Step 0 is before the bound, so it is not in the session.
        code, out, err = self._run_fixed(
            path, 0, "--since", "2026-11-22T10:01:00Z")
        self.assertEqual(code, 2)
        self.assertEqual(out, "")
        self.assertIn("no step 0", err)
        code, out, err = self._run_fixed(
            path, 1, "--since", "2026-11-22T10:01:00Z", "--json")
        self.assertEqual(code, 0)
        window = json.loads(out)
        self.assertEqual(window["total"], 2)
        self.assertEqual([int(s["spanId"], 16)
                          for s in window["spans"]], [1])

    def test_until_narrows_the_session(self):
        path = self._fixed_log()
        code, out, err = self._run_fixed(
            path, 2, "--until", "2026-11-22T10:01:00Z")
        self.assertEqual(code, 2)
        self.assertEqual(out, "")
        self.assertIn("no step 2", err)
        # The window clamps at the narrowed session's end: step 2
        # is dropped, so --after 1 past step 1 adds nothing.
        code, out, err = self._run_fixed(
            path, 1, "--until", "2026-11-22T10:01:00Z", "--after", 1,
            "--json")
        self.assertEqual(code, 0)
        window = json.loads(out)
        self.assertEqual(window["count"], 1)
        self.assertEqual(window["total"], 2)

    def test_last_narrows_the_session(self):
        path = self._fixed_log()
        # Every step is days old, so --last 10m keeps none of them.
        code, out, err = self._run_fixed(path, 0, "--last", "10m")
        self.assertEqual(code, 2)
        self.assertEqual(out, "")
        self.assertIn("no step 0", err)
        # The setUp log was recorded just now, so --last 1h keeps
        # the whole session.
        code, out, err = self._run(1, "--last", "1h", "--json")
        self.assertEqual(code, 0)
        window = json.loads(out)
        self.assertEqual(window["total"], 3)

    def test_negative_before_or_after_is_an_error(self):
        for argv in ((1, "--before", -1), (1, "--after", -1)):
            code, out, err = self._run(*argv)
            self.assertEqual(code, 2, argv)
            self.assertEqual(out, "")
            self.assertIn("non-negative", err)

    def test_window_over_100_is_an_error(self):
        code, out, err = self._run(1, "--before", 60, "--after", 50)
        self.assertEqual(code, 2)
        self.assertEqual(out, "")
        self.assertIn("at most 100", err)

    def test_window_of_exactly_100_is_allowed(self):
        code, out, err = self._run(1, "--before", 50, "--after", 49,
                                   "--json")
        self.assertEqual(code, 0)
        window = json.loads(out)
        self.assertEqual(window["count"], 3)

    def test_bad_bounds_are_an_error(self):
        cases = (
            (1, "--since", "2026-11-24T01:00:00"),  # no timezone
            (1, "--last", "10x"),  # not a duration
            (1, "--since", "2026-11-24T02:00:00Z",
             "--until", "2026-11-24T01:00:00Z"),  # reversed
            (1, "--last", "10m",
             "--since", "2026-11-24T01:00:00Z"),  # --last + --since
            (1, "--last", "10m",
             "--until", "2026-11-24T01:00:00Z"),  # --last + --until
        )
        for argv in cases:
            code, out, err = self._run(*argv)
            self.assertEqual(code, 2, argv)
            self.assertEqual(out, "")

    def test_summary_seq_in_window_is_an_error(self):
        with Recorder(self.path, task="t", model="gpt-5"):
            pass  # closing writes the summary record (seq 3)
        code, out, err = self._run(3, "--before", 1)
        self.assertEqual(code, 2)
        self.assertEqual(out, "")
        self.assertIn("no step 3", err)


def _nanos(ts):
    from datetime import datetime, timezone
    dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
    delta = dt - epoch
    seconds = delta.days * 86400 + delta.seconds
    return str(seconds * 1_000_000_000 + delta.microseconds * 1000)


if __name__ == "__main__":
    unittest.main()
