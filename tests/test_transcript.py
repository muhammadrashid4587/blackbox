"""`blackbox transcript`: one step's transcript line, or a window
of lines around it — the same lines the blackbox_transcript MCP
tool returns and the blackbox://session/transcript/{seq} template
serves, readable without an MCP client. The report form and --json
agree with `blackbox replay`; an unknown step, the summary, a
window over 100 lines, or a bad window bound is an error, never an
invented line."""

import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout

from blackbox import Recorder
from blackbox.__main__ import main
from blackbox.replay import (load_records, replay, session_transcript_step,
                             transcript_window)


class TranscriptCliTest(unittest.TestCase):
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
            code = main(["transcript", self.path] + [str(a) for a in argv])
        return code, out.getvalue(), err.getvalue()

    def test_report_is_the_replay_line(self):
        code, out, err = self._run(1)
        self.assertEqual(code, 0)
        self.assertEqual(err, "")
        self.assertEqual(out, session_transcript_step(self.records, 1)
                         + "\n")
        self.assertIn("tool_call", out)
        self.assertIn("search(query=mcp)", out)

    def test_model_call_line_carries_its_prompt(self):
        code, out, err = self._run(0)
        self.assertEqual(code, 0)
        self.assertIn("model_call", out)
        self.assertIn("prompt: hi", out)

    def test_window_around_seq_is_in_log_order(self):
        code, out, err = self._run(1, "--before", "1", "--after", "1")
        self.assertEqual(code, 0)
        lines = out.splitlines()
        self.assertEqual(len(lines), 4)  # model_call line is two lines
        self.assertIn("[0]", lines[0])
        self.assertIn("[1]", lines[2])
        self.assertIn("[2]", lines[3])

    def test_window_clamps_at_the_session_ends(self):
        code, out, err = self._run(0, "--before", "5")
        self.assertEqual(code, 0)
        self.assertIn("[0]", out)
        self.assertNotIn("[1]", out)
        code, out, err = self._run(2, "--after", "5")
        self.assertEqual(code, 0)
        self.assertIn("[2]", out)
        self.assertNotIn("[1]", out)

    def test_json_matches_transcript_window(self):
        code, out, err = self._run(1, "--before", "1", "--after", "1",
                                   "--json")
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(out),
                         transcript_window(self.records, 1,
                                           before=1, after=1))
        data = json.loads(out)
        self.assertEqual(data["seq"], 1)
        self.assertEqual((data["before"], data["after"]), (1, 1))
        self.assertEqual(data["count"], 3)
        self.assertEqual(data["total"], 3)
        self.assertEqual([line["seq"] for line in data["lines"]],
                         [0, 1, 2])

    def test_json_lines_agree_with_replay(self):
        # The same lines `blackbox replay` prints for the whole
        # session, one per step (the summary is bookkeeping, not a
        # step, so it is not in the window).
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            replay(self.path)
        replay_lines = out.getvalue().splitlines()
        code, out, err = self._run(1, "--before", "1", "--after", "1",
                                   "--json")
        self.assertEqual(code, 0)
        data = json.loads(out)
        joined = "\n".join(line["line"] for line in data["lines"])
        self.assertEqual(joined, "\n".join(replay_lines[:4]))

    def test_window_narrows_the_session(self):
        # Every step was recorded just now, so a window starting in
        # 2020 keeps the whole session and one ending in 2020 keeps
        # none of it — the step is not in the session.
        code, out, err = self._run(0, "--since", "2020-01-01T00:00:00Z")
        self.assertEqual(code, 0)
        self.assertIn("[0]", out)
        code, out, err = self._run(0, "--until", "2020-01-01T00:00:00Z")
        self.assertEqual(code, 2)
        self.assertEqual(out, "")
        self.assertIn("no step 0", err)
        code, out, err = self._run(2, "--last", "1h")
        self.assertEqual(code, 0)
        self.assertIn("[2]", out)

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
                main(["transcript", self.path, "abc"])
        self.assertEqual(caught.exception.code, 2)

    def test_negative_before_or_after_is_an_error(self):
        for flag in ("--before", "--after"):
            code, out, err = self._run(1, flag, "-1")
            self.assertEqual(code, 2, flag)
            self.assertEqual(out, "")
            self.assertIn("non-negative", err)

    def test_window_over_100_lines_is_an_error(self):
        code, out, err = self._run(1, "--before", "60", "--after", "50")
        self.assertEqual(code, 2)
        self.assertEqual(out, "")
        self.assertIn("at most 100", err)

    def test_window_of_exactly_100_lines_is_allowed(self):
        code, out, err = self._run(1, "--before", "50", "--after", "49",
                                   "--json")
        self.assertEqual(code, 0)
        data = json.loads(out)
        self.assertEqual(data["count"], 3)

    def test_bad_window_bounds_are_an_error(self):
        for argv in (
                ["--since", ""],
                ["--until", ""],
                ["--last", ""],
                ["--since", "2026-11-24T01:00:00"],  # no timezone
                ["--last", "10x"],  # not a duration
                ["--since", "2026-11-24T02:00:00Z",
                 "--until", "2026-11-24T01:00:00Z"],  # reversed
                ["--last", "10m",
                 "--since", "2026-11-24T01:00:00Z"],
                ["--last", "10m",
                 "--until", "2026-11-24T01:00:00Z"],
        ):
            code, out, err = self._run(1, *argv)
            self.assertEqual(code, 2, argv)
            self.assertEqual(out, "")

    def test_missing_file_is_an_error(self):
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = main(["transcript",
                         os.path.join(self.tmp.name, "nope.jsonl"), "0"])
        self.assertEqual(code, 1)
        self.assertEqual(out.getvalue(), "")
        self.assertIn("file not found", err.getvalue())


if __name__ == "__main__":
    unittest.main()
