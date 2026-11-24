"""`blackbox watch`: tail a live log and print new steps as they land,
in the same format as the `blackbox replay` transcript. By default
watching starts at the end and prints only new steps; --from-start
replays the existing steps first. --type keeps only steps of that
type, --errors-only only steps that look like errors (the same rule
as `blackbox query`). A malformed appended line is
reported, not a crash; a log that shrinks while watched (a Recorder
resuming drops the trailing summary) is re-read without reprinting
what was already printed; Ctrl-C stops cleanly with exit code 0."""

import io
import json
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest import mock

from blackbox import Recorder
from blackbox.__main__ import main
from blackbox.replay import format_record, load_records
from blackbox.watch import watch_log

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class WatchTest(unittest.TestCase):
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

    def tearDown(self):
        self.tmp.cleanup()

    def _run_watch(self, path=None, from_start=False, stop_after=5.0,
                   max_records=None, types=None, errors_only=False):
        """Start watch_log in a thread; returns (thread, result, out, err)."""
        out, err = io.StringIO(), io.StringIO()
        result = {}

        def target():
            result["count"] = watch_log(
                path or self.path, from_start=from_start,
                poll_interval=0.05, output=out, error_output=err,
                stop_after=stop_after, max_records=max_records,
                types=types, errors_only=errors_only)

        thread = threading.Thread(target=target)
        thread.start()
        return thread, result, out, err

    def test_watch_prints_appended_steps(self):
        # Two steps + the summary the Recorder writes on close.
        thread, result, out, err = self._run_watch(max_records=3)
        time.sleep(0.3)
        with Recorder(self.path, task="t", model="gpt-5") as rec:
            rec.record("tool_call", {"tool": "fetch",
                                     "args": {"url": "https://x.co"}})
            rec.record("tool_result", {"tool": "fetch", "result": "ok"})
        thread.join(timeout=10)
        self.assertFalse(thread.is_alive())
        self.assertEqual(result["count"], 3)
        self.assertEqual(err.getvalue(), "")
        lines = out.getvalue()
        self.assertIn("[3]", lines)
        self.assertIn("tool_call", lines)
        self.assertIn("fetch(url=", lines)
        self.assertIn("[4]", lines)
        self.assertIn("tool_result", lines)
        self.assertIn("[5]", lines)  # the summary
        self.assertIn("events: 5", lines)
        # Each step printed once, in the transcript's format.
        self.assertEqual(lines.count("[3]"), 1)
        self.assertEqual(lines.count("[4]"), 1)
        records = load_records(self.path)
        self.assertEqual(lines.splitlines()[0],
                         format_record(records[3]).splitlines()[0])

    def test_watch_from_start_replays_existing_steps(self):
        thread, result, out, err = self._run_watch(from_start=True,
                                                   max_records=4)
        thread.join(timeout=10)
        self.assertFalse(thread.is_alive())
        self.assertEqual(result["count"], 4)
        self.assertEqual(err.getvalue(), "")
        lines = out.getvalue()
        self.assertIn("[0]", lines)
        self.assertIn("model_call", lines)
        self.assertIn("[1]", lines)
        self.assertIn("tool_call", lines)
        self.assertIn("[2]", lines)
        self.assertIn("tool_result", lines)
        self.assertIn("[3]", lines)  # the summary
        self.assertEqual(lines.count("[0]"), 1)

    def test_watch_follow_skips_existing_steps(self):
        thread, result, out, err = self._run_watch(stop_after=0.8)
        thread.join(timeout=10)
        self.assertFalse(thread.is_alive())
        self.assertEqual(result["count"], 0)
        self.assertEqual(out.getvalue(), "")
        self.assertEqual(err.getvalue(), "")

    def test_watch_malformed_appended_line_reported_not_crashed(self):
        # A garbage line, then a hand-built valid step (the watcher
        # prints records, it does not verify the chain).
        thread, result, out, err = self._run_watch(max_records=1)
        time.sleep(0.3)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write("{not json\n")
            f.write(json.dumps(
                {"seq": 99, "ts": "2026-11-24T03:00:00Z",
                 "type": "tool_call",
                 "payload": {"tool": "search", "args": {"query": "mcp"}},
                 "tokens": None, "prev_hash": "0" * 64,
                 "hash": "a" * 64}) + "\n")
        thread.join(timeout=10)
        self.assertFalse(thread.is_alive())
        self.assertEqual(result["count"], 1)
        # Line 5 is the garbage line (3 steps + summary before it).
        self.assertIn("line 5", err.getvalue())
        self.assertIn("not valid JSON", err.getvalue())
        self.assertIn("[99]", out.getvalue())
        self.assertIn("tool_call", out.getvalue())
        self.assertIn("search(query=", out.getvalue())

    def test_watch_non_record_line_reported_not_crashed(self):
        thread, result, out, err = self._run_watch(max_records=1)
        time.sleep(0.3)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"hello": "world"}) + "\n")
            f.write(json.dumps(
                {"seq": 99, "ts": "2026-11-24T03:00:00Z",
                 "type": "tool_call", "payload": {"tool": "search"},
                 "tokens": None, "prev_hash": "0" * 64,
                 "hash": "b" * 64}) + "\n")
        thread.join(timeout=10)
        self.assertFalse(thread.is_alive())
        self.assertEqual(result["count"], 1)
        self.assertIn("not a blackbox record", err.getvalue())
        self.assertIn("[99]", out.getvalue())

    def test_watch_shrinking_log_is_not_reprinted(self):
        # A Recorder resuming drops the trailing summary, so the file
        # shrinks; the watcher re-reads from the start but must not
        # reprint the steps it already saw. The new step reuses the
        # dropped summary's seq, so the hash — not the seq — is what
        # tells it apart.
        thread, result, out, err = self._run_watch(max_records=2)
        time.sleep(0.3)
        with Recorder(self.path, task="t", model="gpt-5") as rec:
            rec.record("tool_call", {"tool": "fetch"})
        thread.join(timeout=10)
        self.assertFalse(thread.is_alive())
        self.assertEqual(result["count"], 2)  # the step + the summary
        lines = out.getvalue()
        self.assertNotIn("[0]", lines)
        self.assertNotIn("[1]", lines)
        self.assertNotIn("[2]", lines)
        self.assertIn("[3]", lines)  # the step, reusing the summary's seq
        self.assertIn("fetch", lines)
        self.assertIn("[4]", lines)  # the new summary
        self.assertEqual(lines.count("[3]"), 1)
        # The stale byte offset (the resume rewrote the log between
        # polls) must not surface as a bogus malformed-line report.
        self.assertEqual(err.getvalue(), "")

    def test_watch_truncated_log_reprints_only_new_steps(self):
        # A rotation that truncates the log to its first two lines,
        # then a resumed run appends a step: the two kept steps must
        # not reprint, the new step must land.
        thread, result, out, err = self._run_watch(max_records=2)
        time.sleep(0.3)
        with open(self.path, "r", encoding="utf-8") as f:
            kept = f.readlines()[:2]
        with open(self.path, "w", encoding="utf-8") as f:
            f.writelines(kept)
        with Recorder(self.path, task="t", model="gpt-5") as rec:
            rec.record("tool_call", {"tool": "fetch"})
        thread.join(timeout=10)
        self.assertFalse(thread.is_alive())
        self.assertEqual(result["count"], 2)  # the step + the summary
        lines = out.getvalue()
        self.assertNotIn("[0]", lines)
        self.assertNotIn("[1]", lines)
        self.assertIn("[2]", lines)
        self.assertIn("fetch", lines)
        self.assertIn("shrank", err.getvalue())

    def test_watch_sqlite_backend(self):
        path = os.path.join(self.tmp.name, "run.db")
        with Recorder(path, task="t", model="gpt-5") as rec:
            rec.record("model_call", {"model": "gpt-5", "prompt": "hi"},
                       tokens={"input": 1200, "output": 340})
            rec.record("tool_call", {"tool": "search",
                                     "args": {"query": "mcp"}})
        thread, result, out, err = self._run_watch(
            path=path, max_records=3)
        time.sleep(0.3)
        with Recorder(path, task="t", model="gpt-5") as rec:
            rec.record("tool_result", {"tool": "search",
                                       "result": "3 results"})
            rec.record("tool_call", {"tool": "fetch"})
        thread.join(timeout=10)
        self.assertFalse(thread.is_alive())
        self.assertEqual(result["count"], 3)  # 2 steps + summary
        lines = out.getvalue()
        self.assertIn("[2]", lines)
        self.assertIn("tool_result", lines)
        self.assertIn("[3]", lines)
        self.assertIn("tool_call", lines)
        self.assertIn("[4]", lines)  # the summary
        self.assertEqual(lines.count("[2]"), 1)
        self.assertEqual(lines.count("[3]"), 1)

    def test_watch_sqlite_from_start(self):
        path = os.path.join(self.tmp.name, "run.db")
        with Recorder(path, task="t", model="gpt-5") as rec:
            rec.record("tool_call", {"tool": "search"})
        thread, result, out, err = self._run_watch(
            path=path, from_start=True, max_records=2)
        thread.join(timeout=10)
        self.assertFalse(thread.is_alive())
        self.assertEqual(result["count"], 2)  # the step + the summary
        self.assertIn("[0]", out.getvalue())
        self.assertIn("tool_call", out.getvalue())

    def test_watch_waits_for_missing_log(self):
        missing = os.path.join(self.tmp.name, "later.jsonl")
        thread, result, out, err = self._run_watch(
            path=missing, from_start=True, max_records=2)
        time.sleep(0.3)
        with Recorder(missing, task="t", model="gpt-5") as rec:
            rec.record("tool_call", {"tool": "search"})
        thread.join(timeout=10)
        self.assertFalse(thread.is_alive())
        self.assertEqual(result["count"], 2)
        self.assertIn("waiting for", err.getvalue())
        self.assertIn("[0]", out.getvalue())
        self.assertIn("tool_call", out.getvalue())

    def test_watch_missing_log_times_out_cleanly(self):
        missing = os.path.join(self.tmp.name, "never.jsonl")
        thread, result, out, err = self._run_watch(
            path=missing, stop_after=0.5)
        thread.join(timeout=10)
        self.assertFalse(thread.is_alive())
        self.assertEqual(result["count"], 0)
        self.assertEqual(out.getvalue(), "")
        self.assertIn("waiting for", err.getvalue())

    def test_watch_bad_poll_interval_is_an_error(self):
        with self.assertRaises(ValueError):
            watch_log(self.path, poll_interval=0)

    def test_watch_type_filter_keeps_only_matching_type(self):
        # Fixture: model_call [0], tool_call [1], tool_result [2],
        # summary [3]. --type tool_call keeps only the tool_call;
        # the summary is dropped, like in `blackbox query`.
        thread, result, out, err = self._run_watch(
            from_start=True, types=["tool_call"], max_records=1)
        thread.join(timeout=10)
        self.assertFalse(thread.is_alive())
        self.assertEqual(result["count"], 1)
        self.assertEqual(err.getvalue(), "")
        lines = out.getvalue()
        self.assertIn("[1]", lines)
        self.assertIn("tool_call", lines)
        self.assertNotIn("[0]", lines)
        self.assertNotIn("[2]", lines)
        self.assertNotIn("events:", lines)

    def test_watch_type_summary_kept_only_when_listed(self):
        thread, result, out, err = self._run_watch(
            from_start=True, types=["summary"], max_records=1)
        thread.join(timeout=10)
        self.assertFalse(thread.is_alive())
        self.assertEqual(result["count"], 1)
        self.assertEqual(err.getvalue(), "")
        lines = out.getvalue()
        self.assertIn("[3]", lines)
        self.assertIn("events:", lines)
        self.assertNotIn("[0]", lines)
        self.assertNotIn("[1]", lines)
        self.assertNotIn("[2]", lines)

    def test_watch_errors_only_skips_clean_steps(self):
        # A clean step lands first, then an error step: the clean
        # step must not count towards max_records, and only the
        # error step is printed.
        thread, result, out, err = self._run_watch(
            errors_only=True, max_records=1)
        time.sleep(0.3)
        with Recorder(self.path, task="t", model="gpt-5") as rec:
            rec.record("tool_call", {"tool": "fetch"})
            rec.record("tool_result",
                       {"tool": "fetch", "result": "error: timeout"})
        thread.join(timeout=10)
        self.assertFalse(thread.is_alive())
        self.assertEqual(result["count"], 1)
        self.assertEqual(err.getvalue(), "")
        lines = out.getvalue()
        self.assertIn("tool_result", lines)
        self.assertIn("timeout", lines)
        self.assertNotIn("tool_call", lines)
        self.assertNotIn("events:", lines)

    def test_watch_type_filter_sqlite_backend(self):
        path = os.path.join(self.tmp.name, "run.db")
        with Recorder(path, task="t", model="gpt-5") as rec:
            rec.record("model_call", {"model": "gpt-5", "prompt": "hi"},
                       tokens={"input": 1200, "output": 340})
            rec.record("tool_call", {"tool": "search"})
        thread, result, out, err = self._run_watch(
            path=path, from_start=True, types=["model_call"],
            max_records=1)
        thread.join(timeout=10)
        self.assertFalse(thread.is_alive())
        self.assertEqual(result["count"], 1)
        self.assertEqual(err.getvalue(), "")
        lines = out.getvalue()
        self.assertIn("[0]", lines)
        self.assertIn("model_call", lines)
        self.assertNotIn("[1]", lines)
        self.assertNotIn("events:", lines)

    def test_watch_cli_passes_filter_flags(self):
        with mock.patch("blackbox.__main__.watch_log") as patched:
            code = main(["watch", self.path, "--type", "tool_call",
                         "--type", "tool_result", "--errors-only"])
        self.assertEqual(code, 0)
        patched.assert_called_once_with(
            self.path, from_start=False, poll_interval=0.2,
            types=["tool_call", "tool_result"], errors_only=True)

    def test_watch_cli_ctrl_c_returns_zero(self):
        with mock.patch("blackbox.__main__.watch_log",
                        side_effect=KeyboardInterrupt):
            code = main(["watch", self.path])
        self.assertEqual(code, 0)

    def test_watch_cli_bad_poll_interval(self):
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = main(["watch", self.path, "--poll-interval", "0"])
        self.assertEqual(code, 2)
        self.assertIn("poll-interval", err.getvalue())

    def test_watch_cli_missing_file_is_an_error(self):
        out, err = io.StringIO(), io.StringIO()
        # A directory is not a log: open() fails instead of waiting.
        with redirect_stdout(out), redirect_stderr(err):
            code = main(["watch", self.tmp.name,
                         "--poll-interval", "0.05"])
        self.assertEqual(code, 1)
        self.assertIn("watch", err.getvalue())

    def test_watch_ctrl_c_exits_zero(self):
        # A real Ctrl-C (SIGINT) against the real CLI in a subprocess.
        proc = subprocess.Popen(
            [sys.executable, "-m", "blackbox", "watch", self.path],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            cwd=REPO_ROOT)
        try:
            time.sleep(0.6)
            self.assertIsNone(proc.poll())
            proc.send_signal(signal.SIGINT)
            rc = proc.wait(timeout=10)
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()
        self.assertEqual(rc, 0)


if __name__ == "__main__":
    unittest.main()
