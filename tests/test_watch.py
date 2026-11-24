"""`blackbox watch`: tail a live log and print new steps as they land,
in the same format as the `blackbox replay` transcript. By default
watching starts at the end and prints only new steps; --from-start
replays the existing steps first. --type keeps only steps of that
type, --errors-only only steps that look like errors (the same rule
as `blackbox query`), --model keeps only model_call steps for one
model and --tool keeps only tool_call / tool_result steps for one
tool (the same match rules as `blackbox query`), and --since /
--until keep only steps whose timestamp falls in the closed window
[since, until] (the same window as `blackbox query`, parsed the way
`blackbox record --ts` parses a timestamp). --last keeps only steps
from the last N seconds/minutes/hours/days, e.g. --last 10m, as a
lower bound of now minus that duration — it cannot be combined
with --since or --until, and a value that is not a duration is an
error, exit code 2. A malformed appended line is reported, not a
crash; a log that shrinks while watched (a
Recorder resuming drops the trailing summary) is re-read without
reprinting what was already printed; Ctrl-C stops cleanly with exit
code 0."""

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
from datetime import datetime, timedelta, timezone
from unittest import mock

from blackbox import Recorder
from blackbox.__main__ import main
from blackbox.replay import format_record, load_records
from blackbox.watch import _parse_duration, watch_log

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
                   max_records=None, types=None, errors_only=False,
                   model=None, tool=None, min_tokens=None, min_cost=None,
                   since=None, until=None, last=None):
        """Start watch_log in a thread; returns (thread, result, out, err)."""
        out, err = io.StringIO(), io.StringIO()
        result = {}

        def target():
            result["count"] = watch_log(
                path or self.path, from_start=from_start,
                poll_interval=0.05, output=out, error_output=err,
                stop_after=stop_after, max_records=max_records,
                types=types, errors_only=errors_only,
                model=model, tool=tool, min_tokens=min_tokens,
                min_cost=min_cost, since=since, until=until, last=last)

        thread = threading.Thread(target=target)
        thread.start()
        return thread, result, out, err

    def _assert_no_bad_line_report(self, err):
        """No malformed-line report on stderr. A Recorder resuming
        drops the trailing summary, so the log shrinks; a poll that
        lands mid-resume legitimately reports the shrink and
        re-reads, so stderr is not asserted empty here — only that
        no line was called malformed, which would mean a real
        desync bug."""
        self.assertNotIn("not valid JSON", err.getvalue())
        self.assertNotIn("not a blackbox record", err.getvalue())

    def _timed_log(self, path=None):
        """A log with steps stamped at fixed times: model_call at
        10:00, tool_call at 11:00, tool_result at 12:00, all on
        2026-11-24 UTC, plus the summary the Recorder writes on
        close (stamped with the real now). Returns the path."""
        path = path or os.path.join(self.tmp.name, "timed.jsonl")
        with Recorder(path, task="t", model="gpt-5") as rec:
            rec.record("model_call", {"model": "gpt-5", "prompt": "hi"},
                       tokens={"input": 10, "output": 2},
                       ts="2026-11-24T10:00:00Z")
            rec.record("tool_call", {"tool": "search",
                                     "args": {"query": "mcp"}},
                       ts="2026-11-24T11:00:00Z")
            rec.record("tool_result", {"tool": "search",
                                       "result": "3 results"},
                       ts="2026-11-24T12:00:00Z")
        return path

    def _recent_log(self, path=None):
        """A log with steps stamped relative to now: model_call 20
        minutes ago, tool_call 5 minutes ago, tool_result just now,
        plus the summary the Recorder writes on close (stamped with
        the real now). Returns the path."""
        path = path or os.path.join(self.tmp.name, "recent.jsonl")
        now = datetime.now(timezone.utc)

        def _ts(minutes_ago):
            return (now - timedelta(minutes=minutes_ago)).strftime(
                "%Y-%m-%dT%H:%M:%SZ")

        with Recorder(path, task="t", model="gpt-5") as rec:
            rec.record("model_call", {"model": "gpt-5", "prompt": "hi"},
                       tokens={"input": 10, "output": 2}, ts=_ts(20))
            rec.record("tool_call", {"tool": "search",
                                     "args": {"query": "mcp"}},
                       ts=_ts(5))
            rec.record("tool_result", {"tool": "search",
                                       "result": "3 results"},
                       ts=_ts(0))
        return path

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
        self._assert_no_bad_line_report(err)
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
        # A poll landing mid-resume may legitimately report the
        # shrink itself, so only the malformed-line report is
        # forbidden here.
        self._assert_no_bad_line_report(err)

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
        self._assert_no_bad_line_report(err)
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

    def test_watch_model_filter_keeps_only_matching_model(self):
        # Two model calls, different models, plus a tool step: only
        # the gpt-5 call prints, and the match is case-insensitive —
        # the same rule as `blackbox query`. The summary is dropped,
        # like in `blackbox query`.
        path = os.path.join(self.tmp.name, "models.jsonl")
        with Recorder(path, task="t", model="gpt-5") as rec:
            rec.record("model_call", {"model": "gpt-5", "prompt": "hi"},
                       tokens={"input": 10, "output": 2})
            rec.record("model_call",
                       {"model": "claude-sonnet-4-5", "prompt": "hi"},
                       tokens={"input": 10, "output": 2})
            rec.record("tool_call", {"tool": "search"})
        thread, result, out, err = self._run_watch(
            path=path, from_start=True, model="GPT-5", max_records=1)
        thread.join(timeout=10)
        self.assertFalse(thread.is_alive())
        self.assertEqual(result["count"], 1)
        self.assertEqual(err.getvalue(), "")
        lines = out.getvalue()
        self.assertIn("[0]", lines)
        self.assertIn("model_call", lines)
        self.assertIn("gpt-5", lines)
        self.assertNotIn("claude", lines)
        self.assertNotIn("[1]", lines)
        self.assertNotIn("[2]", lines)
        self.assertNotIn("events:", lines)

    def test_watch_tool_filter_keeps_only_matching_tool(self):
        # Fixture: model_call [0], tool_call search [1], tool_result
        # search [2], summary [3]. --tool search keeps both search
        # steps and nothing else.
        thread, result, out, err = self._run_watch(
            from_start=True, tool="SEARCH", max_records=2)
        thread.join(timeout=10)
        self.assertFalse(thread.is_alive())
        self.assertEqual(result["count"], 2)
        self.assertEqual(err.getvalue(), "")
        lines = out.getvalue()
        self.assertIn("[1]", lines)
        self.assertIn("tool_call", lines)
        self.assertIn("[2]", lines)
        self.assertIn("tool_result", lines)
        self.assertNotIn("[0]", lines)
        self.assertNotIn("events:", lines)

    def test_watch_model_filter_sqlite_backend(self):
        path = os.path.join(self.tmp.name, "models.db")
        with Recorder(path, task="t", model="gpt-5") as rec:
            rec.record("model_call", {"model": "gpt-5", "prompt": "hi"},
                       tokens={"input": 10, "output": 2})
            rec.record("model_call",
                       {"model": "claude-sonnet-4-5", "prompt": "hi"},
                       tokens={"input": 10, "output": 2})
            rec.record("tool_call", {"tool": "search"})
        thread, result, out, err = self._run_watch(
            path=path, from_start=True, model="claude-sonnet-4-5",
            max_records=1)
        thread.join(timeout=10)
        self.assertFalse(thread.is_alive())
        self.assertEqual(result["count"], 1)
        self.assertEqual(err.getvalue(), "")
        lines = out.getvalue()
        self.assertIn("[1]", lines)
        self.assertIn("claude-sonnet-4-5", lines)
        self.assertNotIn("[0]", lines)
        self.assertNotIn("[2]", lines)
        self.assertNotIn("events:", lines)

    def test_watch_model_and_type_filters_combine_with_and(self):
        # --model gpt-5 with --type tool_call matches nothing: a
        # model_call is not a tool_call, and the tool steps are not
        # gpt-5 calls. Nothing prints and the watch still stops.
        thread, result, out, err = self._run_watch(
            from_start=True, model="gpt-5", types=["tool_call"],
            stop_after=0.8)
        thread.join(timeout=10)
        self.assertFalse(thread.is_alive())
        self.assertEqual(result["count"], 0)
        self.assertEqual(out.getvalue(), "")
        self.assertEqual(err.getvalue(), "")

    def test_watch_tool_and_errors_only_combine_with_and(self):
        # A clean search result lands, then an error fetch result:
        # --tool search --errors-only prints nothing for the clean
        # search steps, and the fetch error is the wrong tool.
        thread, result, out, err = self._run_watch(
            tool="search", errors_only=True, max_records=1)
        time.sleep(0.3)
        with Recorder(self.path, task="t", model="gpt-5") as rec:
            rec.record("tool_result",
                       {"tool": "search", "result": "error: timeout"})
        thread.join(timeout=10)
        self.assertFalse(thread.is_alive())
        self.assertEqual(result["count"], 1)
        self._assert_no_bad_line_report(err)
        lines = out.getvalue()
        self.assertIn("tool_result", lines)
        self.assertIn("timeout", lines)
        self.assertNotIn("events:", lines)

    def test_watch_min_tokens_keeps_only_heavy_steps(self):
        # Fixture: model_call [0] with 1540 tokens, tool_call [1]
        # and tool_result [2] with none, summary [3]. --min-tokens
        # 100 keeps only the model call; the summary is dropped,
        # like in `blackbox query`.
        thread, result, out, err = self._run_watch(
            from_start=True, min_tokens=100, max_records=1)
        thread.join(timeout=10)
        self.assertFalse(thread.is_alive())
        self.assertEqual(result["count"], 1)
        self.assertEqual(err.getvalue(), "")
        lines = out.getvalue()
        self.assertIn("[0]", lines)
        self.assertIn("model_call", lines)
        self.assertNotIn("[1]", lines)
        self.assertNotIn("[2]", lines)
        self.assertNotIn("events:", lines)

    def test_watch_min_tokens_zero_still_drops_summary(self):
        # --min-tokens 0 matches every step (even one with no
        # tokens), but the summary is a roll-up, not a step, so it
        # is still dropped under a filter — the same rule as
        # `blackbox query`.
        thread, result, out, err = self._run_watch(
            from_start=True, min_tokens=0, max_records=3)
        thread.join(timeout=10)
        self.assertFalse(thread.is_alive())
        self.assertEqual(result["count"], 3)
        self.assertEqual(err.getvalue(), "")
        lines = out.getvalue()
        self.assertIn("[0]", lines)
        self.assertIn("[1]", lines)
        self.assertIn("[2]", lines)
        self.assertNotIn("events:", lines)

    def test_watch_min_cost_keeps_only_expensive_calls(self):
        # gpt-5 1200/340 costs $0.0049, gpt-5-nano 10/2 costs
        # $0.000001, and a plain int token count has no cost at
        # all — so --min-cost 0.004 keeps only the gpt-5 call,
        # the same rule as `blackbox query`.
        path = os.path.join(self.tmp.name, "costs.jsonl")
        with Recorder(path, task="t", model="gpt-5") as rec:
            rec.record("model_call", {"model": "gpt-5", "prompt": "hi"},
                       tokens={"input": 1200, "output": 340})
            rec.record("model_call",
                       {"model": "gpt-5-nano", "prompt": "hi"},
                       tokens={"input": 10, "output": 2})
            rec.record("model_call", {"model": "gpt-5", "prompt": "hi"},
                       tokens=5000)
        thread, result, out, err = self._run_watch(
            path=path, from_start=True, min_cost=0.004, stop_after=0.8)
        thread.join(timeout=10)
        self.assertFalse(thread.is_alive())
        self.assertEqual(result["count"], 1)
        self.assertEqual(err.getvalue(), "")
        lines = out.getvalue()
        self.assertIn("[0]", lines)
        self.assertIn("gpt-5", lines)
        self.assertNotIn("gpt-5-nano", lines)
        self.assertNotIn("[1]", lines)
        self.assertNotIn("[2]", lines)
        self.assertNotIn("events:", lines)

    def test_watch_min_cost_sqlite_backend(self):
        path = os.path.join(self.tmp.name, "costs.db")
        with Recorder(path, task="t", model="gpt-5") as rec:
            rec.record("model_call", {"model": "gpt-5", "prompt": "hi"},
                       tokens={"input": 1200, "output": 340})
            rec.record("model_call",
                       {"model": "gpt-5-nano", "prompt": "hi"},
                       tokens={"input": 10, "output": 2})
        thread, result, out, err = self._run_watch(
            path=path, from_start=True, min_cost=0.004, stop_after=0.8)
        thread.join(timeout=10)
        self.assertFalse(thread.is_alive())
        self.assertEqual(result["count"], 1)
        self.assertEqual(err.getvalue(), "")
        lines = out.getvalue()
        self.assertIn("[0]", lines)
        self.assertNotIn("gpt-5-nano", lines)
        self.assertNotIn("events:", lines)

    def test_watch_min_tokens_and_type_combine_with_and(self):
        # --min-tokens 100 with --type tool_call matches nothing:
        # the heavy step is a model_call, and the tool steps carry
        # no tokens. Nothing prints and the watch still stops.
        thread, result, out, err = self._run_watch(
            from_start=True, min_tokens=100, types=["tool_call"],
            stop_after=0.8)
        thread.join(timeout=10)
        self.assertFalse(thread.is_alive())
        self.assertEqual(result["count"], 0)
        self.assertEqual(out.getvalue(), "")
        self.assertEqual(err.getvalue(), "")

    def test_watch_since_until_keeps_only_window(self):
        # Steps at 10:00, 11:00, 12:00 plus the summary. --since
        # 11:00 --until 12:00 keeps the 11:00 and 12:00 steps — both
        # bounds inclusive, the same closed window as `blackbox
        # query` — and drops the 10:00 step and the summary.
        path = self._timed_log()
        thread, result, out, err = self._run_watch(
            path=path, from_start=True,
            since="2026-11-24T11:00:00Z",
            until="2026-11-24T12:00:00Z", max_records=2)
        thread.join(timeout=10)
        self.assertFalse(thread.is_alive())
        self.assertEqual(result["count"], 2)
        self.assertEqual(err.getvalue(), "")
        lines = out.getvalue()
        self.assertIn("[1]", lines)
        self.assertIn("tool_call", lines)
        self.assertIn("[2]", lines)
        self.assertIn("tool_result", lines)
        self.assertNotIn("[0]", lines)
        self.assertNotIn("events:", lines)

    def test_watch_since_until_boundaries_are_inclusive(self):
        # --since and --until at the same instant keep exactly the
        # step stamped at that instant: both bounds are inclusive.
        path = self._timed_log()
        thread, result, out, err = self._run_watch(
            path=path, from_start=True,
            since="2026-11-24T11:00:00Z",
            until="2026-11-24T11:00:00Z", max_records=1)
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

    def test_watch_since_until_sqlite_backend(self):
        path = self._timed_log(os.path.join(self.tmp.name, "timed.db"))
        thread, result, out, err = self._run_watch(
            path=path, from_start=True,
            since="2026-11-24T11:00:00Z",
            until="2026-11-24T12:00:00Z", max_records=2)
        thread.join(timeout=10)
        self.assertFalse(thread.is_alive())
        self.assertEqual(result["count"], 2)
        self.assertEqual(err.getvalue(), "")
        lines = out.getvalue()
        self.assertIn("[1]", lines)
        self.assertIn("tool_call", lines)
        self.assertIn("[2]", lines)
        self.assertIn("tool_result", lines)
        self.assertNotIn("[0]", lines)
        self.assertNotIn("events:", lines)

    def test_watch_since_until_and_type_combine_with_and(self):
        # --since 10:00 --until 12:00 --type tool_call keeps only
        # the tool steps in the window: the 10:00 model_call is in
        # the window but the wrong type, and the summary is dropped
        # under a filter, like in `blackbox query`.
        path = self._timed_log()
        thread, result, out, err = self._run_watch(
            path=path, from_start=True,
            since="2026-11-24T10:00:00Z",
            until="2026-11-24T12:00:00Z", types=["tool_call"],
            max_records=1)
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

    def test_watch_since_with_offset_is_normalised_to_utc(self):
        # --since given with a +02:00 offset means 11:00 UTC, the
        # same normalisation `blackbox record --ts` applies: the
        # 10:00 step is dropped and the 11:00 and 12:00 steps kept.
        path = self._timed_log()
        thread, result, out, err = self._run_watch(
            path=path, from_start=True,
            since="2026-11-24T13:00:00+02:00",
            until="2026-11-24T12:00:00Z", max_records=2)
        thread.join(timeout=10)
        self.assertFalse(thread.is_alive())
        self.assertEqual(result["count"], 2)
        self.assertEqual(err.getvalue(), "")
        lines = out.getvalue()
        self.assertIn("[1]", lines)
        self.assertIn("[2]", lines)
        self.assertNotIn("[0]", lines)
        self.assertNotIn("events:", lines)

    def test_watch_fractional_second_step_after_whole_second_until(self):
        # A step stamped 11:00:00.5 is after --until 11:00:00, so it
        # is dropped. The bounds compare as datetimes, not strings:
        # a plain string comparison would order ".5" before "Z" and
        # wrongly keep it.
        path = os.path.join(self.tmp.name, "frac.jsonl")
        with Recorder(path, task="t", model="gpt-5") as rec:
            rec.record("tool_call", {"tool": "search"},
                       ts="2026-11-24T11:00:00Z")
            rec.record("tool_call", {"tool": "fetch"},
                       ts="2026-11-24T11:00:00.500000Z")
        thread, result, out, err = self._run_watch(
            path=path, from_start=True,
            until="2026-11-24T11:00:00Z", max_records=1)
        thread.join(timeout=10)
        self.assertFalse(thread.is_alive())
        self.assertEqual(result["count"], 1)
        self.assertEqual(err.getvalue(), "")
        lines = out.getvalue()
        self.assertIn("[0]", lines)
        self.assertIn("search", lines)
        self.assertNotIn("fetch", lines)
        self.assertNotIn("events:", lines)

    def test_watch_time_filtered_step_not_reprinted_after_shrink(self):
        # A live step outside the window is filtered out and marked
        # seen: when the Recorder resumes (dropping the summary, so
        # the log shrinks and is re-read) it must not print, while
        # the in-window step that lands with it does.
        path = self._timed_log()
        thread, result, out, err = self._run_watch(
            path=path, since="2026-11-24T11:30:00Z", stop_after=2.5)
        time.sleep(0.3)
        with Recorder(path, task="t", model="gpt-5") as rec:
            rec.record("tool_call", {"tool": "fetch"},
                       ts="2026-11-24T10:30:00Z")
            rec.record("tool_result", {"tool": "search",
                                       "result": "ok"},
                       ts="2026-11-24T12:00:00Z")
        thread.join(timeout=10)
        self.assertFalse(thread.is_alive())
        self.assertEqual(result["count"], 1)
        lines = out.getvalue()
        self.assertIn("[4]", lines)  # the in-window step
        self.assertIn("tool_result", lines)
        self.assertIn("search", lines)
        self.assertNotIn("fetch", lines)  # filtered out, not reprinted
        self.assertNotIn("[0]", lines)
        self.assertNotIn("[1]", lines)
        self.assertNotIn("events:", lines)

    def test_parse_duration_units(self):
        self.assertEqual(_parse_duration("30s"), timedelta(seconds=30))
        self.assertEqual(_parse_duration("10m"), timedelta(minutes=10))
        self.assertEqual(_parse_duration("2h"), timedelta(hours=2))
        self.assertEqual(_parse_duration("1d"), timedelta(days=1))
        self.assertEqual(_parse_duration("1h30m"),
                         timedelta(hours=1, minutes=30))
        self.assertEqual(_parse_duration("1d2h3m4s"),
                         timedelta(days=1, hours=2, minutes=3,
                                   seconds=4))

    def test_parse_duration_rejects_garbage(self):
        for bad in ("10", "10x", "m", "", "0m", "0s", "-5m", "1.5h",
                    "10 m", "1h30x", "1m1h"):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    _parse_duration(bad)

    def test_watch_last_keeps_only_recent_steps(self):
        # Steps 20 minutes ago, 5 minutes ago and just now, plus
        # the summary. --last 10m keeps the 5-minute and just-now
        # steps and drops the 20-minute step and the summary.
        path = self._recent_log()
        thread, result, out, err = self._run_watch(
            path=path, from_start=True, last="10m", max_records=2)
        thread.join(timeout=10)
        self.assertFalse(thread.is_alive())
        self.assertEqual(result["count"], 2)
        self.assertEqual(err.getvalue(), "")
        lines = out.getvalue()
        self.assertIn("[1]", lines)
        self.assertIn("tool_call", lines)
        self.assertIn("[2]", lines)
        self.assertIn("tool_result", lines)
        self.assertNotIn("[0]", lines)
        self.assertNotIn("events:", lines)

    def test_watch_last_sqlite_backend(self):
        path = self._recent_log(os.path.join(self.tmp.name, "recent.db"))
        thread, result, out, err = self._run_watch(
            path=path, from_start=True, last="10m", max_records=2)
        thread.join(timeout=10)
        self.assertFalse(thread.is_alive())
        self.assertEqual(result["count"], 2)
        self.assertEqual(err.getvalue(), "")
        lines = out.getvalue()
        self.assertIn("[1]", lines)
        self.assertIn("[2]", lines)
        self.assertNotIn("[0]", lines)
        self.assertNotIn("events:", lines)

    def test_watch_last_and_type_combine_with_and(self):
        # --last 10m --type tool_result keeps only the just-now
        # tool_result: the 5-minute tool_call is recent but the
        # wrong type.
        path = self._recent_log()
        thread, result, out, err = self._run_watch(
            path=path, from_start=True, last="10m",
            types=["tool_result"], max_records=1)
        thread.join(timeout=10)
        self.assertFalse(thread.is_alive())
        self.assertEqual(result["count"], 1)
        self.assertEqual(err.getvalue(), "")
        lines = out.getvalue()
        self.assertIn("[2]", lines)
        self.assertIn("tool_result", lines)
        self.assertNotIn("[1]", lines)
        self.assertNotIn("[0]", lines)
        self.assertNotIn("events:", lines)

    def test_watch_last_live_steps_always_pass(self):
        # A step that lands while watching is stamped at now, later
        # than the bound fixed when watching started, so it passes
        # --last even with a short window. The summary is dropped
        # under a filter, like in `blackbox query`.
        thread, result, out, err = self._run_watch(
            last="1m", max_records=1)
        time.sleep(0.3)
        with Recorder(self.path, task="t", model="gpt-5") as rec:
            rec.record("tool_call", {"tool": "fetch"})
        thread.join(timeout=10)
        self.assertFalse(thread.is_alive())
        self.assertEqual(result["count"], 1)
        lines = out.getvalue()
        self.assertIn("[3]", lines)
        self.assertIn("fetch", lines)
        self.assertNotIn("events:", lines)
        self._assert_no_bad_line_report(err)

    def test_watch_last_filtered_step_not_reprinted_after_shrink(self):
        # A live step stamped 20 minutes ago is filtered out by
        # --last and marked seen: when the Recorder resumes
        # (dropping the summary, so the log shrinks and is re-read)
        # it must not print, while the fresh step that lands with
        # it does.
        path = self._recent_log()
        thread, result, out, err = self._run_watch(
            path=path, last="10m", stop_after=2.5)
        time.sleep(0.3)
        now = datetime.now(timezone.utc)

        def _ts(minutes_ago):
            return (now - timedelta(minutes=minutes_ago)).strftime(
                "%Y-%m-%dT%H:%M:%SZ")

        with Recorder(path, task="t", model="gpt-5") as rec:
            rec.record("tool_call", {"tool": "fetch"}, ts=_ts(20))
            rec.record("tool_result", {"tool": "search",
                                       "result": "ok"}, ts=_ts(0))
        thread.join(timeout=10)
        self.assertFalse(thread.is_alive())
        self.assertEqual(result["count"], 1)
        lines = out.getvalue()
        self.assertIn("[4]", lines)  # the fresh step
        self.assertIn("tool_result", lines)
        self.assertIn("search", lines)
        self.assertNotIn("fetch", lines)  # filtered out, not reprinted
        self.assertNotIn("[0]", lines)
        self.assertNotIn("[1]", lines)
        self.assertNotIn("events:", lines)

    def test_watch_cli_passes_last_flag(self):
        with mock.patch("blackbox.__main__.watch_log") as patched:
            code = main(["watch", self.path, "--last", "10m"])
        self.assertEqual(code, 0)
        patched.assert_called_once_with(
            self.path, from_start=False, poll_interval=0.2,
            types=None, errors_only=False,
            model=None, tool=None, min_tokens=None, min_cost=None,
            since=None, until=None, last="10m")

    def test_watch_cli_bad_last_exits_two(self):
        for bad in ("10", "10x"):
            with self.subTest(bad=bad):
                out, err = io.StringIO(), io.StringIO()
                with redirect_stdout(out), redirect_stderr(err):
                    code = main(["watch", self.path, "--last", bad])
                self.assertEqual(code, 2)
                self.assertIn("--last", err.getvalue())

    def test_watch_cli_last_with_since_exits_two(self):
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = main(["watch", self.path, "--last", "10m",
                         "--since", "2026-11-24T11:00:00Z"])
        self.assertEqual(code, 2)
        self.assertIn("--last", err.getvalue())

    def test_watch_cli_last_with_until_exits_two(self):
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = main(["watch", self.path, "--last", "10m",
                         "--until", "2026-11-24T12:00:00Z"])
        self.assertEqual(code, 2)
        self.assertIn("--last", err.getvalue())

    def test_watch_cli_passes_since_and_until_flags(self):
        with mock.patch("blackbox.__main__.watch_log") as patched:
            code = main(["watch", self.path,
                         "--since", "2026-11-24T11:00:00Z",
                         "--until", "2026-11-24T12:00:00Z"])
        self.assertEqual(code, 0)
        patched.assert_called_once_with(
            self.path, from_start=False, poll_interval=0.2,
            types=None, errors_only=False,
            model=None, tool=None, min_tokens=None, min_cost=None,
            since="2026-11-24T11:00:00Z",
            until="2026-11-24T12:00:00Z", last=None)

    def test_watch_cli_bad_since_exits_two(self):
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = main(["watch", self.path, "--since", "not-a-time"])
        self.assertEqual(code, 2)
        self.assertIn("--since", err.getvalue())

    def test_watch_cli_naive_since_exits_two(self):
        # A timestamp with no timezone is an error, never a silent
        # assumption of UTC — the same rule as `blackbox record --ts`.
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = main(["watch", self.path,
                         "--since", "2026-11-24T11:00:00"])
        self.assertEqual(code, 2)
        self.assertIn("--since", err.getvalue())

    def test_watch_cli_bad_until_exits_two(self):
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = main(["watch", self.path, "--until", "yesterday"])
        self.assertEqual(code, 2)
        self.assertIn("--until", err.getvalue())

    def test_watch_cli_since_after_until_exits_two(self):
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = main(["watch", self.path,
                         "--since", "2026-11-24T12:00:00Z",
                         "--until", "2026-11-24T11:00:00Z"])
        self.assertEqual(code, 2)
        self.assertIn("--since", err.getvalue())

    def test_watch_cli_passes_filter_flags(self):
        with mock.patch("blackbox.__main__.watch_log") as patched:
            code = main(["watch", self.path, "--type", "tool_call",
                         "--type", "tool_result", "--errors-only"])
        self.assertEqual(code, 0)
        patched.assert_called_once_with(
            self.path, from_start=False, poll_interval=0.2,
            types=["tool_call", "tool_result"], errors_only=True,
            model=None, tool=None, min_tokens=None, min_cost=None,
            since=None, until=None, last=None)

    def test_watch_cli_passes_model_and_tool_flags(self):
        with mock.patch("blackbox.__main__.watch_log") as patched:
            code = main(["watch", self.path, "--model", "gpt-5",
                         "--tool", "search"])
        self.assertEqual(code, 0)
        patched.assert_called_once_with(
            self.path, from_start=False, poll_interval=0.2,
            types=None, errors_only=False,
            model="gpt-5", tool="search", min_tokens=None,
            min_cost=None, since=None, until=None, last=None)

    def test_watch_cli_passes_min_tokens_and_min_cost_flags(self):
        with mock.patch("blackbox.__main__.watch_log") as patched:
            code = main(["watch", self.path, "--min-tokens", "100",
                         "--min-cost", "0.01"])
        self.assertEqual(code, 0)
        patched.assert_called_once_with(
            self.path, from_start=False, poll_interval=0.2,
            types=None, errors_only=False,
            model=None, tool=None, min_tokens=100, min_cost=0.01,
            since=None, until=None, last=None)

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
