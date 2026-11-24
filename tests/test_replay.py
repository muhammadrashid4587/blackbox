"""Replay: every step appears in order. Diff: changed tool calls are
flagged, token totals are reported."""

import os
import tempfile
import unittest

from blackbox import Recorder
from blackbox.replay import (diff, diff_records, load_records,
                             model_usage, replay, session_diff)


def _write_run(path, tool="search", query="first", tokens=100):
    with Recorder(path) as rec:
        rec.record("model_call", {"model": "m", "prompt": "plan"}, tokens=50)
        rec.record("tool_call", {"tool": tool, "args": {"query": query}})
        rec.record("tool_result", {"tool": tool, "result": "ok"})
        rec.record("model_call", {"model": "m", "prompt": "summarise"},
                   tokens=tokens)


class ReplayTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "run.jsonl")

    def tearDown(self):
        self.tmp.cleanup()

    def test_replay_contains_every_step_in_order(self):
        _write_run(self.path)
        import io
        from contextlib import redirect_stdout
        buf = io.StringIO()
        with redirect_stdout(buf):
            records = replay(self.path)
        out = buf.getvalue()
        self.assertEqual(len(records), 5)  # 4 steps + summary
        for needle in ("model_call", "tool_call", "tool_result", "summary",
                       "search", "plan", "summarise"):
            self.assertIn(needle, out)
        self.assertLess(out.index("plan"), out.index("search"))
        self.assertLess(out.index("search"), out.index("summarise"))

    def test_diff_flags_changed_tool_call(self):
        path_a = os.path.join(self.tmp.name, "a.jsonl")
        path_b = os.path.join(self.tmp.name, "b.jsonl")
        _write_run(path_a, tool="search", query="first")
        _write_run(path_b, tool="fetch", query="first")
        import io
        from contextlib import redirect_stdout
        buf = io.StringIO()
        with redirect_stdout(buf):
            result = diff(path_a, path_b)
        # Both the tool_call (step 1) and its tool_result (step 2) name
        # the tool, so both differ.
        self.assertEqual(result["changed"], [1, 2])
        self.assertEqual(result["added"], [])
        self.assertEqual(result["removed"], [])
        self.assertIn("CHANGED", buf.getvalue())

    def test_diff_identical_runs(self):
        path_a = os.path.join(self.tmp.name, "a.jsonl")
        path_b = os.path.join(self.tmp.name, "b.jsonl")
        _write_run(path_a)
        _write_run(path_b)
        import io
        from contextlib import redirect_stdout
        buf = io.StringIO()
        with redirect_stdout(buf):
            result = diff(path_a, path_b)
        self.assertEqual(result["changed"], [])
        self.assertIn("IDENTICAL", buf.getvalue())

    def test_diff_token_totals(self):
        path_a = os.path.join(self.tmp.name, "a.jsonl")
        path_b = os.path.join(self.tmp.name, "b.jsonl")
        _write_run(path_a, tokens=100)
        _write_run(path_b, tokens=250)
        result = diff(path_a, path_b)
        self.assertEqual(result["tokens_a"], 150)
        self.assertEqual(result["tokens_b"], 300)
        self.assertEqual(result["changed"], [3])

    def test_diff_added_step(self):
        path_a = os.path.join(self.tmp.name, "a.jsonl")
        path_b = os.path.join(self.tmp.name, "b.jsonl")
        _write_run(path_a)
        with Recorder(path_b) as rec:
            rec.record("model_call", {"model": "m", "prompt": "plan"}, tokens=50)
            rec.record("tool_call", {"tool": "search", "args": {"query": "first"}})
            rec.record("tool_result", {"tool": "search", "result": "ok"})
            rec.record("model_call", {"model": "m", "prompt": "summarise"},
                       tokens=100)
            rec.record("note", {"text": "extra step"})
        result = diff(path_a, path_b)
        self.assertEqual(result["added"], [4])
        self.assertEqual(result["changed"], [])

    def test_diff_records_finds_without_printing(self):
        # The pure computation behind diff(): same findings, no
        # output, so the MCP prompt can format them itself instead
        # of printing onto the stdio channel.
        path_a = os.path.join(self.tmp.name, "a.jsonl")
        path_b = os.path.join(self.tmp.name, "b.jsonl")
        _write_run(path_a, tool="search", query="first")
        _write_run(path_b, tool="fetch", query="first")
        import io
        from contextlib import redirect_stdout
        buf = io.StringIO()
        with redirect_stdout(buf):
            result = diff_records(load_records(path_a),
                                  load_records(path_b))
        self.assertEqual(buf.getvalue(), "")
        self.assertEqual(result["changed"], [1, 2])
        self.assertEqual(result["added"], [])
        self.assertEqual(result["removed"], [])
        self.assertEqual(result["tokens_a"], result["tokens_b"])

    def test_model_usage_splits_priced_tokens(self):
        records = [
            {"type": "model_call", "payload": {"model": "gpt-5"},
             "tokens": {"input": 1200, "output": 340}},
            {"type": "model_call", "payload": {"model": "gpt-5"},
             "tokens": 50},
            {"type": "model_call", "payload": {},
             "tokens": {"input": 10, "output": 5}},
            {"type": "tool_call", "payload": {"tool": "search"},
             "tokens": 99},
        ]
        usage = model_usage(records, default_model="gpt-5-mini")
        self.assertEqual(usage["gpt-5"],
                         {"input_tokens": 1200, "output_tokens": 340,
                          "tokens": 50})
        # No model in the payload falls back to the default model.
        self.assertEqual(usage["gpt-5-mini"],
                         {"input_tokens": 10, "output_tokens": 5,
                          "tokens": 0})
        # Tool calls are not model usage, whatever their tokens say.
        self.assertNotIn("search", usage)

    def test_session_diff_changed_added_per_model_new_error(self):
        # Run A: 4 steps, gpt-5 1300 in + 390 out = $0.005525.
        # Run B: same shape, a different query and an error result
        # at steps 1-2, plus one extra fetch call at the end.
        path_a = os.path.join(self.tmp.name, "a.jsonl")
        path_b = os.path.join(self.tmp.name, "b.jsonl")
        with Recorder(path_a) as rec:
            rec.record("model_call", {"model": "gpt-5",
                                      "prompt": "hi"},
                       tokens={"input": 1200, "output": 340})
            rec.record("tool_call", {"tool": "search",
                                     "args": {"query": "mcp"}})
            rec.record("tool_result", {"tool": "search",
                                       "result": "3 results"})
            rec.record("model_call", {"model": "gpt-5",
                                      "prompt": "again"},
                       tokens={"input": 100, "output": 50})
        with Recorder(path_b) as rec:
            rec.record("model_call", {"model": "gpt-5",
                                      "prompt": "hi"},
                       tokens={"input": 1200, "output": 340})
            rec.record("tool_call", {"tool": "search",
                                     "args": {"query": "other"}})
            rec.record("tool_result", {"tool": "search",
                                       "error": "timed out"})
            rec.record("model_call", {"model": "gpt-5",
                                      "prompt": "again"},
                       tokens={"input": 100, "output": 50})
            rec.record("tool_call", {"tool": "fetch",
                                     "args": {"url":
                                              "https://example.com"}})
        result = session_diff(load_records(path_a),
                              load_records(path_b))
        self.assertEqual(result["run_a"], {"steps": 4, "tokens": 1690})
        self.assertEqual(result["run_b"], {"steps": 5, "tokens": 1690})
        self.assertEqual(result["tokens_diff"], 0)
        self.assertFalse(result["identical"])
        self.assertEqual([c["position"] for c in result["changed"]],
                         [1, 2])
        self.assertEqual(result["changed"][1]["step_a"]["type"],
                         "tool_result")
        self.assertEqual(result["changed"][1]["step_b"]["payload"],
                         {"tool": "search", "error": "timed out"})
        self.assertEqual(len(result["added"]), 1)
        self.assertEqual(result["added"][0]["payload"]["tool"], "fetch")
        self.assertEqual(result["removed"], [])
        gpt5 = result["per_model"]["gpt-5"]
        self.assertEqual(gpt5["tokens_a"], 1690)
        self.assertEqual(gpt5["tokens_b"], 1690)
        self.assertEqual(gpt5["tokens_delta"], 0)
        self.assertEqual(gpt5["cost_usd_a"], 0.005525)
        self.assertEqual(gpt5["cost_usd_b"], 0.005525)
        self.assertEqual(gpt5["cost_usd_delta"], 0)
        self.assertEqual(len(result["new_errors"]), 1)
        self.assertEqual(result["new_errors"][0]["payload"],
                         {"tool": "search", "error": "timed out"})
        self.assertEqual(result["fixed_errors"], [])
        self.assertEqual(result["errors_a"], 0)
        self.assertEqual(result["errors_b"], 1)

    def test_session_diff_fixed_error_removed_step_unpriced(self):
        # Run A has an error step and a model call with a plain int
        # token count on an unknown model; run B fixes the error
        # and drops the model call.
        path_a = os.path.join(self.tmp.name, "a.jsonl")
        path_b = os.path.join(self.tmp.name, "b.jsonl")
        with Recorder(path_a) as rec:
            rec.record("tool_call", {"tool": "search",
                                     "args": {"query": "mcp"}})
            rec.record("tool_result", {"tool": "search",
                                       "error": "upstream 500"})
            rec.record("model_call", {"model": "demo-model",
                                      "prompt": "again"}, tokens=100)
        with Recorder(path_b) as rec:
            rec.record("tool_call", {"tool": "search",
                                     "args": {"query": "mcp"}})
            rec.record("tool_result", {"tool": "search",
                                       "result": "3 results"})
        result = session_diff(load_records(path_a),
                              load_records(path_b))
        self.assertEqual(result["run_a"], {"steps": 3, "tokens": 100})
        self.assertEqual(result["run_b"], {"steps": 2, "tokens": 0})
        self.assertEqual(result["tokens_diff"], -100)
        self.assertEqual([c["position"] for c in result["changed"]],
                         [1])
        self.assertEqual(result["added"], [])
        self.assertEqual(len(result["removed"]), 1)
        self.assertEqual(result["removed"][0]["type"], "model_call")
        # Unknown model, plain int tokens: counted, never priced.
        demo = result["per_model"]["demo-model"]
        self.assertEqual(demo["tokens_a"], 100)
        self.assertEqual(demo["tokens_b"], 0)
        self.assertEqual(demo["tokens_delta"], -100)
        self.assertIsNone(demo["cost_usd_a"])
        self.assertIsNone(demo["cost_usd_b"])
        self.assertIsNone(demo["cost_usd_delta"])
        self.assertEqual(result["new_errors"], [])
        self.assertEqual(len(result["fixed_errors"]), 1)
        self.assertEqual(result["fixed_errors"][0]["payload"],
                         {"tool": "search", "error": "upstream 500"})
        self.assertEqual(result["errors_a"], 1)
        self.assertEqual(result["errors_b"], 0)

    def test_session_diff_identical_runs(self):
        path_a = os.path.join(self.tmp.name, "a.jsonl")
        path_b = os.path.join(self.tmp.name, "b.jsonl")
        _write_run(path_a)
        _write_run(path_b)
        result = session_diff(load_records(path_a),
                              load_records(path_b))
        self.assertTrue(result["identical"])
        self.assertEqual(result["changed"], [])
        self.assertEqual(result["added"], [])
        self.assertEqual(result["removed"], [])
        self.assertEqual(result["tokens_diff"], 0)
        self.assertEqual(result["new_errors"], [])
        self.assertEqual(result["fixed_errors"], [])

    def test_session_diff_empty_runs(self):
        result = session_diff([], [])
        self.assertTrue(result["identical"])
        self.assertEqual(result["run_a"], {"steps": 0, "tokens": 0})
        self.assertEqual(result["run_b"], {"steps": 0, "tokens": 0})
        self.assertEqual(result["per_model"], {})
        self.assertEqual(result["errors_a"], 0)
        self.assertEqual(result["errors_b"], 0)


def _record(seq, event_type, payload, ts="2026-11-22T10:00:00Z"):
    return {"seq": seq, "ts": ts, "type": event_type, "payload": payload,
            "tokens": None, "prev_hash": "0" * 64, "hash": "0" * 64}


def _demo_records():
    return [
        _record(0, "model_call", {"model": "gpt-5", "prompt": "plan"},
                "2026-11-22T10:00:00Z"),
        _record(1, "tool_call", {"tool": "search", "args": {"q": "x"}},
                "2026-11-22T10:01:00Z"),
        _record(2, "tool_result", {"tool": "search", "result": "ok"},
                "2026-11-22T10:02:00Z"),
        _record(3, "model_call", {"model": "gpt-5",
                                  "prompt": "summarise"},
                "2026-11-22T10:03:00Z"),
        _record(4, "tool_call", {"tool": "fetch", "args": {"url": "u"}},
                "2026-11-22T10:04:00Z"),
        _record(5, "tool_result", {"tool": "fetch", "result": "page"},
                "2026-11-22T10:05:00Z"),
        _record(6, "summary",
                {"events": 6, "total_tokens": 0,
                 "started": "2026-11-22T10:00:00Z",
                 "ended": "2026-11-22T10:06:00Z"},
                "2026-11-22T10:06:00Z"),
    ]


def _write_log(directory, records=None):
    """Write records as a .jsonl log and return its path."""
    import json
    path = os.path.join(directory, "run.jsonl")
    with open(path, "w") as handle:
        for record in records if records is not None else _demo_records():
            handle.write(json.dumps(record) + "\n")
    return path


class ReplayWindowTest(unittest.TestCase):
    def test_since_until_narrows_the_transcript(self):
        import io
        from contextlib import redirect_stdout
        buf = io.StringIO()
        with tempfile.TemporaryDirectory() as d:
            path = _write_log(d)
            with redirect_stdout(buf):
                records = replay(path, since="2026-11-22T10:02:00Z",
                                 until="2026-11-22T10:04:00Z")
        # Steps 2-4 only; the summary is bookkeeping about the
        # whole log, so it is kept even though its timestamp is
        # outside the window.
        self.assertEqual(buf.getvalue(),
                         "[2] 2026-11-22T10:02:00Z tool_result  "
                         "search -> ok\n"
                         "[3] 2026-11-22T10:03:00Z model_call  "
                         "model=gpt-5\n"
                         "    prompt: summarise\n"
                         "[4] 2026-11-22T10:04:00Z tool_call  "
                         "fetch(url=u)\n"
                         "[6] 2026-11-22T10:06:00Z summary\n"
                         "    events: 6  tokens: 0  "
                         "2026-11-22T10:00:00Z -> "
                         "2026-11-22T10:06:00Z\n")
        self.assertEqual([r["seq"] for r in records], [2, 3, 4, 6])

    def test_last_keeps_only_recent_steps(self):
        import io
        from contextlib import redirect_stdout
        from datetime import datetime, timezone
        buf = io.StringIO()
        with tempfile.TemporaryDirectory() as d:
            path = _write_log(d)
            now = datetime(2026, 11, 22, 10, 6, tzinfo=timezone.utc)
            with redirect_stdout(buf):
                records = replay(path, last="3m", now=now)
        out = buf.getvalue()
        self.assertIn("[3] 2026-11-22T10:03:00Z model_call", out)
        self.assertIn("[4] 2026-11-22T10:04:00Z tool_call", out)
        self.assertIn("[5] 2026-11-22T10:05:00Z tool_result", out)
        self.assertNotIn("[0] ", out)
        self.assertNotIn("[1] ", out)
        self.assertNotIn("[2] ", out)
        self.assertEqual([r["seq"] for r in records], [3, 4, 5, 6])

    def test_step_with_unparseable_timestamp_never_matches(self):
        import io
        from contextlib import redirect_stdout
        records = _demo_records()
        records.insert(3, _record(99, "tool_call", {"tool": "search"},
                                  "not-a-time"))
        buf = io.StringIO()
        with tempfile.TemporaryDirectory() as d:
            path = _write_log(d, records)
            with redirect_stdout(buf):
                replay(path, since="2026-11-22T10:00:00Z")
        self.assertNotIn("[99]", buf.getvalue())
        self.assertIn("[5] 2026-11-22T10:05:00Z tool_result", buf.getvalue())

    def test_bad_last_value_raises(self):
        with tempfile.TemporaryDirectory() as d:
            path = _write_log(d)
            with self.assertRaises(ValueError):
                replay(path, last="soon")

    def test_since_after_until_raises(self):
        with tempfile.TemporaryDirectory() as d:
            path = _write_log(d)
            with self.assertRaises(ValueError):
                replay(path, since="2026-11-22T10:05:00Z",
                       until="2026-11-22T10:01:00Z")

    def test_last_combined_with_since_raises(self):
        with tempfile.TemporaryDirectory() as d:
            path = _write_log(d)
            with self.assertRaises(ValueError):
                replay(path, last="10m",
                       since="2026-11-22T10:00:00Z")

    def test_window_works_on_sqlite_store(self):
        import io
        from contextlib import redirect_stdout
        buf = io.StringIO()
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "run.db")
            with Recorder(path) as rec:
                rec.record("model_call", {"model": "gpt-5",
                                          "prompt": "plan"},
                           ts="2026-11-22T10:00:00Z")
                rec.record("tool_call", {"tool": "search",
                                         "args": {"q": "x"}},
                           ts="2026-11-22T10:01:00Z")
                rec.record("tool_result", {"tool": "search",
                                           "result": "ok"},
                           ts="2026-11-22T10:02:00Z")
            with redirect_stdout(buf):
                records = replay(path, since="2026-11-22T10:01:00Z")
        out = buf.getvalue()
        self.assertNotIn("plan", out)
        self.assertIn("search", out)
        self.assertEqual([r["seq"] for r in records], [1, 2, 3])

    def test_cli_replay_window_and_bad_window_exit_code(self):
        import io
        from contextlib import redirect_stdout
        from blackbox.__main__ import main
        buf = io.StringIO()
        with tempfile.TemporaryDirectory() as d:
            path = _write_log(d)
            with redirect_stdout(buf):
                self.assertEqual(main(["replay", path,
                                       "--since", "2026-11-22T10:02:00Z",
                                       "--until", "2026-11-22T10:04:00Z"]),
                                 0)
            out = buf.getvalue()
            self.assertIn("[2] 2026-11-22T10:02:00Z tool_result", out)
            self.assertIn("[4] 2026-11-22T10:04:00Z tool_call", out)
            self.assertNotIn("[0] ", out)
            self.assertNotIn("[5] ", out)
            self.assertEqual(main(["replay", path, "--last", "soon"]), 2)
            self.assertEqual(main(["replay", path, "--last", "10m",
                                   "--since", "2026-11-22T10:00:00Z"]), 2)
            self.assertEqual(main(["replay", path,
                                   "--since", "2026-11-22T10:05:00Z",
                                   "--until", "2026-11-22T10:01:00Z"]), 2)


if __name__ == "__main__":
    unittest.main()
