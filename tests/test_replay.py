"""Replay: every step appears in order. Diff: changed tool calls are
flagged, token totals are reported."""

import os
import tempfile
import unittest

from blackbox import Recorder
from blackbox.replay import diff, diff_records, load_records, replay


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


if __name__ == "__main__":
    unittest.main()
