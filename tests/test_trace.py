"""Trace: tool_results pair with their tool_calls (FIFO per tool),
tool_calls hang off the model that decided them, model_calls hang
off the previous step, orphans fall back to the previous step, and
the summary is never a node."""

import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout

from blackbox import Recorder
from blackbox.__main__ import main
from blackbox.trace import build_trace, link_parents, trace_log


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
        _record(3, "model_call", {"model": "gpt-5", "prompt": "summarise"},
                "2026-11-22T10:03:00Z"),
        _record(4, "tool_call", {"tool": "fetch", "args": {"url": "u"}},
                "2026-11-22T10:04:00Z"),
        _record(5, "tool_result", {"tool": "fetch", "result": "page"},
                "2026-11-22T10:05:00Z"),
        _record(6, "summary", {"events": 6, "total_tokens": 0},
                "2026-11-22T10:06:00Z"),
    ]


class LinkParentsTest(unittest.TestCase):
    def test_agent_loop_is_a_chain_of_parents(self):
        parents = link_parents(_demo_records())
        self.assertEqual(parents, {0: None, 1: 0, 2: 1, 3: 2, 4: 3, 5: 4})

    def test_parallel_calls_to_same_tool_pair_fifo(self):
        records = [
            _record(0, "model_call", {"model": "gpt-5"}),
            _record(1, "tool_call", {"tool": "search", "args": {"q": "a"}}),
            _record(2, "tool_call", {"tool": "search", "args": {"q": "b"}}),
            _record(3, "tool_result", {"tool": "search", "result": "ra"}),
            _record(4, "tool_result", {"tool": "search", "result": "rb"}),
        ]
        parents = link_parents(records)
        self.assertEqual(parents[3], 1)
        self.assertEqual(parents[4], 2)

    def test_tool_result_for_other_tool_pairs_with_its_own_call(self):
        records = [
            _record(0, "model_call", {"model": "gpt-5"}),
            _record(1, "tool_call", {"tool": "search"}),
            _record(2, "tool_call", {"tool": "fetch"}),
            _record(3, "tool_result", {"tool": "fetch", "result": "page"}),
            _record(4, "tool_result", {"tool": "search", "result": "ok"}),
        ]
        parents = link_parents(records)
        self.assertEqual(parents[3], 2)
        self.assertEqual(parents[4], 1)

    def test_orphan_tool_result_falls_back_to_previous_step(self):
        records = [
            _record(0, "model_call", {"model": "gpt-5"}),
            _record(1, "tool_result", {"tool": "search", "result": "ok"}),
        ]
        self.assertEqual(link_parents(records), {0: None, 1: 0})

    def test_tool_call_before_any_model_call_falls_back(self):
        records = [_record(0, "tool_call", {"tool": "search"})]
        self.assertEqual(link_parents(records), {0: None})

    def test_second_model_call_after_first_hangs_off_previous(self):
        records = [
            _record(0, "model_call", {"model": "gpt-5"}),
            _record(1, "model_call", {"model": "gpt-5"}),
        ]
        self.assertEqual(link_parents(records), {0: None, 1: 0})

    def test_empty_and_summary_only_have_no_parents(self):
        self.assertEqual(link_parents([]), {})
        self.assertEqual(link_parents([_record(0, "summary", {})]), {})

    def test_unknown_type_falls_back_to_previous_step(self):
        records = [
            _record(0, "model_call", {"model": "gpt-5"}),
            _record(1, "custom_step", {"note": "hi"}),
        ]
        self.assertEqual(link_parents(records), {0: None, 1: 0})


class BuildTraceTest(unittest.TestCase):
    def test_demo_run_is_one_tree(self):
        roots = build_trace(_demo_records())
        self.assertEqual(len(roots), 1)
        self.assertEqual(roots[0]["seq"], 0)
        self.assertEqual(roots[0]["label"], "model_call gpt-5")
        # model -> tool_call -> tool_result -> model -> tool_call -> result
        chain = [roots[0]]
        while chain[-1]["children"]:
            chain.append(chain[-1]["children"][0])
        self.assertEqual([n["seq"] for n in chain], [0, 1, 2, 3, 4, 5])

    def test_model_call_after_tool_call_hangs_off_it(self):
        records = [
            _record(0, "tool_call", {"tool": "search"}),
            _record(1, "model_call", {"model": "gpt-5"}),
        ]
        roots = build_trace(records)
        # seq 1 is a child of seq 0 (previous step), so one root.
        self.assertEqual(len(roots), 1)
        self.assertEqual(roots[0]["seq"], 0)

    def test_children_are_in_log_order(self):
        records = [
            _record(0, "model_call", {"model": "gpt-5"}),
            _record(1, "tool_call", {"tool": "search"}),
            _record(2, "tool_call", {"tool": "fetch"}),
        ]
        roots = build_trace(records)
        self.assertEqual([c["seq"] for c in roots[0]["children"]], [1, 2])


class TraceLogTest(unittest.TestCase):
    def test_printed_tree_is_indented_by_depth(self):
        buf = io.StringIO()
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "run.jsonl")
            with Recorder(path, task="t") as rec:
                rec.record("model_call", {"model": "gpt-5", "prompt": "p"},
                           tokens=10)
                rec.record("tool_call", {"tool": "search", "args": {}})
                rec.record("tool_result", {"tool": "search", "result": "ok"})
            with redirect_stdout(buf):
                trace_log(path)
        self.assertEqual(buf.getvalue(),
                         "[0] model_call gpt-5\n"
                         "  [1] tool_call search\n"
                         "    [2] tool_result search\n")

    def test_json_output_is_the_nested_tree(self):
        buf = io.StringIO()
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "run.db")
            with Recorder(path, task="t") as rec:
                rec.record("model_call", {"model": "gpt-5", "prompt": "p"},
                           tokens=10)
                rec.record("tool_call", {"tool": "search", "args": {}})
            with redirect_stdout(buf):
                trace_log(path, as_json=True)
        tree = json.loads(buf.getvalue())
        self.assertEqual(tree[0]["seq"], 0)
        self.assertEqual(tree[0]["children"][0]["seq"], 1)

    def test_cli_trace_command(self):
        buf = io.StringIO()
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "run.jsonl")
            with Recorder(path, task="t") as rec:
                rec.record("model_call", {"model": "gpt-5", "prompt": "p"},
                           tokens=10)
            with redirect_stdout(buf):
                self.assertEqual(main(["trace", path]), 0)
        self.assertEqual(buf.getvalue(), "[0] model_call gpt-5\n")


if __name__ == "__main__":
    unittest.main()
