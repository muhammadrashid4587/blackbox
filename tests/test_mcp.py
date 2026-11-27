"""MCP server: handshake, tools/list, and the three tools over
newline-delimited JSON-RPC on stdio."""

import io
import json
import os
import tempfile
import unittest
from datetime import datetime, timezone

from blackbox import Recorder
from blackbox.mcp_server import MCPServer
from blackbox.otel import records_to_spans
from blackbox.replay import load_records
from blackbox.verify import verify


def _request(msg_id, method, params=None):
    msg = {"jsonrpc": "2.0", "id": msg_id, "method": method}
    if params is not None:
        msg["params"] = params
    return msg


def _call(server, msg_id, tool, arguments):
    return server.handle_message(_request(
        msg_id, "tools/call",
        {"name": tool, "arguments": arguments}))


def _nanos(ts):
    """Unix nanoseconds for an ISO timestamp, exact like otel._nanos."""
    epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
    delta = (datetime.fromisoformat(ts.replace("Z", "+00:00")) - epoch)
    return str((delta.days * 86400 + delta.seconds) * 1_000_000_000
               + delta.microseconds * 1000)


class MCPServerTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "run.jsonl")
        self.server = MCPServer(self.path, task="mcp test")

    def tearDown(self):
        try:
            self.server.close()
        except Exception:
            pass
        self.tmp.cleanup()

    def _init(self):
        response = self.server.handle_message(
            _request(1, "initialize", {"protocolVersion": "2024-11-05"}))
        self.assertEqual(response["result"]["serverInfo"]["name"],
                         "blackbox")
        self.assertIn("tools", response["result"]["capabilities"])
        return response

    def test_initialize_returns_server_info(self):
        self._init()

    def test_tools_call_before_initialize_is_rejected(self):
        response = _call(self.server, 1, "blackbox_verify", {})
        self.assertIn("error", response)
        self.assertEqual(response["error"]["code"], -32600)

    def test_tools_list(self):
        self._init()
        response = self.server.handle_message(
            _request(2, "tools/list"))
        names = [t["name"] for t in response["result"]["tools"]]
        self.assertEqual(names, ["blackbox_record", "blackbox_verify",
                                 "blackbox_replay", "blackbox_query",
                                 "blackbox_sample"])
        for tool in response["result"]["tools"]:
            self.assertIn("inputSchema", tool)

    def test_record_tool_appends_a_step(self):
        self._init()
        response = _call(self.server, 2, "blackbox_record", {
            "type": "tool_call",
            "payload": {"tool": "search", "args": {"query": "mcp"}},
        })
        self.assertFalse(response["result"]["isError"])
        self.assertIn("recorded step 0", response["result"]["content"][0]["text"])
        self.server.close()
        records = load_records(self.path)
        self.assertEqual(len(records), 2)  # step + summary
        self.assertEqual(records[0]["payload"]["tool"], "search")

    def test_record_tool_rejects_missing_type(self):
        self._init()
        response = _call(self.server, 2, "blackbox_record", {"payload": {}})
        self.assertEqual(response["error"]["code"], -32602)

    def test_verify_tool_reports_intact_chain(self):
        self._init()
        _call(self.server, 2, "blackbox_record",
              {"type": "note", "payload": {"text": "hello"}})
        response = _call(self.server, 3, "blackbox_verify", {})
        self.assertFalse(response["result"]["isError"])
        self.assertIn("hash chain intact",
                      response["result"]["content"][0]["text"])

    def test_replay_tool_returns_transcript(self):
        self._init()
        _call(self.server, 2, "blackbox_record",
              {"type": "tool_call",
               "payload": {"tool": "search", "args": {"query": "mcp"}}})
        response = _call(self.server, 3, "blackbox_replay", {})
        text = response["result"]["content"][0]["text"]
        self.assertIn("tool_call", text)
        self.assertIn("search", text)

    def _query(self, msg_id, arguments):
        response = _call(self.server, msg_id, "blackbox_query", arguments)
        self.assertNotIn("error", response)
        self.assertFalse(response["result"]["isError"])
        return json.loads(response["result"]["content"][0]["text"])

    def _seed_query_log(self):
        """Three steps with timestamps an hour ago: a priced gpt-5 call,
        an unpriced call for another model, and a failing tool result."""
        from datetime import timedelta
        base = datetime.now(timezone.utc) - timedelta(hours=1)
        self._t0 = base.isoformat().replace("+00:00", "Z")
        self._t1 = (base + timedelta(minutes=1)).isoformat().replace(
            "+00:00", "Z")
        self._t2 = (base + timedelta(minutes=2)).isoformat().replace(
            "+00:00", "Z")
        rec = self.server.recorder
        rec.record("model_call", {"model": "gpt-5", "prompt": "hi"},
                   tokens={"input": 1200, "output": 340}, ts=self._t0)
        rec.record("model_call", {"model": "other-model", "prompt": "hi"},
                   tokens=50, ts=self._t1)
        rec.record("tool_result",
                   {"tool": "search", "result": "error: timeout"},
                   ts=self._t2)

    def test_query_tool_filters_by_model(self):
        self._init()
        self._seed_query_log()
        result = self._query(2, {"model": "GPT-5"})
        self.assertEqual(result["matched"], 1)
        self.assertEqual(result["total"], 3)
        self.assertEqual(result["tokens"], 1540)
        self.assertAlmostEqual(result["cost_usd"], 0.0049, places=6)
        self.assertEqual(result["unpriced"], 0)
        self.assertEqual(result["records"][0]["payload"]["model"], "gpt-5")

    def test_query_tool_filters_by_tool_and_errors_only(self):
        self._init()
        self._seed_query_log()
        result = self._query(2, {"tool": "search", "errorsOnly": True})
        self.assertEqual(result["matched"], 1)
        self.assertEqual(result["records"][0]["type"], "tool_result")
        result = self._query(3, {"errorsOnly": True})
        self.assertEqual(result["matched"], 1)

    def test_query_tool_min_cost_skips_unpriced_calls(self):
        self._init()
        self._seed_query_log()
        result = self._query(2, {"minCost": 0.001})
        self.assertEqual(result["matched"], 1)
        self.assertEqual(result["records"][0]["payload"]["model"], "gpt-5")
        result = self._query(3, {"minCost": 1.0})
        self.assertEqual(result["matched"], 0)
        self.assertEqual(result["records"], [])
        self.assertIsNone(result["cost_usd"])

    def test_query_tool_time_window(self):
        self._init()
        self._seed_query_log()
        result = self._query(2, {"since": self._t1, "until": self._t2})
        self.assertEqual(result["matched"], 2)
        result = self._query(3, {"last": "1d"})
        self.assertEqual(result["matched"], 3)

    def test_query_tool_excludes_summary_unless_asked(self):
        self._init()
        self._seed_query_log()
        self.server.close()
        result = self._query(2, {})
        self.assertEqual(result["matched"], 3)
        self.assertEqual(result["total"], 3)
        result = self._query(3, {"type": ["summary"]})
        self.assertEqual(result["matched"], 1)
        self.assertEqual(result["records"][0]["type"], "summary")

    def test_query_tool_rejects_bad_arguments(self):
        self._init()
        self._seed_query_log()
        for arguments in (
                {"since": "2026-11-24T01:00:00"},  # no timezone
                {"since": "2026-11-24T02:00:00Z",
                 "until": "2026-11-24T01:00:00Z"},  # reversed
                {"last": "10x"},  # not a duration
                {"last": "10m", "since": "2026-11-24T01:00:00Z"},
                {"type": "model_call"},  # must be a list
                {"minTokens": "100"},  # must be an int
                {"minCost": "0.01"},  # must be a number
                {"errorsOnly": "yes"},  # must be a boolean
        ):
            response = _call(self.server, 2, "blackbox_query", arguments)
            self.assertEqual(response["error"]["code"], -32602,
                             arguments)

    def test_unknown_tool_is_an_error(self):
        self._init()
        response = _call(self.server, 2, "blackbox_nope", {})
        self.assertEqual(response["error"]["code"], -32602)

    def test_initialize_advertises_resources_and_prompts(self):
        response = self._init()
        capabilities = response["result"]["capabilities"]
        self.assertIn("resources", capabilities)
        self.assertIn("prompts", capabilities)
        self.assertEqual(response["result"]["serverInfo"]["version"],
                         "0.8.0")

    def test_resources_list(self):
        self._init()
        response = self.server.handle_message(
            _request(2, "resources/list"))
        uris = [r["uri"] for r in response["result"]["resources"]]
        self.assertEqual(uris, ["blackbox://session",
                                "blackbox://session.json",
                                "blackbox://session/aggregates",
                                "blackbox://session/trace",
                                "blackbox://session/diff"])
        for resource in response["result"]["resources"]:
            self.assertIn("mimeType", resource)

    def test_resources_templates_list(self):
        self._init()
        response = self.server.handle_message(
            _request(2, "resources/templates/list"))
        templates = response["result"]["resourceTemplates"]
        self.assertEqual(len(templates), 7)
        self.assertEqual(
            [t["uriTemplate"] for t in templates],
            ["blackbox://session/diff/{other_path}",
             "blackbox://session/trace/{seq}",
             "blackbox://session/aggregates/model/{model}",
             "blackbox://session/aggregates/tool/{tool}",
             "blackbox://session/record/{seq}",
             "blackbox://session/transcript/{seq}",
             "blackbox://session/span/{seq}"])
        self.assertEqual(
            [t["name"] for t in templates],
            ["session/diff/{other_path}", "session/trace/{seq}",
             "session/aggregates/model/{model}",
             "session/aggregates/tool/{tool}",
             "session/record/{seq}",
             "session/transcript/{seq}",
             "session/span/{seq}"])
        self.assertEqual(
            [t["mimeType"] for t in templates],
            ["application/json"] * 5 + ["text/plain",
                                        "application/json"])
        self.assertIn("other run", templates[0]["description"])
        self.assertIn("step", templates[1]["description"])
        self.assertIn("model", templates[2]["description"])
        self.assertIn("tool", templates[3]["description"])
        self.assertIn("record", templates[4]["description"])
        self.assertIn("transcript", templates[5]["description"])
        self.assertIn("span", templates[6]["description"])

    def test_resources_read_session_transcript(self):
        self._init()
        _call(self.server, 2, "blackbox_record",
              {"type": "tool_call",
               "payload": {"tool": "search", "args": {"query": "mcp"}}})
        response = self.server.handle_message(
            _request(3, "resources/read",
                     {"uri": "blackbox://session"}))
        content = response["result"]["contents"][0]
        self.assertEqual(content["uri"], "blackbox://session")
        self.assertEqual(content["mimeType"], "text/plain")
        self.assertIn("tool_call", content["text"])
        self.assertIn("search", content["text"])

    def test_resources_read_session_transcript_window_last(self):
        # The same --last window as `blackbox replay --last`: the
        # session was recorded just now, so last=1h keeps every
        # step and the transcript is the unwindowed one. The
        # server is closed first, so the summary record exists
        # and the transcript ends with the whole log's totals.
        self._init()
        self._record_session()
        self.server.close()
        response = self.server.handle_message(
            _request(6, "resources/read",
                     {"uri": "blackbox://session?last=1h"}))
        text = response["result"]["contents"][0]["text"]
        self.assertIn("tool_call", text)
        self.assertIn("search", text)
        self.assertIn("model_call", text)
        self.assertIn("events: 4", text)

    def test_resources_read_session_transcript_window_since_excludes_all(self):
        # A --since window after every step drops the session to
        # zero steps, so the transcript holds only the summary line
        # — the steps outside the window cannot change it — while
        # the summary record is kept untouched, so the transcript
        # still ends with the whole log's totals.
        self._init()
        self._record_session()
        self.server.close()
        response = self.server.handle_message(
            _request(6, "resources/read",
                     {"uri": "blackbox://session"
                            "?since=2999-01-01T00:00:00Z"}))
        text = response["result"]["contents"][0]["text"]
        self.assertNotIn("tool_call", text)
        self.assertNotIn("model_call", text)
        self.assertIn("events: 4", text)

    def test_resources_read_session_transcript_window_bad_bounds_are_errors(self):
        # A bad bound, a reversed window, --last combined with
        # --since, and a bad --last are INVALID_PARAMS errors —
        # the resource-read analogue of the CLI replay's exit
        # code 2.
        self._init()
        self._record_session()
        base = "blackbox://session"
        for suffix in ("?since=not-a-time",
                       "?since=2026-11-24T02:00:00Z"
                       "&until=2026-11-24T01:00:00Z",
                       "?last=1h&since=2026-11-24T01:00:00Z",
                       "?last=soon"):
            response = self.server.handle_message(
                _request(2, "resources/read",
                         {"uri": base + suffix}))
            self.assertEqual(response["error"]["code"], -32602,
                             suffix)

    def test_resources_read_session_json(self):
        self._init()
        _call(self.server, 2, "blackbox_record",
              {"type": "note", "payload": {"text": "hello"}})
        response = self.server.handle_message(
            _request(3, "resources/read",
                     {"uri": "blackbox://session.json"}))
        content = response["result"]["contents"][0]
        self.assertEqual(content["mimeType"], "application/json")
        records = json.loads(content["text"])
        self.assertEqual(records[0]["type"], "note")
        self.assertIn("hash", records[0])
        self.assertIn("prev_hash", records[0])

    def test_resources_read_session_json_window_last(self):
        # The same --last window as `blackbox replay --last`: the
        # session was recorded just now, so last=1h keeps every
        # step and the records are the unwindowed ones. The server
        # is closed first, so the summary record exists.
        self._init()
        self._record_session()
        self.server.close()
        response = self.server.handle_message(
            _request(6, "resources/read",
                     {"uri": "blackbox://session.json?last=1h"}))
        records = json.loads(
            response["result"]["contents"][0]["text"])
        self.assertEqual([r["type"] for r in records],
                         ["model_call", "tool_call", "tool_result",
                          "model_call", "summary"])

    def test_resources_read_session_json_window_since_excludes_all(self):
        # A --since window after every step drops the session to
        # zero steps, so the records hold only the summary — the
        # steps outside the window cannot change them — while the
        # summary record is kept untouched, so the bookkeeping
        # still describes the whole log.
        self._init()
        self._record_session()
        self.server.close()
        response = self.server.handle_message(
            _request(6, "resources/read",
                     {"uri": "blackbox://session.json"
                            "?since=2999-01-01T00:00:00Z"}))
        records = json.loads(
            response["result"]["contents"][0]["text"])
        self.assertEqual([r["type"] for r in records], ["summary"])
        self.assertEqual(records[0]["payload"]["events"], 4)

    def test_resources_read_session_json_window_bad_bounds_are_errors(self):
        # A bad bound, a reversed window, --last combined with
        # --since, and a bad --last are INVALID_PARAMS errors —
        # the resource-read analogue of the CLI replay's exit
        # code 2.
        self._init()
        self._record_session()
        base = "blackbox://session.json"
        for suffix in ("?since=not-a-time",
                       "?since=2026-11-24T02:00:00Z"
                       "&until=2026-11-24T01:00:00Z",
                       "?last=1h&since=2026-11-24T01:00:00Z",
                       "?last=soon"):
            response = self.server.handle_message(
                _request(2, "resources/read",
                         {"uri": base + suffix}))
            self.assertEqual(response["error"]["code"], -32602,
                             suffix)

    def test_resources_read_session_aggregates(self):
        self._init()
        self._record_session()
        response = self.server.handle_message(
            _request(6, "resources/read",
                     {"uri": "blackbox://session/aggregates"}))
        content = response["result"]["contents"][0]
        self.assertEqual(content["uri"],
                         "blackbox://session/aggregates")
        self.assertEqual(content["mimeType"], "application/json")
        agg = json.loads(content["text"])
        # Same totals `blackbox aggregates` prints for this session:
        # 4 steps, 1690 tokens, gpt-5 1300 in + 390 out = $0.005525.
        self.assertEqual(agg["steps"], 4)
        self.assertEqual(agg["total_tokens"], 1690)
        self.assertEqual(agg["total_cost_usd"], 0.005525)
        self.assertEqual(agg["by_type"],
                         {"model_call": 2, "tool_call": 1,
                          "tool_result": 1})
        self.assertEqual(agg["by_tool"], {"search": 1})
        self.assertEqual(agg["by_model"]["gpt-5"]["calls"], 2)
        self.assertEqual(agg["by_model"]["gpt-5"]["input_tokens"],
                         1300)
        self.assertEqual(agg["by_model"]["gpt-5"]["output_tokens"],
                         390)
        self.assertEqual(agg["by_model"]["gpt-5"]["cost_usd"],
                         0.005525)
        self.assertEqual(agg["errors"], 0)
        self.assertEqual(agg["error_seqs"], [])
        self.assertIsNotNone(agg["slowest_step"])
        self.assertEqual(agg["compacted_count"], 0)
        self.assertEqual(agg["sampled_out"], 0)

    def test_resources_read_session_aggregates_with_error(self):
        self._init()
        _call(self.server, 2, "blackbox_record",
              {"type": "tool_call",
               "payload": {"tool": "search",
                           "args": {"query": "mcp"}}})
        _call(self.server, 3, "blackbox_record",
              {"type": "tool_result",
               "payload": {"tool": "search",
                           "result": "error: timeout"}})
        response = self.server.handle_message(
            _request(4, "resources/read",
                     {"uri": "blackbox://session/aggregates"}))
        agg = json.loads(
            response["result"]["contents"][0]["text"])
        self.assertEqual(agg["steps"], 2)
        self.assertEqual(agg["errors"], 1)
        self.assertEqual(agg["error_seqs"], [1])

    def test_resources_read_session_aggregates_compacted(self):
        self._init()
        self._record_session()
        self.server.close()
        from blackbox.compact import compact_log
        result = compact_log(self.path, keep_n=2)
        self.assertTrue(result["changed"])
        self.assertEqual(result["compacted"], 2)
        self.server = MCPServer(self.path, task="mcp test")
        self._init()
        response = self.server.handle_message(
            _request(2, "resources/read",
                     {"uri": "blackbox://session/aggregates"}))
        agg = json.loads(
            response["result"]["contents"][0]["text"])
        self.assertEqual(agg["steps"], 2)
        self.assertEqual(agg["compacted_count"], 2)

    def test_resources_read_session_aggregates_empty_session(self):
        self._init()
        response = self.server.handle_message(
            _request(2, "resources/read",
                     {"uri": "blackbox://session/aggregates"}))
        agg = json.loads(
            response["result"]["contents"][0]["text"])
        self.assertEqual(agg["steps"], 0)
        self.assertEqual(agg["total_tokens"], 0)
        self.assertEqual(agg["by_type"], {})
        self.assertIsNone(agg["slowest_step"])

    def test_resources_read_session_aggregates_window_last(self):
        # The same --last window as `blackbox aggregates --last`:
        # the session was recorded just now, so last=1h keeps every
        # step and the totals are the unwindowed ones.
        self._init()
        self._record_session()
        response = self.server.handle_message(
            _request(6, "resources/read",
                     {"uri": "blackbox://session/aggregates?last=1h"}))
        agg = json.loads(
            response["result"]["contents"][0]["text"])
        self.assertEqual(agg["steps"], 4)
        self.assertEqual(agg["total_tokens"], 1690)
        self.assertEqual(agg["total_cost_usd"], 0.005525)
        self.assertEqual(agg["by_type"],
                         {"model_call": 2, "tool_call": 1,
                          "tool_result": 1})
        self.assertEqual(agg["by_tool"], {"search": 1})
        self.assertEqual(agg["by_model"]["gpt-5"]["calls"], 2)
        self.assertEqual(agg["errors"], 0)
        self.assertIsNotNone(agg["slowest_step"])
        self.assertEqual(agg["compacted_count"], 0)
        self.assertEqual(agg["sampled_out"], 0)

    def test_resources_read_session_aggregates_window_since_excludes_all(self):
        # A --since window after every step drops the session to
        # zero steps, so the totals are the empty ones — the steps
        # outside the window cannot change them — while the
        # summary record is kept untouched.
        self._init()
        self._record_session()
        response = self.server.handle_message(
            _request(6, "resources/read",
                     {"uri": "blackbox://session/aggregates"
                            "?since=2999-01-01T00:00:00Z"}))
        agg = json.loads(
            response["result"]["contents"][0]["text"])
        self.assertEqual(agg["steps"], 0)
        self.assertIsNone(agg["started"])
        self.assertIsNone(agg["ended"])
        self.assertIsNone(agg["duration_seconds"])
        self.assertEqual(agg["total_tokens"], 0)
        self.assertEqual(agg["total_cost_usd"], 0.0)
        self.assertEqual(agg["by_type"], {})
        self.assertEqual(agg["by_tool"], {})
        self.assertEqual(agg["by_model"], {})
        self.assertEqual(agg["errors"], 0)
        self.assertIsNone(agg["slowest_step"])
        self.assertEqual(agg["compacted_count"], 0)
        self.assertEqual(agg["sampled_out"], 0)

    def test_resources_read_session_aggregates_window_bad_bounds_are_errors(self):
        # A bad bound, a reversed window, --last combined with
        # --since, and a bad --last are INVALID_PARAMS errors —
        # the resource-read analogue of the CLI aggregates' exit
        # code 2.
        self._init()
        self._record_session()
        base = "blackbox://session/aggregates"
        for suffix in ("?since=not-a-time",
                       "?since=2026-11-24T02:00:00Z"
                       "&until=2026-11-24T01:00:00Z",
                       "?last=1h&since=2026-11-24T01:00:00Z",
                       "?last=soon"):
            response = self.server.handle_message(
                _request(2, "resources/read",
                         {"uri": base + suffix}))
            self.assertEqual(response["error"]["code"], -32602,
                             suffix)

    def test_resources_read_session_trace(self):
        self._init()
        self._record_session()
        response = self.server.handle_message(
            _request(6, "resources/read",
                     {"uri": "blackbox://session/trace"}))
        content = response["result"]["contents"][0]
        self.assertEqual(content["uri"],
                         "blackbox://session/trace")
        self.assertEqual(content["mimeType"], "application/json")
        trace = json.loads(content["text"])
        # The same trace `blackbox trace` prints for this session:
        # 4 steps in one chain, model_call -> tool_call ->
        # tool_result -> model_call.
        self.assertEqual(trace["steps"], 4)
        self.assertEqual(trace["roots"], 1)
        self.assertEqual(trace["max_depth"], 4)
        self.assertEqual(trace["by_type"],
                         {"model_call": 2, "tool_call": 1,
                          "tool_result": 1})
        self.assertEqual(trace["errors"], 0)
        self.assertEqual(trace["error_seqs"], [])
        self.assertEqual(trace["error_steps"], [])
        self.assertEqual(len(trace["slowest_steps"]), 3)
        for step in trace["slowest_steps"]:
            self.assertIn(step["seq"], (0, 1, 2))
            self.assertGreaterEqual(step["latency_seconds"], 0)
        self.assertIn(trace["tail_sample_decision"],
                      ("kept_all", "sampled"))
        self.assertIsInstance(trace["parent_gaps"], list)
        tree = trace["tree"]
        self.assertEqual(len(tree), 1)
        self.assertEqual(tree[0]["seq"], 0)
        self.assertEqual(tree[0]["label"], "model_call gpt-5")
        self.assertEqual(tree[0]["children"][0]["seq"], 1)
        self.assertEqual(tree[0]["children"][0]["label"],
                         "tool_call search")
        self.assertEqual(
            tree[0]["children"][0]["children"][0]["seq"], 2)
        self.assertEqual(
            tree[0]["children"][0]["children"][0]["label"],
            "tool_result search")
        self.assertEqual(
            tree[0]["children"][0]["children"][0]["children"][0]
            ["seq"], 3)
        self.assertEqual(
            tree[0]["children"][0]["children"][0]["children"][0]
            ["label"], "model_call gpt-5")

    def test_resources_read_session_trace_with_error(self):
        self._init()
        _call(self.server, 2, "blackbox_record",
              {"type": "tool_call",
               "payload": {"tool": "search",
                           "args": {"query": "mcp"}}})
        _call(self.server, 3, "blackbox_record",
              {"type": "tool_result",
               "payload": {"tool": "search", "error": "timeout"}})
        response = self.server.handle_message(
            _request(4, "resources/read",
                     {"uri": "blackbox://session/trace"}))
        trace = json.loads(
            response["result"]["contents"][0]["text"])
        self.assertEqual(trace["steps"], 2)
        self.assertEqual(trace["errors"], 1)
        self.assertEqual(trace["error_seqs"], [1])
        self.assertEqual(trace["error_steps"],
                         [{"seq": 1, "type": "tool_result",
                           "label": "tool_result search",
                           "error": "timeout"}])
        self.assertEqual(trace["tree"][0]["seq"], 0)
        self.assertEqual(trace["tree"][0]["children"][0]["seq"], 1)

    def test_resources_read_session_trace_empty_session(self):
        self._init()
        response = self.server.handle_message(
            _request(2, "resources/read",
                     {"uri": "blackbox://session/trace"}))
        trace = json.loads(
            response["result"]["contents"][0]["text"])
        self.assertEqual(trace["steps"], 0)
        self.assertEqual(trace["roots"], 0)
        self.assertEqual(trace["max_depth"], 0)
        self.assertEqual(trace["by_type"], {})
        self.assertEqual(trace["slowest_steps"], [])
        self.assertEqual(trace["errors"], 0)
        self.assertEqual(trace["error_seqs"], [])
        self.assertEqual(trace["parent_gaps"], [])
        self.assertEqual(trace["tree"], [])

    def test_resources_read_session_trace_window_last(self):
        # The same --last window as `blackbox trace --last`: the
        # session was recorded just now, so last=1h keeps every
        # step and the trace is the unwindowed one.
        self._init()
        self._record_session()
        response = self.server.handle_message(
            _request(6, "resources/read",
                     {"uri": "blackbox://session/trace?last=1h"}))
        trace = json.loads(
            response["result"]["contents"][0]["text"])
        self.assertEqual(trace["steps"], 4)
        self.assertEqual(trace["roots"], 1)
        self.assertEqual(trace["max_depth"], 4)
        self.assertEqual(trace["by_type"],
                         {"model_call": 2, "tool_call": 1,
                          "tool_result": 1})
        self.assertEqual(trace["errors"], 0)
        self.assertEqual(trace["tree"][0]["seq"], 0)
        self.assertEqual(
            trace["tree"][0]["children"][0]["children"][0]
            ["children"][0]["seq"], 3)

    def test_resources_read_session_trace_window_since_excludes_all(self):
        # A --since window after every step drops the session to
        # zero steps, so the trace is the empty one — the steps
        # outside the window cannot change the tree.
        self._init()
        self._record_session()
        response = self.server.handle_message(
            _request(6, "resources/read",
                     {"uri": "blackbox://session/trace"
                            "?since=2999-01-01T00:00:00Z"}))
        trace = json.loads(
            response["result"]["contents"][0]["text"])
        self.assertEqual(trace["steps"], 0)
        self.assertEqual(trace["roots"], 0)
        self.assertEqual(trace["max_depth"], 0)
        self.assertEqual(trace["by_type"], {})
        self.assertEqual(trace["slowest_steps"], [])
        self.assertEqual(trace["errors"], 0)
        self.assertEqual(trace["error_seqs"], [])
        self.assertEqual(trace["parent_gaps"], [])
        self.assertEqual(trace["tree"], [])

    def test_resources_read_session_trace_window_bad_bounds_are_errors(self):
        # A bad bound, a reversed window, --last combined with
        # --since, and a bad --last are INVALID_PARAMS errors —
        # the resource-read analogue of the CLI trace's exit
        # code 2.
        self._init()
        self._record_session()
        base = "blackbox://session/trace"
        for suffix in ("?since=not-a-time",
                       "?since=2026-11-24T02:00:00Z"
                       "&until=2026-11-24T01:00:00Z",
                       "?last=1h&since=2026-11-24T01:00:00Z",
                       "?last=soon"):
            response = self.server.handle_message(
                _request(2, "resources/read",
                         {"uri": base + suffix}))
            self.assertEqual(response["error"]["code"], -32602,
                             suffix)

    def test_resources_read_session_diff(self):
        self._init()
        self._record_session()
        other = os.path.join(self.tmp.name, "other.jsonl")
        self._write_other_run(other, query="other",
                              extra=True, error_result=True)
        response = self.server.handle_message(
            _request(6, "resources/read",
                     {"uri": "blackbox://session/diff?other_path="
                            + other}))
        content = response["result"]["contents"][0]
        self.assertEqual(content["uri"],
                         "blackbox://session/diff?other_path=" + other)
        self.assertEqual(content["mimeType"], "application/json")
        diff = json.loads(content["text"])
        # Same diff `blackbox diff` prints for these runs: steps
        # 1-2 changed (query, then an error result), one extra
        # fetch call, same 1690 tokens on both sides.
        self.assertEqual(diff["run_a"], {"steps": 4, "tokens": 1690})
        self.assertEqual(diff["run_b"], {"steps": 5, "tokens": 1690})
        self.assertEqual(diff["tokens_diff"], 0)
        self.assertFalse(diff["identical"])
        self.assertEqual([c["position"] for c in diff["changed"]],
                         [1, 2])
        self.assertEqual(len(diff["added"]), 1)
        self.assertEqual(diff["added"][0]["payload"]["tool"], "fetch")
        self.assertEqual(diff["removed"], [])
        gpt5 = diff["per_model"]["gpt-5"]
        self.assertEqual(gpt5["tokens_a"], 1690)
        self.assertEqual(gpt5["tokens_b"], 1690)
        self.assertEqual(gpt5["tokens_delta"], 0)
        self.assertEqual(gpt5["cost_usd_a"], 0.005525)
        self.assertEqual(gpt5["cost_usd_b"], 0.005525)
        self.assertEqual(gpt5["cost_usd_delta"], 0)
        self.assertEqual(len(diff["new_errors"]), 1)
        self.assertEqual(diff["new_errors"][0]["payload"],
                         {"tool": "search", "error": "timed out"})
        self.assertEqual(diff["fixed_errors"], [])
        self.assertEqual(diff["errors_a"], 0)
        self.assertEqual(diff["errors_b"], 1)

    def test_resources_read_session_diff_identical(self):
        self._init()
        self._record_session()
        other = os.path.join(self.tmp.name, "other.jsonl")
        self._write_other_run(other)
        response = self.server.handle_message(
            _request(6, "resources/read",
                     {"uri": "blackbox://session/diff?other_path="
                            + other}))
        diff = json.loads(
            response["result"]["contents"][0]["text"])
        self.assertTrue(diff["identical"])
        self.assertEqual(diff["changed"], [])
        self.assertEqual(diff["added"], [])
        self.assertEqual(diff["removed"], [])
        self.assertEqual(diff["tokens_diff"], 0)

    def test_resources_read_session_diff_empty_session(self):
        self._init()
        other = os.path.join(self.tmp.name, "other.jsonl")
        with Recorder(other) as rec:
            pass
        response = self.server.handle_message(
            _request(2, "resources/read",
                     {"uri": "blackbox://session/diff?other_path="
                            + other}))
        diff = json.loads(
            response["result"]["contents"][0]["text"])
        self.assertTrue(diff["identical"])
        self.assertEqual(diff["run_a"], {"steps": 0, "tokens": 0})
        self.assertEqual(diff["run_b"], {"steps": 0, "tokens": 0})
        self.assertEqual(diff["per_model"], {})

    def test_resources_read_session_diff_needs_other_path(self):
        self._init()
        response = self.server.handle_message(
            _request(2, "resources/read",
                     {"uri": "blackbox://session/diff"}))
        self.assertEqual(response["error"]["code"], -32602)

    def test_resources_read_session_diff_missing_file_is_an_error(self):
        self._init()
        response = self.server.handle_message(
            _request(2, "resources/read",
                     {"uri": "blackbox://session/diff?other_path="
                            + os.path.join(self.tmp.name,
                                           "nope.jsonl")}))
        self.assertEqual(response["error"]["code"], -32602)

    def test_resources_read_session_diff_template(self):
        # The same diff as the query-parameter resource, read by
        # expanding the template: the other run's path is the URI
        # path after blackbox://session/diff/ (an absolute path
        # keeps its leading slash).
        self._init()
        self._record_session()
        other = os.path.join(self.tmp.name, "other.jsonl")
        self._write_other_run(other, query="other",
                              extra=True, error_result=True)
        response = self.server.handle_message(
            _request(6, "resources/read",
                     {"uri": "blackbox://session/diff/" + other}))
        content = response["result"]["contents"][0]
        self.assertEqual(content["uri"],
                         "blackbox://session/diff/" + other)
        self.assertEqual(content["mimeType"], "application/json")
        diff = json.loads(content["text"])
        self.assertEqual(diff["run_a"], {"steps": 4, "tokens": 1690})
        self.assertEqual(diff["run_b"], {"steps": 5, "tokens": 1690})
        self.assertEqual(diff["tokens_diff"], 0)
        self.assertFalse(diff["identical"])
        self.assertEqual([c["position"] for c in diff["changed"]],
                         [1, 2])
        self.assertEqual(len(diff["added"]), 1)
        self.assertEqual(diff["added"][0]["payload"]["tool"], "fetch")
        self.assertEqual(diff["removed"], [])
        gpt5 = diff["per_model"]["gpt-5"]
        self.assertEqual(gpt5["tokens_a"], 1690)
        self.assertEqual(gpt5["tokens_b"], 1690)
        self.assertEqual(gpt5["cost_usd_a"], 0.005525)
        self.assertEqual(gpt5["cost_usd_b"], 0.005525)
        self.assertEqual(gpt5["cost_usd_delta"], 0)
        self.assertEqual(len(diff["new_errors"]), 1)
        self.assertEqual(diff["new_errors"][0]["payload"],
                         {"tool": "search", "error": "timed out"})
        self.assertEqual(diff["fixed_errors"], [])
        self.assertEqual(diff["errors_a"], 0)
        self.assertEqual(diff["errors_b"], 1)

    def test_resources_read_session_diff_window_last(self):
        # The same --last window as `blackbox diff --last`: the
        # session and the other run were both recorded just now,
        # so last=1h keeps every step and the diff is the
        # unwindowed one.
        self._init()
        self._record_session()
        other = os.path.join(self.tmp.name, "other.jsonl")
        self._write_other_run(other, query="other",
                              extra=True, error_result=True)
        response = self.server.handle_message(
            _request(6, "resources/read",
                     {"uri": "blackbox://session/diff?other_path="
                            + other + "&last=1h"}))
        diff = json.loads(
            response["result"]["contents"][0]["text"])
        self.assertEqual(diff["run_a"], {"steps": 4, "tokens": 1690})
        self.assertEqual(diff["run_b"], {"steps": 5, "tokens": 1690})
        self.assertFalse(diff["identical"])
        self.assertEqual([c["position"] for c in diff["changed"]],
                         [1, 2])
        self.assertEqual(len(diff["added"]), 1)

    def test_resources_read_session_diff_window_since_excludes_all(self):
        # A --since window after every step drops both runs to
        # zero steps, so the runs compare identical over the
        # window — the changed steps outside it cannot make
        # them look different.
        self._init()
        self._record_session()
        other = os.path.join(self.tmp.name, "other.jsonl")
        self._write_other_run(other, query="other",
                              extra=True, error_result=True)
        response = self.server.handle_message(
            _request(6, "resources/read",
                     {"uri": "blackbox://session/diff?other_path="
                            + other
                            + "&since=2999-01-01T00:00:00Z"}))
        diff = json.loads(
            response["result"]["contents"][0]["text"])
        self.assertTrue(diff["identical"])
        self.assertEqual(diff["run_a"], {"steps": 0, "tokens": 0})
        self.assertEqual(diff["run_b"], {"steps": 0, "tokens": 0})
        self.assertEqual(diff["changed"], [])
        self.assertEqual(diff["added"], [])
        self.assertEqual(diff["removed"], [])
        self.assertEqual(diff["per_model"], {})

    def test_resources_read_session_diff_window_bad_bounds_are_errors(self):
        # A bad bound, a reversed window, --last combined with
        # --since, and a bad --last are INVALID_PARAMS errors —
        # the resource-read analogue of the CLI diff's exit
        # code 2.
        self._init()
        self._record_session()
        other = os.path.join(self.tmp.name, "other.jsonl")
        self._write_other_run(other)
        base = ("blackbox://session/diff?other_path=" + other)
        for suffix in ("&since=not-a-time",
                       "&since=2026-11-24T02:00:00Z"
                       "&until=2026-11-24T01:00:00Z",
                       "&last=1h&since=2026-11-24T01:00:00Z",
                       "&last=soon"):
            response = self.server.handle_message(
                _request(2, "resources/read",
                         {"uri": base + suffix}))
            self.assertEqual(response["error"]["code"], -32602,
                             suffix)

    def test_resources_read_session_diff_template_window(self):
        # The window rides in the query string after the
        # template path too: last=1h keeps every step, so the
        # template diff is the unwindowed one.
        self._init()
        self._record_session()
        other = os.path.join(self.tmp.name, "other.jsonl")
        self._write_other_run(other, query="other",
                              extra=True, error_result=True)
        response = self.server.handle_message(
            _request(6, "resources/read",
                     {"uri": "blackbox://session/diff/" + other
                            + "?last=1h"}))
        diff = json.loads(
            response["result"]["contents"][0]["text"])
        self.assertEqual(diff["run_a"], {"steps": 4, "tokens": 1690})
        self.assertEqual(diff["run_b"], {"steps": 5, "tokens": 1690})
        self.assertFalse(diff["identical"])
        self.assertEqual([c["position"] for c in diff["changed"]],
                         [1, 2])

    def test_resources_read_session_diff_template_encoded_path(self):
        # A path with a space, percent-encoded in the URI, is
        # decoded before the log is read.
        self._init()
        self._record_session()
        other = os.path.join(self.tmp.name, "other run.jsonl")
        self._write_other_run(other)
        response = self.server.handle_message(
            _request(6, "resources/read",
                     {"uri": "blackbox://session/diff/"
                            + other.replace(" ", "%20")}))
        diff = json.loads(
            response["result"]["contents"][0]["text"])
        self.assertTrue(diff["identical"])
        self.assertEqual(diff["run_a"], {"steps": 4, "tokens": 1690})
        self.assertEqual(diff["run_b"], {"steps": 4, "tokens": 1690})

    def test_resources_read_session_diff_template_needs_other_path(self):
        self._init()
        response = self.server.handle_message(
            _request(2, "resources/read",
                     {"uri": "blackbox://session/diff/"}))
        self.assertEqual(response["error"]["code"], -32602)

    def test_resources_read_session_diff_template_missing_file_is_an_error(self):
        self._init()
        response = self.server.handle_message(
            _request(2, "resources/read",
                     {"uri": "blackbox://session/diff/"
                            + os.path.join(self.tmp.name,
                                           "nope.jsonl")}))
        self.assertEqual(response["error"]["code"], -32602)

    def test_resources_read_session_trace_template(self):
        # The same node the blackbox://session/trace resource
        # carries for step 1: the tool_call and its result and the
        # model_call after it, but not the root model_call above.
        self._init()
        self._record_session()
        response = self.server.handle_message(
            _request(6, "resources/read",
                     {"uri": "blackbox://session/trace/1"}))
        content = response["result"]["contents"][0]
        self.assertEqual(content["uri"], "blackbox://session/trace/1")
        self.assertEqual(content["mimeType"], "application/json")
        node = json.loads(content["text"])
        self.assertEqual(node["seq"], 1)
        self.assertEqual(node["type"], "tool_call")
        self.assertEqual(len(node["children"]), 1)
        self.assertEqual(node["children"][0]["seq"], 2)
        self.assertEqual(node["children"][0]["children"][0]["seq"], 3)

    def test_resources_read_session_trace_template_root(self):
        # The root's subtree is the whole tree.
        self._init()
        self._record_session()
        response = self.server.handle_message(
            _request(6, "resources/read",
                     {"uri": "blackbox://session/trace/0"}))
        node = json.loads(
            response["result"]["contents"][0]["text"])
        chain = [node]
        while chain[-1]["children"]:
            chain.append(chain[-1]["children"][0])
        self.assertEqual([n["seq"] for n in chain], [0, 1, 2, 3])

    def test_resources_read_session_trace_template_unknown_seq(self):
        self._init()
        self._record_session()
        response = self.server.handle_message(
            _request(2, "resources/read",
                     {"uri": "blackbox://session/trace/99"}))
        self.assertEqual(response["error"]["code"], -32602)

    def test_resources_read_session_trace_template_bad_seq(self):
        self._init()
        response = self.server.handle_message(
            _request(2, "resources/read",
                     {"uri": "blackbox://session/trace/abc"}))
        self.assertEqual(response["error"]["code"], -32602)

    def test_resources_read_session_trace_template_needs_seq(self):
        self._init()
        response = self.server.handle_message(
            _request(2, "resources/read",
                     {"uri": "blackbox://session/trace/"}))
        self.assertEqual(response["error"]["code"], -32602)

    def test_resources_read_session_trace_template_window_last(self):
        # The same --last window as `blackbox trace --last`: the
        # session was recorded just now, so last=1h keeps every
        # step and the subtree is the unwindowed one.
        self._init()
        self._record_session()
        response = self.server.handle_message(
            _request(6, "resources/read",
                     {"uri": "blackbox://session/trace/1?last=1h"}))
        node = json.loads(
            response["result"]["contents"][0]["text"])
        self.assertEqual(node["seq"], 1)
        self.assertEqual(node["type"], "tool_call")
        self.assertEqual(len(node["children"]), 1)
        self.assertEqual(node["children"][0]["seq"], 2)
        self.assertEqual(node["children"][0]["children"][0]["seq"], 3)

    def test_resources_read_session_trace_template_window_since_excludes_step(self):
        # A --since window after every step drops the session to
        # zero steps, so the windowed-out step is not in the
        # session and the template read is an error — the window
        # is applied before the subtree is built.
        self._init()
        self._record_session()
        response = self.server.handle_message(
            _request(2, "resources/read",
                     {"uri": "blackbox://session/trace/1"
                            "?since=2999-01-01T00:00:00Z"}))
        self.assertEqual(response["error"]["code"], -32602)

    def test_resources_read_session_trace_template_window_bad_bounds_are_errors(self):
        # A bad bound, a reversed window, --last combined with
        # --since, and a bad --last are INVALID_PARAMS errors —
        # the resource-read analogue of the CLI trace's exit
        # code 2.
        self._init()
        self._record_session()
        base = "blackbox://session/trace/1"
        for suffix in ("?since=not-a-time",
                       "?since=2026-11-24T02:00:00Z"
                       "&until=2026-11-24T01:00:00Z",
                       "?last=1h&since=2026-11-24T01:00:00Z",
                       "?last=soon"):
            response = self.server.handle_message(
                _request(2, "resources/read",
                         {"uri": base + suffix}))
            self.assertEqual(response["error"]["code"], -32602,
                             suffix)

    def test_resources_read_session_aggregates_model_template(self):
        # The same per-model entry `blackbox costs` prints for
        # gpt-5: 2 calls, 1300 in + 390 out = $0.005525.
        self._init()
        self._record_session()
        response = self.server.handle_message(
            _request(6, "resources/read",
                     {"uri": "blackbox://session/aggregates/model/"
                            "gpt-5"}))
        content = response["result"]["contents"][0]
        self.assertEqual(
            content["uri"],
            "blackbox://session/aggregates/model/gpt-5")
        self.assertEqual(content["mimeType"], "application/json")
        entry = json.loads(content["text"])
        self.assertEqual(entry["model"], "gpt-5")
        self.assertEqual(entry["calls"], 2)
        self.assertEqual(entry["input_tokens"], 1300)
        self.assertEqual(entry["output_tokens"], 390)
        self.assertEqual(entry["tokens"], 0)
        self.assertEqual(entry["cost_usd"], 0.005525)

    def test_resources_read_session_aggregates_model_template_slash(self):
        # A model name with a slash in it is one model, not two
        # path segments: the name is the whole remainder of the
        # URI. An unpriced model costs None, never a guessed
        # number.
        self._init()
        _call(self.server, 2, "blackbox_record",
              {"type": "model_call",
               "payload": {"model": "acme/gpt-x", "prompt": "hi"},
               "tokens": {"input": 10, "output": 5}})
        response = self.server.handle_message(
            _request(3, "resources/read",
                     {"uri": "blackbox://session/aggregates/model/"
                            "acme/gpt-x"}))
        entry = json.loads(response["result"]["contents"][0]["text"])
        self.assertEqual(entry["model"], "acme/gpt-x")
        self.assertEqual(entry["calls"], 1)
        self.assertEqual(entry["input_tokens"], 10)
        self.assertEqual(entry["output_tokens"], 5)
        self.assertIsNone(entry["cost_usd"])

    def test_resources_read_session_aggregates_model_template_unknown(self):
        self._init()
        self._record_session()
        response = self.server.handle_message(
            _request(2, "resources/read",
                     {"uri": "blackbox://session/aggregates/model/"
                            "claude-sonnet-4"}))
        self.assertEqual(response["error"]["code"], -32602)

    def test_resources_read_session_aggregates_model_template_needs_name(self):
        self._init()
        response = self.server.handle_message(
            _request(2, "resources/read",
                     {"uri": "blackbox://session/aggregates/model/"}))
        self.assertEqual(response["error"]["code"], -32602)

    def test_resources_read_session_aggregates_tool_template(self):
        # The same per-tool entry `blackbox costs` prints for
        # search: 1 call, charged with the 1200 in + 340 out of
        # the model call that invoked it = $0.0049.
        self._init()
        self._record_session()
        response = self.server.handle_message(
            _request(6, "resources/read",
                     {"uri": "blackbox://session/aggregates/tool/"
                            "search"}))
        content = response["result"]["contents"][0]
        self.assertEqual(
            content["uri"],
            "blackbox://session/aggregates/tool/search")
        self.assertEqual(content["mimeType"], "application/json")
        entry = json.loads(content["text"])
        self.assertEqual(entry["tool"], "search")
        self.assertEqual(entry["calls"], 1)
        self.assertEqual(entry["input_tokens"], 1200)
        self.assertEqual(entry["output_tokens"], 340)
        self.assertEqual(entry["tokens"], 0)
        self.assertEqual(entry["cost_usd"], 0.0049)
        self.assertEqual(entry["unpriced_calls"], 0)

    def test_resources_read_session_aggregates_tool_template_unknown(self):
        self._init()
        self._record_session()
        response = self.server.handle_message(
            _request(2, "resources/read",
                     {"uri": "blackbox://session/aggregates/tool/"
                            "browse"}))
        self.assertEqual(response["error"]["code"], -32602)

    def test_resources_read_session_aggregates_tool_template_needs_name(self):
        self._init()
        response = self.server.handle_message(
            _request(2, "resources/read",
                     {"uri": "blackbox://session/aggregates/tool/"}))
        self.assertEqual(response["error"]["code"], -32602)

    def test_resources_read_session_record_template(self):
        # The same record the blackbox://session.json resource
        # carries for step 1: the tool_call, hash chain included.
        self._init()
        self._record_session()
        response = self.server.handle_message(
            _request(6, "resources/read",
                     {"uri": "blackbox://session/record/1"}))
        content = response["result"]["contents"][0]
        self.assertEqual(content["uri"], "blackbox://session/record/1")
        self.assertEqual(content["mimeType"], "application/json")
        record = json.loads(content["text"])
        self.assertEqual(record["seq"], 1)
        self.assertEqual(record["type"], "tool_call")
        self.assertEqual(record["payload"]["tool"], "search")
        self.assertEqual(record["payload"]["args"], {"query": "mcp"})
        self.assertIn("hash", record)
        self.assertIn("prev_hash", record)

    def test_resources_read_session_record_template_matches_session_json(self):
        # The template and the blackbox://session.json resource
        # always agree: the same record for the same step.
        self._init()
        self._record_session()
        response = self.server.handle_message(
            _request(6, "resources/read",
                     {"uri": "blackbox://session.json"}))
        records = json.loads(
            response["result"]["contents"][0]["text"])
        expected = next(r for r in records if r["seq"] == 2)
        response = self.server.handle_message(
            _request(7, "resources/read",
                     {"uri": "blackbox://session/record/2"}))
        record = json.loads(
            response["result"]["contents"][0]["text"])
        self.assertEqual(record, expected)

    def test_resources_read_session_record_template_unknown_seq(self):
        self._init()
        self._record_session()
        response = self.server.handle_message(
            _request(2, "resources/read",
                     {"uri": "blackbox://session/record/99"}))
        self.assertEqual(response["error"]["code"], -32602)

    def test_resources_read_session_record_template_bad_seq(self):
        self._init()
        response = self.server.handle_message(
            _request(2, "resources/read",
                     {"uri": "blackbox://session/record/abc"}))
        self.assertEqual(response["error"]["code"], -32602)

    def test_resources_read_session_record_template_needs_seq(self):
        self._init()
        response = self.server.handle_message(
            _request(2, "resources/read",
                     {"uri": "blackbox://session/record/"}))
        self.assertEqual(response["error"]["code"], -32602)

    def test_resources_read_session_transcript_template(self):
        # The same formatted line the blackbox://session resource
        # carries for step 1: the tool_call, one line.
        self._init()
        self._record_session()
        response = self.server.handle_message(
            _request(6, "resources/read",
                     {"uri": "blackbox://session/transcript/1"}))
        content = response["result"]["contents"][0]
        self.assertEqual(content["uri"],
                         "blackbox://session/transcript/1")
        self.assertEqual(content["mimeType"], "text/plain")
        self.assertIn("tool_call", content["text"])
        self.assertIn("search(query=mcp)", content["text"])

    def test_resources_read_session_transcript_template_matches_transcript(self):
        # The template and the blackbox://session resource always
        # agree: the template text is the transcript's line(s) for
        # that step, including the prompt line of a model_call.
        self._init()
        self._record_session()
        response = self.server.handle_message(
            _request(6, "resources/read",
                     {"uri": "blackbox://session"}))
        transcript = response["result"]["contents"][0]["text"]
        response = self.server.handle_message(
            _request(7, "resources/read",
                     {"uri": "blackbox://session/transcript/0"}))
        text = response["result"]["contents"][0]["text"]
        self.assertIn(text, transcript)
        self.assertIn("model_call", text)
        self.assertIn("prompt: hi", text)

    def test_resources_read_session_transcript_template_unknown_seq(self):
        self._init()
        self._record_session()
        response = self.server.handle_message(
            _request(2, "resources/read",
                     {"uri": "blackbox://session/transcript/99"}))
        self.assertEqual(response["error"]["code"], -32602)

    def test_resources_read_session_transcript_template_bad_seq(self):
        self._init()
        response = self.server.handle_message(
            _request(2, "resources/read",
                     {"uri": "blackbox://session/transcript/abc"}))
        self.assertEqual(response["error"]["code"], -32602)

    def test_resources_read_session_transcript_template_needs_seq(self):
        self._init()
        response = self.server.handle_message(
            _request(2, "resources/read",
                     {"uri": "blackbox://session/transcript/"}))
        self.assertEqual(response["error"]["code"], -32602)

    def test_resources_read_session_span_template(self):
        # One step as a single OTLP span: the step number as the
        # span id, the step kind as the span name, times from the
        # record's timestamps, and the priced cost as an attribute.
        self._init()
        self._record_session()
        response = self.server.handle_message(
            _request(6, "resources/read",
                     {"uri": "blackbox://session/span/0"}))
        content = response["result"]["contents"][0]
        self.assertEqual(content["uri"], "blackbox://session/span/0")
        self.assertEqual(content["mimeType"], "application/json")
        span = json.loads(content["text"])
        records = load_records(self.path)
        self.assertEqual(span["spanId"], "%016x" % 0)
        self.assertEqual(span["name"], "model_call")
        self.assertEqual(span["traceId"], records[0]["hash"][:32])
        self.assertNotIn("parentSpanId", span)  # first step is the root
        self.assertEqual(span["startTimeUnixNano"],
                         _nanos(records[0]["ts"]))
        self.assertEqual(span["endTimeUnixNano"],
                         _nanos(records[1]["ts"]))
        attrs = {a["key"]: a["value"] for a in span["attributes"]}
        self.assertEqual(attrs["blackbox.seq"]["intValue"], "0")
        self.assertEqual(attrs["blackbox.hash"]["stringValue"],
                         records[0]["hash"])
        self.assertEqual(attrs["gen_ai.request.model"]["stringValue"],
                         "gpt-5")
        self.assertEqual(attrs["gen_ai.usage.input_tokens"]["intValue"],
                         "1200")
        self.assertEqual(attrs["gen_ai.usage.output_tokens"]["intValue"],
                         "340")
        # gpt-5, 1200 in + 340 out = $0.0049.
        self.assertEqual(attrs["blackbox.cost_usd"]["doubleValue"],
                         0.0049)

    def test_resources_read_session_span_template_matches_export_record_and_transcript(self):
        # The template span agrees with the full `blackbox otel`
        # export (same times, parent and attributes — only the
        # span id and name differ, by design), with the raw record,
        # and with the transcript line for the same step.
        self._init()
        self._record_session()
        response = self.server.handle_message(
            _request(6, "resources/read",
                     {"uri": "blackbox://session/span/1"}))
        span = json.loads(response["result"]["contents"][0]["text"])
        records = load_records(self.path)
        exported = records_to_spans(records)[1]
        self.assertEqual(span["traceId"], exported["traceId"])
        self.assertEqual(span["startTimeUnixNano"],
                         exported["startTimeUnixNano"])
        self.assertEqual(span["endTimeUnixNano"],
                         exported["endTimeUnixNano"])
        self.assertEqual(span["parentSpanId"],
                         exported["parentSpanId"])
        self.assertEqual(span["attributes"], exported["attributes"])
        attrs = {a["key"]: a["value"] for a in span["attributes"]}
        self.assertEqual(
            json.loads(attrs["blackbox.payload"]["stringValue"]),
            records[1]["payload"])
        self.assertEqual(attrs["tool.name"]["stringValue"], "search")
        response = self.server.handle_message(
            _request(7, "resources/read",
                     {"uri": "blackbox://session/transcript/1"}))
        line = response["result"]["contents"][0]["text"]
        self.assertIn("tool_call", line)
        self.assertIn("search(query=mcp)", line)

    def test_resources_read_session_span_template_last_step_without_summary(self):
        # No summary yet: the last step's span ends 1 ms after it
        # starts, the same minimum-duration fallback as the full
        # export — no end time is invented from a later step.
        self._init()
        self._record_session()
        response = self.server.handle_message(
            _request(6, "resources/read",
                     {"uri": "blackbox://session/span/3"}))
        span = json.loads(response["result"]["contents"][0]["text"])
        self.assertEqual(int(span["endTimeUnixNano"])
                         - int(span["startTimeUnixNano"]), 1_000_000)

    def test_resources_read_session_span_template_last_step_ends_at_summary(self):
        self._init()
        self._record_session()
        self.server.close()  # writes the summary record
        records = load_records(self.path)
        summary = records[-1]
        self.assertEqual(summary["type"], "summary")
        response = self.server.handle_message(
            _request(6, "resources/read",
                     {"uri": "blackbox://session/span/3"}))
        span = json.loads(response["result"]["contents"][0]["text"])
        self.assertEqual(span["endTimeUnixNano"], _nanos(summary["ts"]))

    def test_resources_read_session_span_template_unpriced_has_no_cost(self):
        # An unknown model, and a model_call with a plain int token
        # count (no input/output split), are both unpriced — the
        # span carries no cost attribute, never a guessed number.
        self._init()
        _call(self.server, 2, "blackbox_record",
              {"type": "model_call",
               "payload": {"model": "mystery-model", "prompt": "hi"},
               "tokens": {"input": 100, "output": 50}})
        _call(self.server, 3, "blackbox_record",
              {"type": "model_call",
               "payload": {"model": "gpt-5", "prompt": "hi"},
               "tokens": 120})
        for seq in (0, 1):
            response = self.server.handle_message(
                _request(6, "resources/read",
                         {"uri": "blackbox://session/span/%d" % seq}))
            span = json.loads(response["result"]["contents"][0]["text"])
            keys = [a["key"] for a in span["attributes"]]
            self.assertNotIn("blackbox.cost_usd", keys)

    def test_resources_read_session_span_template_unknown_seq(self):
        self._init()
        self._record_session()
        response = self.server.handle_message(
            _request(2, "resources/read",
                     {"uri": "blackbox://session/span/99"}))
        self.assertEqual(response["error"]["code"], -32602)

    def test_resources_read_session_span_template_bad_seq(self):
        self._init()
        response = self.server.handle_message(
            _request(2, "resources/read",
                     {"uri": "blackbox://session/span/abc"}))
        self.assertEqual(response["error"]["code"], -32602)

    def test_resources_read_session_span_template_needs_seq(self):
        self._init()
        response = self.server.handle_message(
            _request(2, "resources/read",
                     {"uri": "blackbox://session/span/"}))
        self.assertEqual(response["error"]["code"], -32602)

    def test_resources_read_session_span_template_summary_is_not_a_step(self):
        self._init()
        self._record_session()
        self.server.close()  # writes the summary record (seq 4)
        response = self.server.handle_message(
            _request(2, "resources/read",
                     {"uri": "blackbox://session/span/4"}))
        self.assertEqual(response["error"]["code"], -32602)

    def test_resources_read_unknown_uri_is_an_error(self):
        self._init()
        response = self.server.handle_message(
            _request(2, "resources/read",
                     {"uri": "blackbox://nope"}))
        self.assertEqual(response["error"]["code"], -32602)

    def test_prompts_list(self):
        self._init()
        response = self.server.handle_message(
            _request(2, "prompts/list"))
        prompts = response["result"]["prompts"]
        self.assertEqual([p["name"] for p in prompts],
                         ["analyze_session", "summarize_costs",
                          "summarize_anomalies", "summarize_sampling",
                          "summarize_trace", "summarize_diff",
                          "summarize_retention",
                          "summarize_aggregates"])
        self.assertEqual(prompts[0]["arguments"][0]["name"], "focus")
        self.assertFalse(prompts[0]["arguments"][0]["required"])
        self.assertEqual(prompts[1]["arguments"], [])
        self.assertEqual(prompts[2]["arguments"], [])
        self.assertEqual(prompts[3]["arguments"], [])
        self.assertEqual(prompts[4]["arguments"], [])
        self.assertEqual(prompts[5]["arguments"][0]["name"],
                         "other_path")
        self.assertTrue(prompts[5]["arguments"][0]["required"])
        self.assertEqual(
            [a["name"] for a in prompts[6]["arguments"]],
            ["keep_days", "keep_n"])
        self.assertFalse(prompts[6]["arguments"][0]["required"])
        self.assertFalse(prompts[6]["arguments"][1]["required"])
        self.assertEqual(prompts[7]["arguments"], [])

    def test_prompts_get_includes_transcript_and_verification(self):
        self._init()
        _call(self.server, 2, "blackbox_record",
              {"type": "tool_call",
               "payload": {"tool": "search", "args": {"query": "mcp"}}})
        response = self.server.handle_message(
            _request(3, "prompts/get",
                     {"name": "analyze_session"}))
        message = response["result"]["messages"][0]
        self.assertEqual(message["role"], "user")
        text = message["content"]["text"]
        self.assertIn("tool_call", text)
        self.assertIn("search", text)
        self.assertIn("hash chain intact", text)

    def test_prompts_get_with_focus(self):
        self._init()
        _call(self.server, 2, "blackbox_record",
              {"type": "model_call",
               "payload": {"model": "gpt-5", "prompt": "hi"},
               "tokens": {"input": 10, "output": 5}})
        response = self.server.handle_message(
            _request(3, "prompts/get",
                     {"name": "analyze_session",
                      "arguments": {"focus": "cost"}}))
        text = response["result"]["messages"][0]["content"]["text"]
        self.assertIn("most expensive", text)
        self.assertIn("gpt-5", text)

    def test_prompts_get_unknown_prompt_is_an_error(self):
        self._init()
        response = self.server.handle_message(
            _request(2, "prompts/get", {"name": "nope"}))
        self.assertEqual(response["error"]["code"], -32602)

    def test_prompts_get_summarize_costs(self):
        self._init()
        _call(self.server, 2, "blackbox_record",
              {"type": "model_call",
               "payload": {"model": "gpt-5", "prompt": "hi"},
               "tokens": {"input": 1200, "output": 340}})
        response = self.server.handle_message(
            _request(3, "prompts/get",
                     {"name": "summarize_costs"}))
        message = response["result"]["messages"][0]
        self.assertEqual(message["role"], "user")
        text = message["content"]["text"]
        # 1200 * 1.25 + 340 * 10.00 per 1M tokens = $0.0049.
        self.assertIn("gpt-5", text)
        self.assertIn("1200 input + 340 output tokens", text)
        self.assertIn("$0.0049", text)
        self.assertIn("Total priced cost: $0.0049", text)
        self.assertIn("gpt-5", text)  # transcript included too

    def test_prompts_get_summarize_costs_unpriced_calls(self):
        self._init()
        _call(self.server, 2, "blackbox_record",
              {"type": "model_call",
               "payload": {"model": "gpt-5", "prompt": "hi"},
               "tokens": 500})
        _call(self.server, 3, "blackbox_record",
              {"type": "model_call",
               "payload": {"model": "mystery-model", "prompt": "hi"},
               "tokens": {"input": 10, "output": 5}})
        response = self.server.handle_message(
            _request(4, "prompts/get",
                     {"name": "summarize_costs"}))
        text = response["result"]["messages"][0]["content"]["text"]
        self.assertIn("500 tokens with no input/output split", text)
        self.assertIn("not priced: no input/output split", text)
        self.assertIn("cost unknown: no published price", text)
        self.assertNotIn("Total priced cost", text)

    def test_prompts_get_summarize_costs_empty_session(self):
        self._init()
        response = self.server.handle_message(
            _request(2, "prompts/get",
                     {"name": "summarize_costs"}))
        text = response["result"]["messages"][0]["content"]["text"]
        self.assertIn("No model calls recorded yet.", text)

    def test_prompts_get_summarize_anomalies(self):
        self._init()
        # Error result, from a tool that was called twice with the
        # same arguments.
        _call(self.server, 2, "blackbox_record",
              {"type": "tool_call",
               "payload": {"tool": "search",
                           "args": {"query": "mcp"}}})
        _call(self.server, 3, "blackbox_record",
              {"type": "tool_result",
               "payload": {"tool": "search",
                           "result": "Error: timed out"}})
        _call(self.server, 4, "blackbox_record",
              {"type": "tool_call",
               "payload": {"tool": "search",
                           "args": {"query": "mcp"}}})
        # A call with no result at all, and a step that had a secret
        # redacted before writing.
        _call(self.server, 5, "blackbox_record",
              {"type": "tool_call",
               "payload": {"tool": "fetch",
                           "args": {"url": "https://example.com"}}})
        _call(self.server, 6, "blackbox_record",
              {"type": "tool_call",
               "payload": {"tool": "login",
                           "args": {"api_key": "sk-abcdefghijklmnop"}}})
        response = self.server.handle_message(
            _request(7, "prompts/get",
                     {"name": "summarize_anomalies"}))
        message = response["result"]["messages"][0]
        self.assertEqual(message["role"], "user")
        text = message["content"]["text"]
        self.assertIn("step 1: search returned an error result", text)
        self.assertIn("Error: timed out", text)
        self.assertIn("tool search called 2 times with identical "
                      "arguments", text)
        self.assertIn("step 3: tool_call fetch has no matching "
                      "tool_result", text)
        self.assertIn("step 4: tool_call contained a secret that was "
                      "redacted before writing", text)
        self.assertIn("heuristics", text)
        self.assertIn("search", text)  # transcript included too

    def test_prompts_get_summarize_anomalies_error_field(self):
        self._init()
        _call(self.server, 2, "blackbox_record",
              {"type": "tool_result",
               "payload": {"tool": "search",
                           "error": "upstream 500"}})
        response = self.server.handle_message(
            _request(3, "prompts/get",
                     {"name": "summarize_anomalies"}))
        text = response["result"]["messages"][0]["content"]["text"]
        self.assertIn("step 0: search returned a result with an "
                      "'error' field", text)

    def test_prompts_get_summarize_anomalies_clean_session(self):
        self._init()
        _call(self.server, 2, "blackbox_record",
              {"type": "tool_call",
               "payload": {"tool": "search",
                           "args": {"query": "mcp"}}})
        _call(self.server, 3, "blackbox_record",
              {"type": "tool_result",
               "payload": {"tool": "search", "result": "3 results"}})
        response = self.server.handle_message(
            _request(4, "prompts/get",
                     {"name": "summarize_anomalies"}))
        text = response["result"]["messages"][0]["content"]["text"]
        self.assertIn("No anomalies found", text)

    def test_prompts_get_summarize_anomalies_empty_session(self):
        self._init()
        response = self.server.handle_message(
            _request(2, "prompts/get",
                     {"name": "summarize_anomalies"}))
        text = response["result"]["messages"][0]["content"]["text"]
        self.assertIn("No anomalies found", text)

    def test_prompts_get_summarize_sampling(self):
        self._init()
        # Four steps, no trigger: with seed 0 the RNG keeps steps
        # 2 and 3 (draws 0.84, 0.76, 0.42, 0.26 against 0.5), so
        # the model_call and the search call are sampled out and
        # the kept fetch call has a parent gap.
        _call(self.server, 2, "blackbox_record",
              {"type": "model_call",
               "payload": {"model": "gpt-5", "prompt": "hi"},
               "tokens": 100})
        _call(self.server, 3, "blackbox_record",
              {"type": "tool_call",
               "payload": {"tool": "search",
                           "args": {"query": "mcp"}}})
        _call(self.server, 4, "blackbox_record",
              {"type": "tool_call",
               "payload": {"tool": "fetch",
                           "args": {"url": "https://example.com"}}})
        _call(self.server, 5, "blackbox_record",
              {"type": "tool_result",
               "payload": {"tool": "fetch", "result": "ok"}})
        response = self.server.handle_message(
            _request(6, "prompts/get",
                     {"name": "summarize_sampling"}))
        message = response["result"]["messages"][0]
        self.assertEqual(message["role"], "user")
        text = message["content"]["text"]
        self.assertIn("Decision: sampled", text)
        self.assertIn("Kept 2 of 4 spans, 2 sampled out", text)
        self.assertIn("seed 0", text)
        self.assertIn("Parent gaps: 1 kept span(s)", text)
        self.assertIn("model_call (gpt-5): 1", text)
        self.assertIn("tool_call (search): 1", text)
        self.assertIn("Total priced cost: unpriced", text)
        self.assertIn("still in the log", text)
        self.assertIn("search", text)  # transcript included too

    def test_prompts_get_summarize_sampling_kept_all(self):
        self._init()
        _call(self.server, 2, "blackbox_record",
              {"type": "tool_call",
               "payload": {"tool": "search",
                           "args": {"query": "mcp"}}})
        _call(self.server, 3, "blackbox_record",
              {"type": "tool_result",
               "payload": {"tool": "search",
                           "error": "upstream 500"}})
        response = self.server.handle_message(
            _request(4, "prompts/get",
                     {"name": "summarize_sampling"}))
        text = response["result"]["messages"][0]["content"]["text"]
        self.assertIn("Decision: kept_all", text)
        self.assertIn("error", text)
        self.assertIn("Kept 2 of 2 spans, 0 sampled out", text)
        self.assertIn("Nothing sampled out", text)
        self.assertNotIn("Parent gaps", text)

    def test_prompts_get_summarize_sampling_empty_session(self):
        self._init()
        response = self.server.handle_message(
            _request(2, "prompts/get",
                     {"name": "summarize_sampling"}))
        text = response["result"]["messages"][0]["content"]["text"]
        self.assertIn("No steps recorded yet.", text)

    def test_prompts_get_summarize_trace(self):
        self._init()
        # Six steps, no trigger: with seed 0 the RNG keeps steps
        # 2, 3 and 5 (draws 0.84, 0.76, 0.42, 0.26, 0.51, 0.40
        # against 0.5), so the kept tool_result for search and the
        # kept tool_result for fetch both have a causal parent
        # that was sampled out — 2 parent gaps.
        _call(self.server, 2, "blackbox_record",
              {"type": "model_call",
               "payload": {"model": "gpt-5", "prompt": "hi"},
               "tokens": {"input": 1200, "output": 340}})
        _call(self.server, 3, "blackbox_record",
              {"type": "tool_call",
               "payload": {"tool": "search",
                           "args": {"query": "mcp"}}})
        _call(self.server, 4, "blackbox_record",
              {"type": "tool_result",
               "payload": {"tool": "search", "result": "3 results"}})
        _call(self.server, 5, "blackbox_record",
              {"type": "model_call",
               "payload": {"model": "gpt-5", "prompt": "again"},
               "tokens": {"input": 100, "output": 50}})
        _call(self.server, 6, "blackbox_record",
              {"type": "tool_call",
               "payload": {"tool": "fetch",
                           "args": {"url": "https://example.com"}}})
        _call(self.server, 7, "blackbox_record",
              {"type": "tool_result",
               "payload": {"tool": "fetch", "result": "ok"}})
        response = self.server.handle_message(
            _request(8, "prompts/get",
                     {"name": "summarize_trace"}))
        message = response["result"]["messages"][0]
        self.assertEqual(message["role"], "user")
        text = message["content"]["text"]
        self.assertIn("Trace: 6 steps, 1 root(s), max depth 6", text)
        self.assertIn("model_call: 2, tool_call: 2, tool_result: 2",
                      text)
        self.assertIn("Slowest steps:", text)
        self.assertIn("ms", text)
        self.assertIn("Error steps: none.", text)
        self.assertIn("Parent gaps: 2 kept span(s) whose causal "
                      "parent would be sampled out", text)
        self.assertIn("step 2: tool_result search "
                      "(causal parent step 1)", text)
        self.assertIn("step 5: tool_result fetch "
                      "(causal parent step 4)", text)
        self.assertIn("gpt-5: 1300 input + 390 output tokens, "
                      "$0.005525", text)
        self.assertIn("Total priced cost: $0.005525", text)
        self.assertIn("  fetch: 1", text)
        self.assertIn("  search: 1", text)
        self.assertIn("[0] model_call gpt-5", text)
        self.assertIn("[5] tool_result fetch", text)
        self.assertIn("reconstruction", text)
        self.assertIn("search", text)  # transcript included too

    def test_prompts_get_summarize_trace_kept_all(self):
        self._init()
        _call(self.server, 2, "blackbox_record",
              {"type": "tool_call",
               "payload": {"tool": "search",
                           "args": {"query": "mcp"}}})
        _call(self.server, 3, "blackbox_record",
              {"type": "tool_result",
               "payload": {"tool": "search",
                           "error": "upstream 500"}})
        response = self.server.handle_message(
            _request(4, "prompts/get",
                     {"name": "summarize_trace"}))
        text = response["result"]["messages"][0]["content"]["text"]
        self.assertIn("Trace: 2 steps, 1 root(s), max depth 2", text)
        self.assertIn("step 1: tool_result search: upstream 500", text)
        self.assertIn("Parent gaps: none — the trace would be kept "
                      "whole (trigger: error)", text)
        self.assertIn("Tool calls:", text)
        self.assertIn("  search: 1", text)

    def test_prompts_get_summarize_trace_empty_session(self):
        self._init()
        response = self.server.handle_message(
            _request(2, "prompts/get",
                     {"name": "summarize_trace"}))
        text = response["result"]["messages"][0]["content"]["text"]
        self.assertIn("No steps recorded yet.", text)

    def _write_other_run(self, path, query="mcp", result="3 results",
                         extra=False, error_result=False):
        # Run B: same shape as the session below, with a different
        # search query, a different result, and optionally one extra
        # step at the end. With error_result, the search result
        # carries an error field instead of a result string.
        with Recorder(path) as rec:
            rec.record("model_call", {"model": "gpt-5", "prompt": "hi"},
                       tokens={"input": 1200, "output": 340})
            rec.record("tool_call", {"tool": "search",
                                     "args": {"query": query}})
            if error_result:
                rec.record("tool_result", {"tool": "search",
                                           "error": "timed out"})
            else:
                rec.record("tool_result", {"tool": "search",
                                           "result": result})
            rec.record("model_call", {"model": "gpt-5",
                                      "prompt": "again"},
                       tokens={"input": 100, "output": 50})
            if extra:
                rec.record("tool_call", {"tool": "fetch",
                                         "args": {"url":
                                                  "https://example.com"}})

    def _record_session(self):
        _call(self.server, 2, "blackbox_record",
              {"type": "model_call",
               "payload": {"model": "gpt-5", "prompt": "hi"},
               "tokens": {"input": 1200, "output": 340}})
        _call(self.server, 3, "blackbox_record",
              {"type": "tool_call",
               "payload": {"tool": "search",
                           "args": {"query": "mcp"}}})
        _call(self.server, 4, "blackbox_record",
              {"type": "tool_result",
               "payload": {"tool": "search", "result": "3 results"}})
        _call(self.server, 5, "blackbox_record",
              {"type": "model_call",
               "payload": {"model": "gpt-5", "prompt": "again"},
               "tokens": {"input": 100, "output": 50}})

    def test_prompts_get_summarize_diff(self):
        self._init()
        self._record_session()
        other = os.path.join(self.tmp.name, "other.jsonl")
        self._write_other_run(other, query="other",
                              extra=True, error_result=True)
        response = self.server.handle_message(
            _request(6, "prompts/get",
                     {"name": "summarize_diff",
                      "arguments": {"other_path": other}}))
        message = response["result"]["messages"][0]
        self.assertEqual(message["role"], "user")
        text = message["content"]["text"]
        self.assertIn("Run A (this session): 4 steps, 1690 tokens; "
                      "run B (other run): 5 steps, 1690 tokens "
                      "(token diff +0)", text)
        self.assertIn("CHANGED step 1:", text)
        self.assertIn("CHANGED step 2:", text)
        self.assertIn("ADDED step 4 (only in run B):", text)
        self.assertIn("fetch", text)
        # gpt-5 in both runs: 1300 in + 390 out = $0.005525 each.
        self.assertIn("gpt-5: 1690 tokens -> 1690 tokens (+0), "
                      "$0.005525 -> $0.005525 (+$0)", text)
        self.assertIn("New errors (in run B, not run A):", text)
        self.assertIn("step 2: tool_result search", text)
        self.assertIn("position-by-position", text)
        self.assertIn("Run A transcript (this session):", text)
        self.assertIn("Run B transcript (other run):", text)

    def test_prompts_get_summarize_diff_identical(self):
        self._init()
        self._record_session()
        other = os.path.join(self.tmp.name, "other.jsonl")
        self._write_other_run(other)
        response = self.server.handle_message(
            _request(6, "prompts/get",
                     {"name": "summarize_diff",
                      "arguments": {"other_path": other}}))
        text = response["result"]["messages"][0]["content"]["text"]
        self.assertIn("IDENTICAL: 4 steps match", text)
        self.assertIn("token diff +0", text)
        self.assertIn("Error steps: none in either run.", text)

    def test_prompts_get_summarize_diff_fixed_error(self):
        self._init()
        _call(self.server, 2, "blackbox_record",
              {"type": "tool_call",
               "payload": {"tool": "search",
                           "args": {"query": "mcp"}}})
        _call(self.server, 3, "blackbox_record",
              {"type": "tool_result",
               "payload": {"tool": "search",
                           "error": "upstream 500"}})
        other = os.path.join(self.tmp.name, "other.jsonl")
        with Recorder(other) as rec:
            rec.record("tool_call", {"tool": "search",
                                     "args": {"query": "mcp"}})
            rec.record("tool_result", {"tool": "search",
                                       "result": "3 results"})
        response = self.server.handle_message(
            _request(4, "prompts/get",
                     {"name": "summarize_diff",
                      "arguments": {"other_path": other}}))
        text = response["result"]["messages"][0]["content"]["text"]
        self.assertIn("CHANGED step 1:", text)
        self.assertIn("Errors fixed (in run A, not run B):", text)
        self.assertIn("step 1: tool_result search", text)

    def test_prompts_get_summarize_diff_needs_other_path(self):
        self._init()
        response = self.server.handle_message(
            _request(2, "prompts/get",
                     {"name": "summarize_diff"}))
        self.assertEqual(response["error"]["code"], -32602)

    def test_prompts_get_summarize_diff_missing_file_is_an_error(self):
        self._init()
        response = self.server.handle_message(
            _request(2, "prompts/get",
                     {"name": "summarize_diff",
                      "arguments": {"other_path": os.path.join(
                          self.tmp.name, "nope.jsonl")}}))
        self.assertEqual(response["error"]["code"], -32602)

    def test_prompts_get_summarize_retention(self):
        self._init()
        self._record_session()
        response = self.server.handle_message(
            _request(6, "prompts/get",
                     {"name": "summarize_retention"}))
        message = response["result"]["messages"][0]
        self.assertEqual(message["role"], "user")
        text = message["content"]["text"]
        self.assertIn("Never compacted", text)
        self.assertIn("Log now: 4 step(s)", text)
        self.assertIn("spanning 0.0 day(s)", text)
        self.assertIn("--keep-days 30: would remove nothing "
                      "(4 step(s) kept)", text)
        self.assertIn("--keep-n 1000: would remove nothing "
                      "(4 step(s) kept)", text)
        self.assertIn("Removed records are gone", text)
        self.assertIn("dry runs", text)
        self.assertIn("search", text)  # transcript included too

    def test_prompts_get_summarize_retention_keep_n_argument(self):
        self._init()
        self._record_session()
        response = self.server.handle_message(
            _request(6, "prompts/get",
                     {"name": "summarize_retention",
                      "arguments": {"keep_n": 2}}))
        text = response["result"]["messages"][0]["content"]["text"]
        self.assertIn("--keep-days 30: would remove nothing "
                      "(4 step(s) kept)", text)
        self.assertIn("--keep-n 2: would remove 2 record(s), keep 2",
                      text)
        self.assertIn("new checkpoint would certify head hash", text)

    def test_prompts_get_summarize_retention_compacted(self):
        self._init()
        self._record_session()
        self.server.close()
        from blackbox.compact import compact_log
        result = compact_log(self.path, keep_n=2)
        self.assertTrue(result["changed"])
        self.assertEqual(result["compacted"], 2)
        self.server = MCPServer(self.path, task="mcp test")
        self._init()
        response = self.server.handle_message(
            _request(2, "prompts/get",
                     {"name": "summarize_retention"}))
        text = response["result"]["messages"][0]["content"]["text"]
        self.assertIn("Already compacted: 2 record(s) were removed",
                      text)
        self.assertIn("certified by a checkpoint", text)
        self.assertIn("The checkpoint carries the dropped run's "
                      "totals: 4 events, 1690 tokens", text)
        self.assertIn("Log now: 2 step(s)", text)
        self.assertIn("--keep-n 1000: would remove nothing "
                      "(2 step(s) kept)", text)

    def test_prompts_get_summarize_retention_empty_session(self):
        self._init()
        response = self.server.handle_message(
            _request(2, "prompts/get",
                     {"name": "summarize_retention"}))
        text = response["result"]["messages"][0]["content"]["text"]
        self.assertIn("No steps recorded yet.", text)

    def test_prompts_get_summarize_retention_bad_arguments(self):
        self._init()
        for arguments in ({"keep_days": -1}, {"keep_n": -5},
                          {"keep_days": "30"}, {"keep_n": 2.5}):
            response = self.server.handle_message(
                _request(2, "prompts/get",
                         {"name": "summarize_retention",
                          "arguments": arguments}))
            self.assertEqual(response["error"]["code"], -32602)

    def test_prompts_get_summarize_aggregates(self):
        self._init()
        self._record_session()
        response = self.server.handle_message(
            _request(6, "prompts/get",
                     {"name": "summarize_aggregates"}))
        message = response["result"]["messages"][0]
        self.assertEqual(message["role"], "user")
        text = message["content"]["text"]
        self.assertIn("session: 4 steps", text)
        self.assertIn("tokens: 1690  cost: $0.005525", text)
        self.assertIn("  model_call: 2", text)
        self.assertIn("  tool search: 1 calls", text)
        # gpt-5: 1300 in + 390 out = $0.005525.
        self.assertIn("gpt-5: 2 calls, 1300 in / 390 out / 0 tokens "
                      "-> $0.005525", text)
        self.assertIn("slowest step: seq", text)
        self.assertIn("not in these totals", text)
        self.assertIn("read-only", text)
        self.assertIn("search", text)  # transcript included too

    def test_prompts_get_summarize_aggregates_with_error(self):
        self._init()
        _call(self.server, 2, "blackbox_record",
              {"type": "tool_call",
               "payload": {"tool": "search",
                           "args": {"query": "mcp"}}})
        _call(self.server, 3, "blackbox_record",
              {"type": "tool_result",
               "payload": {"tool": "search",
                           "result": "error: timeout"}})
        response = self.server.handle_message(
            _request(4, "prompts/get",
                     {"name": "summarize_aggregates"}))
        text = response["result"]["messages"][0]["content"]["text"]
        self.assertIn("session: 2 steps", text)
        self.assertIn("errors: 1 (seqs 1)", text)

    def test_prompts_get_summarize_aggregates_compacted(self):
        self._init()
        self._record_session()
        self.server.close()
        from blackbox.compact import compact_log
        result = compact_log(self.path, keep_n=2)
        self.assertTrue(result["changed"])
        self.assertEqual(result["compacted"], 2)
        self.server = MCPServer(self.path, task="mcp test")
        self._init()
        response = self.server.handle_message(
            _request(2, "prompts/get",
                     {"name": "summarize_aggregates"}))
        text = response["result"]["messages"][0]["content"]["text"]
        self.assertIn("session: 2 steps", text)
        self.assertIn("compacted away earlier: 2 records "
                      "(not in these totals)", text)

    def test_prompts_get_summarize_aggregates_empty_session(self):
        self._init()
        response = self.server.handle_message(
            _request(2, "prompts/get",
                     {"name": "summarize_aggregates"}))
        text = response["result"]["messages"][0]["content"]["text"]
        self.assertIn("No steps recorded yet.", text)

    def test_resources_read_before_initialize_is_rejected(self):
        response = self.server.handle_message(
            _request(1, "resources/read",
                     {"uri": "blackbox://session"}))
        self.assertEqual(response["error"]["code"], -32600)

    def test_unknown_method_is_an_error(self):
        self._init()
        response = self.server.handle_message(
            _request(2, "sampling/createMessage"))
        self.assertEqual(response["error"]["code"], -32601)

    def test_notification_gets_no_reply(self):
        self._init()
        notification = {"jsonrpc": "2.0",
                        "method": "notifications/initialized"}
        self.assertIsNone(self.server.handle_message(notification))

    def test_serve_loop_over_stdio(self):
        lines = [
            _request(1, "initialize", {}),
            {"jsonrpc": "2.0", "method": "notifications/initialized"},
            _request(2, "tools/call", {"name": "blackbox_record",
                                       "arguments": {
                                           "type": "note",
                                           "payload": {"text": "via stdio"}}}),
            _request(3, "tools/call", {"name": "blackbox_verify",
                                       "arguments": {}}),
            "not json at all",
        ]
        stdin = io.StringIO("\n".join(
            l if isinstance(l, str) else json.dumps(l) for l in lines) + "\n")
        stdout = io.StringIO()
        self.server.serve(stdin, stdout)
        responses = [json.loads(line) for line in
                     stdout.getvalue().splitlines()]
        # initialize, record, verify, parse error; the notification
        # gets no reply.
        self.assertEqual(len(responses), 4)
        self.assertEqual(responses[0]["id"], 1)
        self.assertIn("recorded step 0",
                      responses[1]["result"]["content"][0]["text"])
        self.assertIn("hash chain intact",
                      responses[2]["result"]["content"][0]["text"])
        self.assertEqual(responses[3]["error"]["code"], -32700)
        # EOF closed the recorder: a summary record was written.
        self.assertTrue(verify(self.path)["ok"])
        records = load_records(self.path)
        self.assertEqual(records[-1]["type"], "summary")

    def test_recorded_run_verifies_after_close(self):
        self._init()
        _call(self.server, 2, "blackbox_record",
              {"type": "model_call",
               "payload": {"model": "gpt-5", "prompt": "hi"},
               "tokens": {"input": 10, "output": 5}})
        self.server.close()
        result = verify(self.path)
        self.assertTrue(result["ok"], result)


if __name__ == "__main__":
    unittest.main()
