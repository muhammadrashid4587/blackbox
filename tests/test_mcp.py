"""MCP server: handshake, tools/list, and the three tools over
newline-delimited JSON-RPC on stdio."""

import io
import json
import os
import tempfile
import unittest

from blackbox.mcp_server import MCPServer
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
                                 "blackbox_replay", "blackbox_sample"])
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
                                "blackbox://session.json"])
        for resource in response["result"]["resources"]:
            self.assertIn("mimeType", resource)

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
                          "summarize_anomalies"])
        self.assertEqual(prompts[0]["arguments"][0]["name"], "focus")
        self.assertFalse(prompts[0]["arguments"][0]["required"])
        self.assertEqual(prompts[1]["arguments"], [])
        self.assertEqual(prompts[2]["arguments"], [])

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
