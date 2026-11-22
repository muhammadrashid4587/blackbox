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
                                 "blackbox_replay"])
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

    def test_unknown_method_is_an_error(self):
        self._init()
        response = self.server.handle_message(
            _request(2, "resources/list"))
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
