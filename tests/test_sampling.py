"""MCP sampling: Sampler, StdioTransport, and the blackbox_sample
tool, all against a fake client transport — no real model needed."""

import io
import json
import os
import tempfile
import unittest

from blackbox.mcp_server import MCPServer
from blackbox.replay import load_records
from blackbox.sampling import (Sampler, SamplingDeclinedError, SamplingError,
                               SamplingNotSupportedError, StdioTransport)
from blackbox.verify import verify

SAMPLE_RESULT = {
    "model": "test-model",
    "stopReason": "endTurn",
    "role": "assistant",
    "content": {"type": "text", "text": "The session looks clean."},
}


class FakeTransport:
    """A fake MCP client: answers sampling/createMessage with a canned
    result, or raises the error it was given. Records every request."""

    def __init__(self, result=None, error=None):
        self.result = result if result is not None else SAMPLE_RESULT
        self.error = error
        self.requests = []

    def request(self, method, params):
        self.requests.append((method, params))
        if self.error is not None:
            raise self.error
        return self.result


def _messages():
    return [{"role": "user", "content": "Summarise this session."}]


class SamplerTest(unittest.TestCase):
    def test_sample_returns_text_model_and_stop_reason(self):
        sampler = Sampler(FakeTransport())
        result = sampler.sample(_messages())
        self.assertEqual(result.text, "The session looks clean.")
        self.assertEqual(result.model, "test-model")
        self.assertEqual(result.stop_reason, "endTurn")
        self.assertEqual(result.role, "assistant")

    def test_sample_sends_spec_params(self):
        transport = FakeTransport()
        sampler = Sampler(transport)
        sampler.sample(
            _messages(),
            model_preferences={"hints": [{"name": "fast"}],
                               "costPriority": 0.8},
            system_prompt="Be brief.",
            include_context="thisServer",
            temperature=0.5,
            max_tokens=256,
            stop_sequences=["STOP"])
        self.assertEqual(len(transport.requests), 1)
        method, params = transport.requests[0]
        self.assertEqual(method, "sampling/createMessage")
        self.assertEqual(params["messages"], [
            {"role": "user",
             "content": {"type": "text", "text": "Summarise this session."}}])
        self.assertEqual(params["modelPreferences"],
                         {"hints": [{"name": "fast"}], "costPriority": 0.8})
        self.assertEqual(params["systemPrompt"], "Be brief.")
        self.assertEqual(params["includeContext"], "thisServer")
        self.assertEqual(params["temperature"], 0.5)
        self.assertEqual(params["maxTokens"], 256)
        self.assertEqual(params["stopSequences"], ["STOP"])

    def test_sample_defaults_to_max_tokens_1024(self):
        transport = FakeTransport()
        Sampler(transport).sample(_messages())
        self.assertEqual(transport.requests[0][1]["maxTokens"], 1024)

    def test_sample_normalises_string_content(self):
        transport = FakeTransport()
        Sampler(transport).sample(
            [{"role": "assistant", "content": "Got it."}])
        messages = transport.requests[0][1]["messages"]
        self.assertEqual(messages, [{"role": "assistant", "content": {
            "type": "text", "text": "Got it."}}])

    def test_sample_redacts_secrets_before_sending(self):
        transport = FakeTransport()
        sampler = Sampler(transport)
        sampler.sample(
            [{"role": "user",
              "content": "My key is sk-abcdefghijklmnopqrst, analyse this."}],
            system_prompt="Bearer abcdefghijklmnopqrstuvwxyz0123456789")
        sent = transport.requests[0][1]
        text = sent["messages"][0]["content"]["text"]
        self.assertNotIn("sk-abcdefghijklmnopqrst", text)
        self.assertIn("[REDACTED]", text)
        self.assertNotIn("abcdefghijklmnopqrstuvwxyz0123456789",
                         sent["systemPrompt"])
        self.assertIn("[REDACTED]", sent["systemPrompt"])

    def test_sample_unsupported_propagates(self):
        sampler = Sampler(FakeTransport(
            error=SamplingNotSupportedError("no sampling here")))
        with self.assertRaises(SamplingNotSupportedError):
            sampler.sample(_messages())

    def test_sample_declined_propagates(self):
        sampler = Sampler(FakeTransport(
            error=SamplingDeclinedError("client refused")))
        with self.assertRaises(SamplingDeclinedError):
            sampler.sample(_messages())

    def test_sample_or_none_returns_none_on_decline(self):
        sampler = Sampler(FakeTransport(
            error=SamplingDeclinedError("client refused")))
        self.assertIsNone(sampler.sample_or_none(_messages()))

    def test_sample_or_none_returns_none_when_unsupported(self):
        sampler = Sampler(FakeTransport(
            error=SamplingNotSupportedError("no sampling here")))
        self.assertIsNone(sampler.sample_or_none(_messages()))

    def test_sample_or_none_still_returns_result_on_success(self):
        sampler = Sampler(FakeTransport())
        result = sampler.sample_or_none(_messages())
        self.assertEqual(result.text, "The session looks clean.")

    def test_sample_rejects_empty_messages(self):
        with self.assertRaises(ValueError):
            Sampler(FakeTransport()).sample([])

    def test_sample_rejects_bad_role(self):
        with self.assertRaises(ValueError):
            Sampler(FakeTransport()).sample(
                [{"role": "system", "content": "hi"}])

    def test_sample_rejects_bad_content(self):
        with self.assertRaises(ValueError):
            Sampler(FakeTransport()).sample(
                [{"role": "user", "content": {"type": "image"}}])

    def test_sample_rejects_bad_max_tokens(self):
        sampler = Sampler(FakeTransport())
        for bad in (0, -5, "1024", 1.5, True):
            with self.assertRaises(ValueError, msg="max_tokens=%r" % (bad,)):
                sampler.sample(_messages(), max_tokens=bad)

    def test_sample_rejects_bad_include_context(self):
        with self.assertRaises(ValueError):
            Sampler(FakeTransport()).sample(
                _messages(), include_context="everywhere")

    def test_sample_rejects_bad_priority(self):
        with self.assertRaises(ValueError):
            Sampler(FakeTransport()).sample(
                _messages(), model_preferences={"costPriority": 1.5})

    def test_sample_rejects_bad_temperature(self):
        with self.assertRaises(ValueError):
            Sampler(FakeTransport()).sample(
                _messages(), temperature=-0.5)

    def test_sample_rejects_non_text_result(self):
        sampler = Sampler(FakeTransport(result={
            "model": "test-model", "stopReason": "endTurn",
            "content": {"type": "image", "data": "abcd"}}))
        with self.assertRaises(SamplingError):
            sampler.sample(_messages())

    def test_sample_rejects_non_object_result(self):
        sampler = Sampler(FakeTransport(result="not a dict"))
        with self.assertRaises(SamplingError):
            sampler.sample(_messages())

    def test_nothing_sent_when_arguments_invalid(self):
        transport = FakeTransport()
        with self.assertRaises(ValueError):
            Sampler(transport).sample([])
        self.assertEqual(transport.requests, [])


class StdioTransportTest(unittest.TestCase):
    def _transport_with_reply(self, reply):
        stdin = io.StringIO(json.dumps(reply) + "\n")
        stdout = io.StringIO()
        return StdioTransport(stdin, stdout), stdout

    def test_request_writes_jsonrpc_and_returns_result(self):
        transport, stdout = self._transport_with_reply(
            {"jsonrpc": "2.0", "id": 1, "result": SAMPLE_RESULT})
        result = transport.request("sampling/createMessage",
                                   {"messages": _messages(), "maxTokens": 10})
        sent = json.loads(stdout.getvalue().strip())
        self.assertEqual(sent["jsonrpc"], "2.0")
        self.assertEqual(sent["id"], 1)
        self.assertEqual(sent["method"], "sampling/createMessage")
        self.assertEqual(sent["params"]["maxTokens"], 10)
        self.assertEqual(result, SAMPLE_RESULT)

    def test_request_ids_increment(self):
        stdin = io.StringIO(
            json.dumps({"jsonrpc": "2.0", "id": 1, "result": {}}) + "\n"
            + json.dumps({"jsonrpc": "2.0", "id": 2, "result": {}}) + "\n")
        stdout = io.StringIO()
        transport = StdioTransport(stdin, stdout)
        transport.request("sampling/createMessage", {})
        transport.request("sampling/createMessage", {})
        ids = [json.loads(line)["id"]
               for line in stdout.getvalue().splitlines()]
        self.assertEqual(ids, [1, 2])

    def test_method_not_found_is_not_supported(self):
        transport, _ = self._transport_with_reply(
            {"jsonrpc": "2.0", "id": 1,
             "error": {"code": -32601, "message": "Method not found"}})
        with self.assertRaises(SamplingNotSupportedError):
            transport.request("sampling/createMessage", {})

    def test_other_error_is_declined(self):
        transport, _ = self._transport_with_reply(
            {"jsonrpc": "2.0", "id": 1,
             "error": {"code": -32602, "message": "Sampling declined"}})
        with self.assertRaises(SamplingDeclinedError):
            transport.request("sampling/createMessage", {})

    def test_notification_before_response_is_queued_not_lost(self):
        lines = [
            {"jsonrpc": "2.0", "method": "notifications/progress",
             "params": {"progress": 1}},
            {"jsonrpc": "2.0", "id": 1, "result": SAMPLE_RESULT},
        ]
        stdin = io.StringIO("\n".join(json.dumps(l) for l in lines) + "\n")
        stdout = io.StringIO()
        transport = StdioTransport(stdin, stdout)
        result = transport.request("sampling/createMessage", {})
        self.assertEqual(result, SAMPLE_RESULT)
        self.assertEqual(len(transport.pending), 1)
        queued = json.loads(transport.pending[0])
        self.assertEqual(queued["method"], "notifications/progress")
        # The main loop reads queued lines first.
        self.assertEqual(json.loads(transport.next_line())["method"],
                         "notifications/progress")
        self.assertEqual(transport.pending, [])

    def test_eof_while_waiting_is_an_error(self):
        stdin = io.StringIO("")
        stdout = io.StringIO()
        transport = StdioTransport(stdin, stdout)
        with self.assertRaises(SamplingError):
            transport.request("sampling/createMessage", {})

    def test_invalid_json_line_is_an_error(self):
        stdin = io.StringIO("this is not json\n")
        stdout = io.StringIO()
        transport = StdioTransport(stdin, stdout)
        with self.assertRaises(SamplingError):
            transport.request("sampling/createMessage", {})


def _request(msg_id, method, params=None):
    msg = {"jsonrpc": "2.0", "id": msg_id, "method": method}
    if params is not None:
        msg["params"] = params
    return msg


class MCPServerSamplingTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "run.jsonl")
        self.server = MCPServer(self.path, task="sampling test",
                                sampler=Sampler(FakeTransport()))

    def tearDown(self):
        try:
            self.server.close()
        except Exception:
            pass
        self.tmp.cleanup()

    def _init(self):
        self.server.handle_message(
            _request(1, "initialize", {"protocolVersion": "2024-11-05"}))

    def test_sample_tool_lists(self):
        self._init()
        response = self.server.handle_message(_request(2, "tools/list"))
        names = [t["name"] for t in response["result"]["tools"]]
        self.assertIn("blackbox_sample", names)

    def test_sample_tool_returns_sampled_text(self):
        self._init()
        response = self.server.handle_message(_request(
            2, "tools/call",
            {"name": "blackbox_sample",
             "arguments": {"messages": _messages()}}))
        self.assertFalse(response["result"]["isError"])
        self.assertEqual(response["result"]["content"][0]["text"],
                         "The session looks clean.")

    def test_sample_tool_records_a_model_call(self):
        self._init()
        self.server.handle_message(_request(
            2, "tools/call",
            {"name": "blackbox_sample",
             "arguments": {"messages": _messages()}}))
        self.server.close()
        records = load_records(self.path)
        model_calls = [r for r in records if r["type"] == "model_call"]
        self.assertEqual(len(model_calls), 1)
        self.assertEqual(model_calls[0]["payload"]["model"], "test-model")
        self.assertEqual(model_calls[0]["payload"]["text"],
                         "The session looks clean.")
        self.assertEqual(model_calls[0]["payload"]["stop_reason"],
                         "endTurn")
        self.assertTrue(verify(self.path)["ok"])

    def test_sample_tool_without_sampler_is_an_error(self):
        server = MCPServer(os.path.join(self.tmp.name, "other.jsonl"),
                           sampler=None)
        try:
            server.handle_message(
                _request(1, "initialize",
                         {"protocolVersion": "2024-11-05"}))
            response = server.handle_message(_request(
                2, "tools/call",
                {"name": "blackbox_sample",
                 "arguments": {"messages": _messages()}}))
            self.assertTrue(response["result"]["isError"])
            self.assertIn("not supported",
                          response["result"]["content"][0]["text"])
        finally:
            server.close()

    def test_sample_tool_declined_is_an_error_not_a_crash(self):
        server = MCPServer(
            os.path.join(self.tmp.name, "declined.jsonl"),
            sampler=Sampler(FakeTransport(
                error=SamplingDeclinedError("client refused"))))
        try:
            server.handle_message(
                _request(1, "initialize",
                         {"protocolVersion": "2024-11-05"}))
            response = server.handle_message(_request(
                2, "tools/call",
                {"name": "blackbox_sample",
                 "arguments": {"messages": _messages()}}))
            self.assertTrue(response["result"]["isError"])
            self.assertIn("declined",
                          response["result"]["content"][0]["text"])
        finally:
            server.close()

    def test_sample_tool_missing_messages_is_an_error(self):
        self._init()
        response = self.server.handle_message(_request(
            2, "tools/call",
            {"name": "blackbox_sample", "arguments": {}}))
        self.assertEqual(response["error"]["code"], -32602)

    def test_sample_tool_accepts_wire_names(self):
        self._init()
        response = self.server.handle_message(_request(
            2, "tools/call",
            {"name": "blackbox_sample",
             "arguments": {"messages": _messages(),
                           "systemPrompt": "Be brief.",
                           "maxTokens": 64}}))
        self.assertFalse(response["result"]["isError"])
        transport = self.server.sampler._transport
        _, params = transport.requests[0]
        self.assertEqual(params["systemPrompt"], "Be brief.")
        self.assertEqual(params["maxTokens"], 64)


if __name__ == "__main__":
    unittest.main()
