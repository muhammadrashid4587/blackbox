"""Demo MCP sampling with a fake client — no real model needed.

A fake client answers sampling/createMessage with a canned
completion, so this shows the whole flow: Sampler.sample() with
redaction of a secret in the prompt, the decline fallback, and the
blackbox_sample tool on the MCP server recording the sampled
completion as a model_call step.
"""

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from blackbox.mcp_server import MCPServer
from blackbox.replay import load_records
from blackbox.sampling import (Sampler, SamplingDeclinedError,
                               SamplingNotSupportedError)
from blackbox.verify import verify


class FakeClient:
    """A pretend MCP client: answers sampling with a canned reply,
    and checks the prompt it was sent was already redacted."""

    def __init__(self, decline=False):
        self.decline = decline
        self.seen_params = None

    def request(self, method, params):
        assert method == "sampling/createMessage", method
        self.seen_params = params
        if self.decline:
            raise SamplingDeclinedError("the client said no")
        # The prompt must arrive redacted: the fake API key in the
        # message must not reach the client.
        text = json.dumps(params)
        assert "sk-abcdefghijklmnopqrst" not in text, "secret leaked!"
        assert "[REDACTED]" in text, "nothing was redacted"
        return {
            "model": "demo-model",
            "stopReason": "endTurn",
            "role": "assistant",
            "content": {"type": "text",
                        "text": "The session has 3 tool calls, no errors."},
        }


def main():
    print("== Sampler with a fake client ==")
    sampler = Sampler(FakeClient())
    result = sampler.sample(
        [{"role": "user",
          "content": "Summarise this run. My key is sk-abcdefghijklmnopqrst."}],
        system_prompt="Be brief.",
        max_tokens=256)
    print("sampled text: %s" % result.text)
    print("model: %s, stop reason: %s" % (result.model, result.stop_reason))

    print("\n== Decline fallback: sample_or_none returns None ==")
    declining = Sampler(FakeClient(decline=True))
    print("sample_or_none: %r" % declining.sample_or_none(
        [{"role": "user", "content": "Summarise this run."}]))
    try:
        declining.sample([{"role": "user", "content": "Summarise this run."}])
    except SamplingNotSupportedError:
        print("sample() raised SamplingNotSupportedError")
    except SamplingDeclinedError as exc:
        print("sample() raised SamplingDeclinedError: %s" % exc)

    print("\n== blackbox_sample tool on the MCP server ==")
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "demo.jsonl")
        server = MCPServer(path, task="sampling demo",
                           sampler=Sampler(FakeClient()))
        server.handle_message({"jsonrpc": "2.0", "id": 1,
                               "method": "initialize", "params": {}})
        response = server.handle_message({
            "jsonrpc": "2.0", "id": 2, "method": "tools/call",
            "params": {"name": "blackbox_sample",
                       "arguments": {"messages": [
                           {"role": "user",
                            "content": "Summarise this session. My key is "
                                       "sk-abcdefghijklmnopqrst."}]}}})
        print("tool result: %s"
              % response["result"]["content"][0]["text"])
        server.close()
        records = load_records(path)
        print("log: %d records, verifies: %s"
              % (len(records), verify(path)["ok"]))
        print("recorded model_call: %s"
              % json.dumps(records[0]["payload"], sort_keys=True))


if __name__ == "__main__":
    main()
