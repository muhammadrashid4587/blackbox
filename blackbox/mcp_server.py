"""MCP server wrapper: expose blackbox to any MCP client.

A minimal MCP server over stdio, speaking newline-delimited JSON-RPC
2.0 (the MCP stdio transport). No dependencies beyond the standard
library, like the rest of blackbox.

The server holds one Recorder open on the configured log and exposes
three tools:

    blackbox_record   append one step to the log
    blackbox_verify   check the log's hash chain
    blackbox_replay   read the log back as a transcript

Start it from the CLI:

    python -m blackbox mcp data/run.jsonl

and point any MCP client at that command as a stdio server. The
client handshake is the standard one: `initialize`, then
`notifications/initialized`, then `tools/list` / `tools/call`.

What this is not (yet): no resources (`cinder://`-style URIs), no
prompts, no sampling. Those are the next slices.
"""

import json
import sys

from .recorder import Recorder
from .replay import format_record, load_records
from .verify import verify

PROTOCOL_VERSION = "2024-11-05"
SERVER_NAME = "blackbox"
SERVER_VERSION = "0.4.0"

# JSON-RPC error codes.
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603


def _result(msg_id, result):
    return {"jsonrpc": "2.0", "id": msg_id, "result": result}


def _error(msg_id, code, message):
    return {"jsonrpc": "2.0", "id": msg_id,
            "error": {"code": code, "message": message}}


TOOLS = [
    {
        "name": "blackbox_record",
        "description": ("Append one agent step to the blackbox log. "
                        "The payload is redacted before it is hashed or "
                        "written, like every record."),
        "inputSchema": {
            "type": "object",
            "properties": {
                "type": {"type": "string",
                         "description": "Step type, e.g. model_call, "
                                        "tool_call, tool_result"},
                "payload": {"type": "object",
                            "description": "Step data"},
                "tokens": {"description": "Token count (int) or "
                                          "input/output split "
                                          "({\"input\": n, \"output\": n})"},
            },
            "required": ["type"],
        },
    },
    {
        "name": "blackbox_verify",
        "description": ("Check the log's hash chain and report whether "
                        "it is intact, or the first broken record."),
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "blackbox_replay",
        "description": "Read the log back as a readable transcript.",
        "inputSchema": {"type": "object", "properties": {}},
    },
]


class MCPServer:
    """A blackbox MCP server bound to one log file."""

    def __init__(self, log_path, task=None, model=None):
        self.log_path = log_path
        self.recorder = Recorder(log_path, task=task, model=model)
        self._initialized = False

    def handle_message(self, message):
        """Handle one decoded JSON-RPC message.

        Returns a response dict, or None for notifications (messages
        with no id), which get no reply.
        """
        if not isinstance(message, dict):
            return _error(None, INVALID_REQUEST,
                          "message must be a JSON object")
        method = message.get("method")
        msg_id = message.get("id")
        is_notification = "id" not in message
        if not isinstance(method, str):
            return _error(msg_id, INVALID_REQUEST,
                          "message has no method")
        if method.startswith("notifications/"):
            # Notifications are fire-and-forget; no reply ever.
            return None
        if method == "initialize":
            return self._initialize(msg_id, message.get("params") or {})
        if not self._initialized:
            return _error(msg_id, INVALID_REQUEST,
                          "client must initialize first")
        if method == "ping":
            return _result(msg_id, {})
        if method == "tools/list":
            return _result(msg_id, {"tools": TOOLS})
        if method == "tools/call":
            return self._tools_call(msg_id, message.get("params") or {})
        return _error(msg_id, METHOD_NOT_FOUND,
                      "unknown method: %s" % method)

    def _initialize(self, msg_id, params):
        self._initialized = True
        return _result(msg_id, {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {"tools": {}},
            "serverInfo": {"name": SERVER_NAME,
                           "version": SERVER_VERSION},
        })

    def _tools_call(self, msg_id, params):
        name = params.get("name")
        arguments = params.get("arguments") or {}
        if not isinstance(arguments, dict):
            return _error(msg_id, INVALID_PARAMS,
                          "arguments must be an object")
        if name == "blackbox_record":
            return self._tool_record(msg_id, arguments)
        if name == "blackbox_verify":
            return self._tool_verify(msg_id)
        if name == "blackbox_replay":
            return self._tool_replay(msg_id)
        return _error(msg_id, INVALID_PARAMS, "unknown tool: %r" % (name,))

    def _tool_result(self, msg_id, text, is_error=False):
        return _result(msg_id, {
            "content": [{"type": "text", "text": text}],
            "isError": is_error,
        })

    def _tool_record(self, msg_id, arguments):
        event_type = arguments.get("type")
        if not isinstance(event_type, str) or not event_type:
            return _error(msg_id, INVALID_PARAMS,
                          "blackbox_record needs a string 'type'")
        payload = arguments.get("payload") or {}
        if not isinstance(payload, dict):
            return _error(msg_id, INVALID_PARAMS,
                          "payload must be an object")
        tokens = arguments.get("tokens")
        if tokens is not None and not isinstance(tokens, (int, dict)):
            return _error(msg_id, INVALID_PARAMS,
                          "tokens must be an int or an input/output object")
        seq = self.recorder.record(event_type, payload, tokens=tokens)
        return self._tool_result(
            msg_id, "recorded step %d (%s)" % (seq, event_type))

    def _tool_verify(self, msg_id):
        result = verify(self.log_path)
        if result["ok"]:
            text = "OK: %d records, hash chain intact" % result["records"]
        else:
            text = "FAIL: %s" % result["error"]
        return self._tool_result(msg_id, text, is_error=not result["ok"])

    def _tool_replay(self, msg_id):
        records = load_records(self.log_path)
        text = "\n".join(format_record(r) for r in records)
        return self._tool_result(msg_id, text)

    def serve(self, stdin=None, stdout=None):
        """Read JSON-RPC lines from stdin, write responses to stdout.

        Runs until EOF (or Ctrl-C / a broken pipe), then closes the
        recorder, which writes the summary record.
        """
        stdin = stdin if stdin is not None else sys.stdin
        stdout = stdout if stdout is not None else sys.stdout
        try:
            for line in stdin:
                if not line.strip():
                    continue
                try:
                    message = json.loads(line)
                except json.JSONDecodeError:
                    response = _error(None, PARSE_ERROR,
                                      "line is not valid JSON")
                else:
                    try:
                        response = self.handle_message(message)
                    except Exception as exc:  # keep the server alive
                        response = _error(message.get("id")
                                          if isinstance(message, dict)
                                          else None,
                                          INTERNAL_ERROR, str(exc))
                if response is not None:
                    stdout.write(json.dumps(response) + "\n")
                    stdout.flush()
        finally:
            self.close()

    def close(self):
        self.recorder.close()


def serve_stdio(log_path, task=None, model=None):
    """Run the MCP server on stdio for one log file."""
    MCPServer(log_path, task=task, model=model).serve()
