"""MCP server wrapper: expose blackbox to any MCP client.

A minimal MCP server over stdio, speaking newline-delimited JSON-RPC
2.0 (the MCP stdio transport). No dependencies beyond the standard
library, like the rest of blackbox.

The server holds one Recorder open on the configured log and exposes
three tools:

    blackbox_record   append one step to the log
    blackbox_verify   check the log's hash chain
    blackbox_replay   read the log back as a transcript

It also exposes the session as two resources, and three prompts:

    blackbox://session        the session as a readable transcript
    blackbox://session.json   the session's records as JSON
    analyze_session           a prompt template for analysing the
                              session (optional focus argument)
    summarize_costs           a prompt template for summarising the
                              session's cost per model
    summarize_anomalies       a prompt template for summarising the
                              session's suspicious steps: error
                              results, repeated tool calls, unanswered
                              calls, redacted secrets

Start it from the CLI:

    python -m blackbox mcp data/run.jsonl

and point any MCP client at that command as a stdio server. The
client handshake is the standard one: `initialize`, then
`notifications/initialized`, then `tools/list` / `tools/call`.

Sampling is a client capability, so the server asks the client for
completions instead of calling a model itself: the `blackbox_sample`
tool sends a `sampling/createMessage` request to the connected
client and returns the sampled text. The prompt is redacted before
it is sent, and the sampled completion is recorded in the log as a
`model_call` step, like any other model call. If the client
declines or does not support sampling, the tool reports an error
instead of failing the server. The Sampler in `sampling.py` is the
same request over stdio, usable without the tool.
"""

import json
import re
import sys

from .costs import cost_for
from .recorder import Recorder
from .redact import REDACTED
from .replay import _short, format_record, load_records
from .sampling import (Sampler, SamplingDeclinedError, SamplingError,
                       SamplingNotSupportedError, StdioTransport)
from .verify import verify

PROTOCOL_VERSION = "2024-11-05"
SERVER_NAME = "blackbox"
SERVER_VERSION = "0.8.0"

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
    {
        "name": "blackbox_sample",
        "description": ("Ask the connected client for an LLM completion "
                        "(MCP sampling). The prompt is redacted before "
                        "it is sent, and the sampled text is recorded in "
                        "the log as a model_call step. Returns an error "
                        "if the client declines or does not support "
                        "sampling."),
        "inputSchema": {
            "type": "object",
            "properties": {
                "messages": {"type": "array",
                             "description": "Messages to sample from: "
                                            "each {\"role\": \"user\" | "
                                            "\"assistant\", \"content\": "
                                            "text or a text block}"},
                "systemPrompt": {"type": "string",
                                 "description": "Optional system prompt"},
                "modelPreferences": {"type": "object",
                                     "description": "Optional model "
                                                    "preferences: hints, "
                                                    "costPriority, "
                                                    "speedPriority, "
                                                    "intelligencePriority"},
                "maxTokens": {"type": "integer",
                              "description": "Maximum tokens to sample "
                                             "(default 1024)"},
                "includeContext": {"type": "string",
                                   "description": "Optional: none, "
                                                  "thisServer or "
                                                  "allServers"},
                "temperature": {"type": "number",
                                "description": "Optional sampling "
                                               "temperature"},
                "stopSequences": {"type": "array",
                                  "description": "Optional stop "
                                                 "sequences"},
            },
            "required": ["messages"],
        },
    },
]

# The session as readable resources. Both read from the same log the
# tools write to, so a client always sees the current session; the
# records are already redacted at write time, so nothing secret
# leaks through a resource read.
RESOURCES = [
    {
        "uri": "blackbox://session",
        "name": "session",
        "description": ("The recorded session as a readable transcript, "
                        "one step per line, like `blackbox replay`."),
        "mimeType": "text/plain",
    },
    {
        "uri": "blackbox://session.json",
        "name": "session.json",
        "description": ("The recorded session's raw records as a JSON "
                        "array, hash chain included."),
        "mimeType": "application/json",
    },
]

PROMPTS = [
    {
        "name": "analyze_session",
        "description": ("Analyse the recorded session: what the agent "
                        "did, in what order, where it went wrong, and "
                        "what it cost."),
        "arguments": [
            {
                "name": "focus",
                "description": ("What to focus on, e.g. errors, cost, "
                                "tool usage, or a step type like "
                                "model_call. Optional; defaults to the "
                                "whole session."),
                "required": False,
            },
        ],
    },
    {
        "name": "summarize_costs",
        "description": ("Summarise the recorded session's cost: tokens "
                        "and USD per model, and which calls look "
                        "wasteful."),
        "arguments": [],
    },
    {
        "name": "summarize_anomalies",
        "description": ("Summarise the recorded session's suspicious "
                        "steps: error results, repeated identical tool "
                        "calls, tool calls with no result, and steps "
                        "that had a secret redacted."),
        "arguments": [],
    },
]

PROMPT_FOCUS_LINES = {
    "errors": ("Pay special attention to steps that look like errors or "
               "failures: tool results reporting failure, retries, and "
               "any step whose payload mentions an error."),
    "cost": ("Pay special attention to cost: total tokens per model, "
             "which model calls were the most expensive, and whether "
             "any call looks wasteful."),
    "tool usage": ("Pay special attention to tool usage: which tools "
                   "were called, with what arguments, and whether any "
                   "call looks redundant or wrong."),
}

ANALYZE_PROMPT = (
    "Analyse this blackbox session: the recorded steps an AI agent "
    "took, in order, with timestamps, token counts and a hash-chained "
    "summary. Summarise what the agent was trying to do, list the "
    "steps it took in order, and call out anything that looks wrong, "
    "wasteful or surprising. {focus}\n\n"
    "Session transcript:\n{transcript}\n\n"
    "The session verifies as: {verification}"
)

SUMMARIZE_COSTS_PROMPT = (
    "Summarise the cost of this blackbox session: which models were "
    "called, how many tokens each used, what it cost in USD, and "
    "whether any call looks wasteful. Prices are the providers' "
    "published list prices and are estimates, not a bill. Calls with "
    "no input/output token split, or a model with no published "
    "price, are listed as unpriced, never guessed.\n\n"
    "Cost breakdown:\n{costs}\n\n"
    "Session transcript:\n{transcript}"
)

SUMMARIZE_ANOMALIES_PROMPT = (
    "Summarise the suspicious steps in this blackbox session: error "
    "results, repeated identical tool calls, tool calls with no "
    "result, and steps where a secret was redacted before writing. "
    "For each anomaly, say what happened, which step it was, and "
    "what the agent should have done differently. These findings "
    "are heuristics computed from the log, not a verdict — a "
    "repeated call may be a legitimate retry, and an error result "
    "may be expected. Say so where it matters.\n\n"
    "Anomalies found:\n{anomalies}\n\n"
    "Session transcript:\n{transcript}"
)


class MCPServer:
    """A blackbox MCP server bound to one log file.

    `sampler` is an optional Sampler for MCP sampling: if given,
    the `blackbox_sample` tool (and `sample()`) ask the connected
    client for completions through it. When the server runs on
    stdio via serve(), a Sampler over the same stdio streams is
    attached automatically if none was given.
    """

    def __init__(self, log_path, task=None, model=None, sampler=None):
        self.log_path = log_path
        self.recorder = Recorder(log_path, task=task, model=model)
        self.sampler = sampler
        self._transport = None
        self._initialized = False

    def sample(self, messages, **kwargs):
        """Ask the connected client for one completion.

        Raises SamplingNotSupportedError when no sampler is
        attached (the server is not connected to a client that can
        sample), and the Sampler's other errors when the client
        declines or answers badly.
        """
        if self.sampler is None:
            raise SamplingNotSupportedError(
                "no sampling client attached: start the server with "
                "serve() over stdio, or pass a Sampler")
        return self.sampler.sample(messages, **kwargs)

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
        if method == "resources/list":
            return _result(msg_id, {"resources": RESOURCES})
        if method == "resources/read":
            return self._resources_read(msg_id, message.get("params") or {})
        if method == "prompts/list":
            return _result(msg_id, {"prompts": PROMPTS})
        if method == "prompts/get":
            return self._prompts_get(msg_id, message.get("params") or {})
        return _error(msg_id, METHOD_NOT_FOUND,
                      "unknown method: %s" % method)

    def _initialize(self, msg_id, params):
        self._initialized = True
        return _result(msg_id, {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {"tools": {}, "resources": {}, "prompts": {}},
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
        if name == "blackbox_sample":
            return self._tool_sample(msg_id, arguments)
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
        return self._tool_result(msg_id, self._transcript())

    def _tool_sample(self, msg_id, arguments):
        messages = arguments.get("messages")
        if messages is None:
            return _error(msg_id, INVALID_PARAMS,
                          "blackbox_sample needs 'messages'")
        kwargs = {}
        for key in ("model_preferences", "system_prompt",
                    "include_context", "temperature", "max_tokens",
                    "stop_sequences"):
            # Tool arguments use the MCP wire names (camelCase);
            # Sampler.sample() takes snake_case.
            wire = {"model_preferences": "modelPreferences",
                    "system_prompt": "systemPrompt",
                    "include_context": "includeContext",
                    "temperature": "temperature",
                    "max_tokens": "maxTokens",
                    "stop_sequences": "stopSequences"}[key]
            if wire in arguments:
                kwargs[key] = arguments[wire]
        try:
            result = self.sample(messages, **kwargs)
        except SamplingNotSupportedError as exc:
            return self._tool_result(msg_id, "sampling not supported: %s"
                                     % exc, is_error=True)
        except SamplingDeclinedError as exc:
            return self._tool_result(msg_id, "sampling declined: %s" % exc,
                                     is_error=True)
        except SamplingError as exc:
            return self._tool_result(msg_id, "sampling failed: %s" % exc,
                                     is_error=True)
        except ValueError as exc:
            return _error(msg_id, INVALID_PARAMS, str(exc))
        # A sampled completion is a model call in this session: record
        # it like any other, so the flight record stays complete. The
        # recorder redacts before writing, as always.
        self.recorder.record("model_call", {
            "model": result.model or "unknown",
            "text": result.text,
            "stop_reason": result.stop_reason,
        })
        return self._tool_result(msg_id, result.text)

    def _transcript(self):
        records = load_records(self.log_path)
        return "\n".join(format_record(r) for r in records)

    def _cost_lines(self):
        """Per-model token usage and USD cost, as readable lines.

        Computed from the recorded model_call records, with the same
        rules as the Recorder's summary: a plain int token count has
        no input/output split and is not priced, and an unknown model
        costs None, never a guessed number.
        """
        usage = {}
        for record in load_records(self.log_path):
            if record.get("type") != "model_call":
                continue
            model = ((record.get("payload") or {}).get("model")
                     or self.recorder.model)
            if not model:
                continue
            entry = usage.setdefault(
                model, {"input_tokens": 0, "output_tokens": 0,
                        "tokens": 0})
            tokens = record.get("tokens")
            if isinstance(tokens, dict):
                if isinstance(tokens.get("input"), int):
                    entry["input_tokens"] += tokens["input"]
                if isinstance(tokens.get("output"), int):
                    entry["output_tokens"] += tokens["output"]
            elif isinstance(tokens, int):
                entry["tokens"] += tokens
        if not usage:
            return "No model calls recorded yet."
        lines = []
        total = 0.0
        priced_any = False
        for model, entry in sorted(usage.items()):
            parts = []
            if entry["input_tokens"] or entry["output_tokens"]:
                parts.append("%d input + %d output tokens"
                             % (entry["input_tokens"],
                                entry["output_tokens"]))
            if entry["tokens"]:
                parts.append("%d tokens with no input/output split"
                             % entry["tokens"])
            cost = None
            if entry["input_tokens"] or entry["output_tokens"]:
                cost = cost_for(model, entry["input_tokens"],
                                entry["output_tokens"])
            if cost is not None:
                parts.append("$%s" % ("%.6f" % cost).rstrip("0")
                             .rstrip("."))
                total += cost
                priced_any = True
            elif entry["input_tokens"] or entry["output_tokens"]:
                parts.append("cost unknown: no published price "
                             "for this model")
            else:
                parts.append("not priced: no input/output split")
            lines.append("%s: %s" % (model, ", ".join(parts)))
        if priced_any:
            lines.append("Total priced cost: $%s"
                         % ("%.6f" % total).rstrip("0").rstrip("."))
        return "\n".join(lines)

    def _anomaly_lines(self):
        """Suspicious steps in the session, as readable lines.

        Four heuristics, computed from the log (the summary record
        is skipped — it is bookkeeping, not agent behaviour):
        tool results that look like errors, the same tool called
        with identical arguments more than once, tool calls with no
        matching result, and steps that had a secret redacted before
        writing. A repeated call may be a legitimate retry, so the
        prompt says to treat these as leads, not verdicts.
        """
        records = [r for r in load_records(self.log_path)
                   if r.get("type") != "summary"]
        findings = []
        for record in records:
            if record.get("type") != "tool_result":
                continue
            payload = record.get("payload") or {}
            result = payload.get("result", "")
            if "error" in payload:
                findings.append(
                    "step %d: %s returned a result with an 'error' "
                    "field" % (record["seq"], payload.get("tool", "?")))
            elif (isinstance(result, str)
                    and re.search(r"error|fail", result, re.IGNORECASE)):
                findings.append(
                    "step %d: %s returned an error result: %s"
                    % (record["seq"], payload.get("tool", "?"),
                       _short(result)))
        calls = {}
        for record in records:
            if record.get("type") != "tool_call":
                continue
            payload = record.get("payload") or {}
            key = json.dumps({"tool": payload.get("tool"),
                              "args": payload.get("args", {})},
                             sort_keys=True)
            calls.setdefault(key, []).append(record["seq"])
        for key, seqs in calls.items():
            if len(seqs) > 1:
                signature = json.loads(key)
                findings.append(
                    "tool %s called %d times with identical arguments "
                    "%s (steps %s) — possible retry loop"
                    % (signature.get("tool"), len(seqs),
                       json.dumps(signature.get("args"), sort_keys=True),
                       ", ".join(str(s) for s in seqs)))
        pending = {}
        for record in records:
            payload = record.get("payload") or {}
            tool = payload.get("tool")
            if record.get("type") == "tool_call":
                pending.setdefault(tool, []).append(record["seq"])
            elif record.get("type") == "tool_result":
                queue = pending.get(tool)
                if queue:
                    queue.pop(0)
                    if not queue:
                        del pending[tool]
        for tool in sorted(pending, key=lambda t: pending[t][0]):
            for seq in pending[tool]:
                findings.append(
                    "step %d: tool_call %s has no matching tool_result"
                    % (seq, tool))
        for record in records:
            payload = record.get("payload") or {}
            if REDACTED in json.dumps(payload):
                findings.append(
                    "step %d: %s contained a secret that was redacted "
                    "before writing" % (record["seq"],
                                        record.get("type")))
        if not findings:
            return ("No anomalies found: no error results, no repeated "
                    "tool calls, no unanswered tool calls, and no "
                    "redacted secrets.")
        return "\n".join(findings)

    def _resources_read(self, msg_id, params):
        uri = params.get("uri")
        if uri == "blackbox://session":
            text = self._transcript()
            return _result(msg_id, {"contents": [
                {"uri": uri, "mimeType": "text/plain", "text": text}]})
        if uri == "blackbox://session.json":
            records = load_records(self.log_path)
            text = json.dumps(records, indent=2, sort_keys=True)
            return _result(msg_id, {"contents": [
                {"uri": uri, "mimeType": "application/json",
                 "text": text}]})
        return _error(msg_id, INVALID_PARAMS,
                      "unknown resource: %r" % (uri,))

    def _prompts_get(self, msg_id, params):
        name = params.get("name")
        arguments = params.get("arguments") or {}
        if not isinstance(arguments, dict):
            return _error(msg_id, INVALID_PARAMS,
                          "arguments must be an object")
        description = next((p["description"] for p in PROMPTS
                            if p["name"] == name), None)
        if description is None:
            return _error(msg_id, INVALID_PARAMS,
                          "unknown prompt: %r" % (name,))
        if name == "summarize_costs":
            text = SUMMARIZE_COSTS_PROMPT.format(
                costs=self._cost_lines(),
                transcript=self._transcript())
            return _result(msg_id, {
                "description": description,
                "messages": [{"role": "user",
                              "content": {"type": "text", "text": text}}],
            })
        if name == "summarize_anomalies":
            text = SUMMARIZE_ANOMALIES_PROMPT.format(
                anomalies=self._anomaly_lines(),
                transcript=self._transcript())
            return _result(msg_id, {
                "description": description,
                "messages": [{"role": "user",
                              "content": {"type": "text", "text": text}}],
            })
        focus = arguments.get("focus")
        if focus is not None and not isinstance(focus, str):
            return _error(msg_id, INVALID_PARAMS,
                          "focus must be a string")
        focus_line = ""
        if focus:
            focus_line = PROMPT_FOCUS_LINES.get(
                focus, "Focus on: %s." % focus)
        result = verify(self.log_path)
        if result["ok"]:
            verification = ("OK: %d records, hash chain intact"
                            % result["records"])
        else:
            verification = "FAIL: %s" % result["error"]
        text = ANALYZE_PROMPT.format(
            focus=focus_line, transcript=self._transcript(),
            verification=verification)
        return _result(msg_id, {
            "description": description,
            "messages": [{"role": "user",
                          "content": {"type": "text", "text": text}}],
        })

    def _read_line(self, stdin):
        """The next input line. When a stdio sampling transport is
        attached, read through it: it drains lines it queued while
        waiting for a sampling response first, so a sampling request
        issued mid-session does not swallow the client's messages."""
        if self._transport is not None:
            return self._transport.next_line()
        return stdin.readline()

    def serve(self, stdin=None, stdout=None):
        """Read JSON-RPC lines from stdin, write responses to stdout.

        Runs until EOF (or Ctrl-C / a broken pipe), then closes the
        recorder, which writes the summary record. A Sampler over
        the same stdio streams is attached automatically if none
        was given, so the blackbox_sample tool can ask the client
        for completions.
        """
        stdin = stdin if stdin is not None else sys.stdin
        stdout = stdout if stdout is not None else sys.stdout
        if self.sampler is None:
            self._transport = StdioTransport(stdin, stdout)
            self.sampler = Sampler(self._transport)
        try:
            while True:
                line = self._read_line(stdin)
                if not line:
                    break
                if not line.strip():
                    continue
                try:
                    message = json.loads(line)
                except json.JSONDecodeError:
                    response = _error(None, PARSE_ERROR,
                                      "line is not valid JSON")
                else:
                    if isinstance(message, dict) \
                            and "method" not in message:
                        # A response to a server-initiated request
                        # (e.g. a sampling response answered out of
                        # order and queued by the transport): not a
                        # request, nothing to answer.
                        continue
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
