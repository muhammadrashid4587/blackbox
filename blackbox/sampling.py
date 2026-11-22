"""MCP sampling: ask the connected client for an LLM completion.

In MCP, sampling is a client capability, not a server one: the
server sends a `sampling/createMessage` request and the client —
which owns the model connection — answers with the sampled text.
So blackbox's server can ask the client to summarise a session, or
to judge a step, without ever holding a model API key itself.

Two pieces:

    Sampler         the clean API: `sample()` sends one request and
                    returns a SamplingResult (text, model, stop
                    reason). It validates its arguments, redacts
                    secrets from the prompt before sending, and
                    raises a clear error when the client declines or
                    does not support sampling. `sample_or_none()` is
                    the same call with a None fallback instead of
                    an exception.
    StdioTransport  the JSON-RPC transport for a stdio server: it
                    writes one request line, then reads lines until
                    the response with the matching id arrives. Any
                    other line in between (a notification, or a
                    request from the client) is queued in `pending`
                    for the server's main loop to handle, so a
                    sampling request issued mid-session does not
                    swallow the client's messages.

A transport is anything with a `request(method, params)` method
that returns the result dict of the JSON-RPC response and raises a
SamplingError on a JSON-RPC error. Tests use a fake transport, so
no real model is needed.
"""

import json

from .redact import redact

SAMPLING_METHOD = "sampling/createMessage"

# JSON-RPC error code for "method not found": the client telling us
# it has no sampling capability at all.
METHOD_NOT_FOUND = -32601

INCLUDE_CONTEXT_VALUES = ("none", "thisServer", "allServers")
MESSAGE_ROLES = ("user", "assistant")


class SamplingError(Exception):
    """Something went wrong asking the client for a completion."""


class SamplingNotSupportedError(SamplingError):
    """The client does not support sampling (no sampling capability,
    or it answered `sampling/createMessage` with method-not-found)."""


class SamplingDeclinedError(SamplingError):
    """The client refused this sampling request, or answered with a
    JSON-RPC error that is not method-not-found."""


class SamplingResult:
    """One sampled completion from the client."""

    def __init__(self, text, model=None, stop_reason=None, role="assistant"):
        self.text = text
        self.model = model
        self.stop_reason = stop_reason
        self.role = role

    def __repr__(self):
        return ("SamplingResult(model=%r, stop_reason=%r, text=%r)"
                % (self.model, self.stop_reason, self.text))


def _validate_messages(messages):
    """Check the message list and normalise string content to a text
    block, as the sampling spec expects. Raises ValueError."""
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a non-empty list")
    normalised = []
    for message in messages:
        if not isinstance(message, dict):
            raise ValueError("each message must be an object")
        role = message.get("role")
        if role not in MESSAGE_ROLES:
            raise ValueError("message role must be 'user' or 'assistant', "
                             "got %r" % (role,))
        content = message.get("content")
        if isinstance(content, str):
            content = {"type": "text", "text": content}
        elif (isinstance(content, dict)
                and content.get("type") == "text"
                and isinstance(content.get("text"), str)):
            content = {"type": "text", "text": content["text"]}
        else:
            raise ValueError(
                "message content must be a string or a text block, "
                "got %r" % (content,))
        normalised.append({"role": role, "content": content})
    return normalised


def _validate_model_preferences(preferences):
    """Check a modelPreferences object. Raises ValueError."""
    if preferences is None:
        return None
    if not isinstance(preferences, dict):
        raise ValueError("modelPreferences must be an object")
    for key in ("costPriority", "speedPriority", "intelligencePriority"):
        value = preferences.get(key)
        if value is not None and not (
                isinstance(value, (int, float)) and 0 <= value <= 1):
            raise ValueError("%s must be a number between 0 and 1, "
                             "got %r" % (key, value))
    hints = preferences.get("hints")
    if hints is not None:
        if (not isinstance(hints, list)
                or any(not isinstance(h, dict)
                       or not isinstance(h.get("name"), str)
                       for h in hints)):
            raise ValueError("modelPreferences.hints must be a list of "
                             "objects with a string 'name'")
    return preferences


class Sampler:
    """Ask the connected MCP client for LLM completions.

    `transport` is any object with a `request(method, params)`
    method returning the result dict of the JSON-RPC response, and
    raising SamplingError on a JSON-RPC error. Use StdioTransport
    for a real stdio client, or a fake in tests.
    """

    def __init__(self, transport):
        self._transport = transport

    def sample(self, messages, model_preferences=None, system_prompt=None,
               include_context=None, temperature=None, max_tokens=1024,
               stop_sequences=None):
        """Request one completion and return a SamplingResult.

        `messages` is a list of {"role": "user"|"assistant",
        "content": str | {"type": "text", "text": str}}. The prompt
        is redacted before it is sent, so a secret in the messages
        or the system prompt never reaches the client.

        Raises SamplingNotSupportedError if the client has no
        sampling capability, SamplingDeclinedError if it refuses
        the request, SamplingError on a bad or non-text result, and
        ValueError on invalid arguments (before anything is sent).
        """
        messages = _validate_messages(messages)
        model_preferences = _validate_model_preferences(model_preferences)
        if system_prompt is not None and not isinstance(system_prompt, str):
            raise ValueError("system_prompt must be a string")
        if include_context is not None \
                and include_context not in INCLUDE_CONTEXT_VALUES:
            raise ValueError("include_context must be one of %s, got %r"
                             % (", ".join(INCLUDE_CONTEXT_VALUES),
                                include_context))
        if temperature is not None and not (
                isinstance(temperature, (int, float)) and temperature >= 0):
            raise ValueError("temperature must be a number >= 0, got %r"
                             % (temperature,))
        if not isinstance(max_tokens, int) or isinstance(max_tokens, bool) \
                or max_tokens <= 0:
            raise ValueError("max_tokens must be a positive int, got %r"
                             % (max_tokens,))
        if stop_sequences is not None and (
                not isinstance(stop_sequences, list)
                or any(not isinstance(s, str) for s in stop_sequences)):
            raise ValueError("stop_sequences must be a list of strings")
        params = {"messages": messages, "maxTokens": max_tokens}
        if model_preferences is not None:
            params["modelPreferences"] = model_preferences
        if system_prompt is not None:
            params["systemPrompt"] = system_prompt
        if include_context is not None:
            params["includeContext"] = include_context
        if temperature is not None:
            params["temperature"] = temperature
        if stop_sequences is not None:
            params["stopSequences"] = stop_sequences
        # Redact before sending: a secret in the prompt must never
        # reach the client, just as it never reaches the log.
        params = redact(params)
        result = self._transport.request(SAMPLING_METHOD, params)
        return _parse_result(result)

    def sample_or_none(self, messages, **kwargs):
        """Like sample(), but returns None instead of raising when
        the client declines, does not support sampling, or answers
        badly. Invalid arguments still raise ValueError — that is a
        caller bug, not a client refusal."""
        try:
            return self.sample(messages, **kwargs)
        except SamplingError:
            return None


def _parse_result(result):
    """Turn a sampling/createMessage result into a SamplingResult.
    Raises SamplingError on a bad or non-text result."""
    if not isinstance(result, dict):
        raise SamplingError("sampling result must be an object, got %r"
                            % (result,))
    content = result.get("content")
    if not isinstance(content, dict) or content.get("type") != "text" \
            or not isinstance(content.get("text"), str):
        raise SamplingError(
            "sampling result has no text content: %r" % (result,))
    return SamplingResult(
        text=content["text"],
        model=result.get("model"),
        stop_reason=result.get("stopReason"),
        role=result.get("role", "assistant"))


class StdioTransport:
    """A JSON-RPC transport over newline-delimited stdio.

    `request()` writes one request line and reads lines until the
    response with the matching id arrives. Lines that are not that
    response — notifications, or requests from the client — are
    queued in `pending`, in order, for the caller to handle; the
    server's main loop reads through `next_line()`, which drains
    the queue first, so nothing is lost.
    """

    def __init__(self, stdin, stdout):
        self._stdin = stdin
        self._stdout = stdout
        self._next_id = 1
        self.pending = []

    def next_line(self):
        """The next input line: queued lines first, then stdin."""
        if self.pending:
            return self.pending.pop(0)
        return self._stdin.readline()

    def request(self, method, params):
        """Send one request and return its result dict.

        Raises SamplingNotSupportedError on a method-not-found
        error, SamplingDeclinedError on any other JSON-RPC error,
        and SamplingError if the client closes the connection or
        sends a line that is not valid JSON.
        """
        msg_id = self._next_id
        self._next_id += 1
        request = {"jsonrpc": "2.0", "id": msg_id, "method": method,
                   "params": params}
        self._stdout.write(json.dumps(request) + "\n")
        self._stdout.flush()
        # A response to this id may already be queued: an earlier
        # request can read it while waiting for a different id.
        for line in list(self.pending):
            message = json.loads(line)
            if isinstance(message, dict) and message.get("id") == msg_id:
                self.pending.remove(line)
                return self._handle_response(message)
        while True:
            # Read fresh input here, not next_line(): a line queued
            # into pending is not our response, and next_line()
            # would hand it straight back, looping forever.
            line = self._stdin.readline()
            if not line:
                raise SamplingError(
                    "client closed the connection while waiting for a "
                    "sampling response")
            try:
                message = json.loads(line)
            except json.JSONDecodeError:
                raise SamplingError(
                    "client sent a line that is not valid JSON: %r"
                    % (line.strip(),))
            if not isinstance(message, dict) or "id" not in message:
                # A notification, or a request from the client: not
                # our response. Queue it for the main loop.
                self.pending.append(line)
                continue
            if message.get("id") != msg_id:
                # A response to some other request: queue it too.
                self.pending.append(line)
                continue
            return self._handle_response(message)

    def _handle_response(self, message):
        """Turn a matching JSON-RPC response into its result dict,
        or raise the right SamplingError for an error response."""
        if "error" in message:
            error = message["error"] or {}
            code = error.get("code")
            text = error.get("message", "unknown error")
            if code == METHOD_NOT_FOUND:
                raise SamplingNotSupportedError(
                    "client does not support sampling: %s" % text)
            raise SamplingDeclinedError(
                "client declined the sampling request: %s" % text)
        if "result" not in message:
            raise SamplingError(
                "sampling response has neither result nor error: %r"
                % (message,))
        return message["result"]
