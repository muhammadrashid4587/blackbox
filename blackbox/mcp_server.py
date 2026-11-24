"""MCP server wrapper: expose blackbox to any MCP client.

A minimal MCP server over stdio, speaking newline-delimited JSON-RPC
2.0 (the MCP stdio transport). No dependencies beyond the standard
library, like the rest of blackbox.

The server holds one Recorder open on the configured log and exposes
three tools:

    blackbox_record   append one step to the log
    blackbox_verify   check the log's hash chain
    blackbox_replay   read the log back as a transcript

It also exposes the session as five resources, six resource
templates, and eight prompts:

    blackbox://session        the session as a readable transcript
    blackbox://session.json   the session's records as JSON
    blackbox://session/aggregates
                              the session's totals as JSON, the same
                              totals `blackbox aggregates` prints
    blackbox://session/trace  the session's reconstructed causal
                              trace as JSON, the same trace
                              `blackbox trace` prints
    blackbox://session/diff?other_path=<path>
                              what changed between this session and
                              another run as JSON, the same diff
                              `blackbox diff` prints
    blackbox://session/diff/{other_path}
                              the same diff as a resource template:
                              a client lists it with
                              resources/templates/list and reads it
                              by expanding the template with the
                              other run's log path, so the other
                              run is named in the URI itself instead
                              of a query parameter
    blackbox://session/trace/{seq}
                              one branch of the session's causal
                              trace as a resource template: the
                              subtree rooted at step seq, the same
                              node the blackbox://session/trace
                              resource carries for that step, so a
                              client can read one branch without
                              the whole tree
    blackbox://session/aggregates/model/{model}
                              one model's totals as a resource
                              template: calls, input/output tokens
                              and USD cost for that model — the
                              same entry the blackbox costs report
                              carries for that model, so a client
                              can read one model's spend without
                              the whole breakdown
    blackbox://session/aggregates/tool/{tool}
                              one tool's totals as a resource
                              template: calls, the input/output
                              tokens of the model calls that
                              invoked it, and their USD cost — the
                              same entry the blackbox costs report
                              carries for that tool, so a client
                              can read one tool's cost without the
                              whole breakdown
    blackbox://session/record/{seq}
                              one step's raw record as a resource
                              template: the record for step seq as
                              JSON — the same record the
                              blackbox://session.json resource
                              carries for that step, so a client can
                              read one step without the whole log
    blackbox://session/transcript/{seq}
                              one step's formatted transcript line
                              as a resource template: the line the
                              blackbox://session resource carries
                              for step seq — the same line
                              `blackbox replay` prints for that
                              step — so a client can read one step
                              without the whole transcript
    blackbox://session/span/{seq}
                              one step as a single OTLP span as a
                              resource template: the span for step
                              seq — the step kind as the span name,
                              the step number as the span id,
                              start/end times from the record's
                              timestamps, and model/tool/token/cost
                              attributes where the record has them
                              (an unpriced call carries no cost,
                              never a guessed number) — so a client
                              can read one step as a span without
                              exporting the whole trace
    analyze_session           a prompt template for analysing the
                              session (optional focus argument)
    summarize_costs           a prompt template for summarising the
                              session's cost per model
    summarize_anomalies       a prompt template for summarising the
                              session's suspicious steps: error
                              results, repeated tool calls, unanswered
                              calls, redacted secrets
    summarize_sampling        a prompt template for summarising the
                              session's tail-sampling decisions: kept
                              vs sampled-out spans, which triggers
                              fired, parent gaps, and a per-model /
                              per-type breakdown of the sampled-out
                              spans
    summarize_trace           a prompt template for summarising the
                              session's reconstructed causal trace:
                              step count, span-type breakdown, the
                              slowest steps, error steps, parent gaps,
                              and per-tool / per-model cost and token
                              totals
    summarize_diff            a prompt template for summarising what
                              changed between this session and another
                              run: changed, added and removed steps,
                              per-model token and cost deltas, and
                              new vs fixed errors
    summarize_retention       a prompt template for summarising the
                              session's retention state: what earlier
                              compactions removed, how much history
                              is still in the log, and what the
                              default retention rules would remove
                              next
    summarize_aggregates      a prompt template for summarising the
                              session at a glance: step count and
                              duration, per-type and per-tool counts,
                              per-model tokens and cost, error
                              steps, the slowest step, and how many
                              records an earlier compaction removed
                              or sampling dropped

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
from datetime import datetime, timezone
from urllib.parse import parse_qs, unquote, urlparse

from .aggregates import cost_totals, format_aggregates, session_aggregates
from .compact import _parse_ts, compact_log
from .costs import cost_for
from .otel import session_span
from .recorder import Recorder
from .redact import REDACTED
from .replay import (_short, diff_records, format_record, load_records,
                    model_usage, session_diff)
from .sampling import (Sampler, SamplingDeclinedError, SamplingError,
                       SamplingNotSupportedError, StdioTransport)
from .tail_sampling import (_is_error, _span_durations_ms,
                            tail_sample_records)
from .trace import (_label, _tree_depth, build_trace, link_parents,
                    session_trace, trace_subtree)
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
    {
        "uri": "blackbox://session/aggregates",
        "name": "session/aggregates",
        "description": ("The recorded session's totals as JSON: step "
                        "count and duration, per-type and per-tool "
                        "counts, per-model tokens and cost, error "
                        "steps, the slowest step, and how many "
                        "records an earlier compaction removed or "
                        "sampling dropped — the same totals "
                        "`blackbox aggregates` prints."),
        "mimeType": "application/json",
    },
    {
        "uri": "blackbox://session/trace",
        "name": "session/trace",
        "description": ("The recorded session's reconstructed causal "
                        "trace as JSON: step count, span-type "
                        "breakdown, the slowest steps, error steps, "
                        "parent gaps, and the parent-child tree — "
                        "the same trace `blackbox trace` prints."),
        "mimeType": "application/json",
    },
    {
        "uri": "blackbox://session/diff",
        "name": "session/diff",
        "description": ("What changed between the recorded session "
                        "and another run as JSON: changed, added and "
                        "removed steps (position by position, like "
                        "`blackbox diff`), per-model token and cost "
                        "deltas, and new vs fixed errors — the same "
                        "diff the summarize_diff prompt summarises. "
                        "Read it as blackbox://session/diff"
                        "?other_path=<path to the other run's log "
                        "(.jsonl or .db)>; without other_path the "
                        "read is an error."),
        "mimeType": "application/json",
    },
]

# Resource templates: the diff against a named run, one branch of
# the causal trace, one model's or one tool's cost totals, one
# step's raw record, one step's transcript line, and one step as a
# single OTLP span, so a client can discover them from
# resources/templates/list and expand the template instead of
# building a URI by hand. The diff template
# parameter is the other run's log path, exactly the other_path the
# prompt takes and the query-parameter resource takes — the template,
# the prompt and the resource all read the same diff, so they always
# agree. The parameter is a path, so it may contain slashes:
# blackbox://session/diff/data/run-b.jsonl names the run
# data/run-b.jsonl, and an absolute path keeps its leading slash
# (blackbox://session/diff//tmp/run-b.jsonl). A client that needs
# to name a path with spaces or other special characters
# percent-encodes it, and the server decodes it on read. The trace
# template parameter is a step number: blackbox://session/trace/2
# is the subtree rooted at step 2, the same node
# blackbox://session/trace carries for that step — the template
# and the resource both read `trace_subtree` over `build_trace`,
# so they always agree. The aggregates templates name one model
# or one tool: blackbox://session/aggregates/model/gpt-5 is the
# per-model entry for gpt-5 and
# blackbox://session/aggregates/tool/search the per-tool entry
# for search — the same entries `cost_totals` computes for the
# `blackbox costs` report, so the template and the report always
# agree. A model or tool name may itself contain slashes, so the
# name is the whole remainder of the URI after the prefix,
# percent-decoded on read, like the diff template's path.
RESOURCE_TEMPLATES = [
    {
        "uriTemplate": "blackbox://session/diff/{other_path}",
        "name": "session/diff/{other_path}",
        "description": ("What changed between the recorded session "
                        "and another run as JSON: changed, added and "
                        "removed steps (position by position, like "
                        "`blackbox diff`), per-model token and cost "
                        "deltas, and new vs fixed errors — the same "
                        "diff the summarize_diff prompt summarises "
                        "and the blackbox://session/diff resource "
                        "exposes. Read it by expanding the template "
                        "with the other run's log path (.jsonl or "
                        ".db), e.g. blackbox://session/diff/"
                        "data/run-b.jsonl."),
        "mimeType": "application/json",
    },
    {
        "uriTemplate": "blackbox://session/trace/{seq}",
        "name": "session/trace/{seq}",
        "description": ("One branch of the recorded session's "
                        "reconstructed causal trace as JSON: the "
                        "subtree rooted at step seq — the same node "
                        "the blackbox://session/trace resource "
                        "carries for that step, with its children "
                        "in log order — so a client can read one "
                        "branch without the whole tree. Read it by "
                        "expanding the template with a step number, "
                        "e.g. blackbox://session/trace/2."),
        "mimeType": "application/json",
    },
    {
        "uriTemplate": "blackbox://session/aggregates/model/{model}",
        "name": "session/aggregates/model/{model}",
        "description": ("One model's totals as JSON: calls, "
                        "input/output tokens and USD cost for that "
                        "model — the same per-model entry the "
                        "`blackbox costs` report carries, so a "
                        "client can read one model's spend without "
                        "the whole breakdown. Read it by expanding "
                        "the template with a model name, e.g. "
                        "blackbox://session/aggregates/model/gpt-5."),
        "mimeType": "application/json",
    },
    {
        "uriTemplate": "blackbox://session/aggregates/tool/{tool}",
        "name": "session/aggregates/tool/{tool}",
        "description": ("One tool's totals as JSON: calls, the "
                        "input/output tokens of the model calls "
                        "that invoked it, and their USD cost — the "
                        "same per-tool entry the `blackbox costs` "
                        "report carries, so a client can read one "
                        "tool's cost without the whole breakdown. "
                        "Read it by expanding the template with a "
                        "tool name, e.g. blackbox://session/"
                        "aggregates/tool/search."),
        "mimeType": "application/json",
    },
    {
        "uriTemplate": "blackbox://session/record/{seq}",
        "name": "session/record/{seq}",
        "description": ("One step's raw record as JSON: the record "
                        "for step seq — seq, timestamp, type, "
                        "payload, tokens and the hash chain — the "
                        "same record the blackbox://session.json "
                        "resource carries for that step, so a "
                        "client can read one step without the whole "
                        "log. Read it by expanding the template "
                        "with a step number, e.g. "
                        "blackbox://session/record/2."),
        "mimeType": "application/json",
    },
    {
        "uriTemplate": "blackbox://session/transcript/{seq}",
        "name": "session/transcript/{seq}",
        "description": ("One step's formatted transcript line as "
                        "text: the line the blackbox://session "
                        "resource carries for step seq — the same "
                        "line `blackbox replay` prints for that "
                        "step — so a client can read one step "
                        "without the whole transcript. Read it by "
                        "expanding the template with a step number, "
                        "e.g. blackbox://session/transcript/2."),
        "mimeType": "text/plain",
    },
    {
        "uriTemplate": "blackbox://session/span/{seq}",
        "name": "session/span/{seq}",
        "description": ("One step as a single OTLP span as JSON: "
                        "the span for step seq — the step kind as "
                        "the span name, the step number as the "
                        "span id, start/end times from the "
                        "record's timestamps, and model/tool/token/"
                        "cost attributes where the record has them "
                        "— the same span the `blackbox otel` "
                        "export carries for that step, so a client "
                        "can read one step as a span without "
                        "exporting the whole trace. A model_call "
                        "with an input/output token split and a "
                        "known price carries its USD cost; an "
                        "unpriced call carries none, never a "
                        "guessed number. Read it by expanding the "
                        "template with a step number, e.g. "
                        "blackbox://session/span/2."),
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
    {
        "name": "summarize_sampling",
        "description": ("Summarise the recorded session's tail-sampling "
                        "decisions: kept vs sampled-out spans, which "
                        "triggers fired, parent gaps, and a per-model "
                        "and per-type breakdown of the sampled-out "
                        "spans."),
        "arguments": [],
    },
    {
        "name": "summarize_trace",
        "description": ("Summarise the recorded session's reconstructed "
                        "causal trace: step count, span-type breakdown, "
                        "the slowest steps, error steps, parent gaps, "
                        "and per-tool and per-model cost and token "
                        "totals."),
        "arguments": [],
    },
    {
        "name": "summarize_diff",
        "description": ("Summarise what changed between the recorded "
                        "session and another run: changed, added and "
                        "removed steps, per-model token and cost "
                        "deltas, and new vs fixed errors."),
        "arguments": [
            {
                "name": "other_path",
                "description": ("Path to the second run's log "
                                "(.jsonl or .db) to compare this "
                                "session against."),
                "required": True,
            },
        ],
    },
    {
        "name": "summarize_retention",
        "description": ("Summarise the recorded session's retention "
                        "state: what earlier compactions removed, "
                        "how much history is still in the log, and "
                        "what the default retention rules would "
                        "remove next."),
        "arguments": [
            {
                "name": "keep_days",
                "description": ("Days of history to keep in the "
                                "--keep-days projection (default 30)."),
                "required": False,
            },
            {
                "name": "keep_n",
                "description": ("Newest records to keep in the "
                                "--keep-n projection (default 1000)."),
                "required": False,
            },
        ],
    },
    {
        "name": "summarize_aggregates",
        "description": ("Summarise the recorded session at a glance: "
                        "step count and duration, per-type and "
                        "per-tool counts, per-model tokens and cost, "
                        "error steps, the slowest step, and how many "
                        "records an earlier compaction removed or "
                        "sampling dropped."),
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

SUMMARIZE_SAMPLING_PROMPT = (
    "Summarise the tail-sampling decisions for this blackbox "
    "session: which spans a tail-sampled export would keep and "
    "which it would sample out, which triggers fired, and what "
    "was sampled out by type and model. The decision below uses "
    "the tail-sample defaults: a trace with an error step, a span "
    "slower than 5 s or a cost over $1 keeps every span, and a "
    "boring trace is sampled down at keep rate 0.5 with seed 0, "
    "so the decision is reproducible. The sampled-out spans are "
    "still in the log — tail sampling only decides what an export "
    "keeps, it never deletes steps. A parent gap is a kept span "
    "whose chain parent was sampled out, so the export names a "
    "parent that is not in it.\n\n"
    "Sampling decision:\n{sampling}\n\n"
    "Session transcript:\n{transcript}"
)

SUMMARIZE_TRACE_PROMPT = (
    "Summarise this blackbox session as one reconstructed causal "
    "trace: the parent-child tree rebuilt from the steps, not the "
    "flat log order. A tool_result hangs off the tool_call it "
    "answers, a tool_call off the model_call that decided it, and "
    "a model_call off the step before it — the links are a "
    "reconstruction from step order, not recorded parent pointers, "
    "so treat them as a best reading of the order, not a claim "
    "about intent. For each part of the summary below, say what "
    "it means for the run: which steps took the longest, which "
    "steps errored, and where the trace would break apart. A "
    "parent gap is a kept span whose causal parent — the step "
    "the trace links it to — would be sampled out by the "
    "tail-sample defaults (keep rate 0.5, seed 0, so the "
    "decision is reproducible): in a sampled export that span "
    "would dangle without its parent. The sampled-out spans are "
    "still in the log — tail sampling only decides what an "
    "export keeps, it never deletes steps.\n\n"
    "Trace summary:\n{trace}\n\n"
    "Session transcript:\n{transcript}"
)

SUMMARIZE_DIFF_PROMPT = (
    "Summarise what changed between two blackbox runs of the same "
    "task: run A is this session, run B is the other run. The diff "
    "below is position-by-position, like `blackbox diff`: a step "
    "at the same position with a different payload is changed, "
    "extra steps in run B are added, steps missing from run B "
    "are removed. Say what the agent did differently in run B, "
    "whether the change looks like progress or a regression, and "
    "what drove the token and cost deltas. Prices are the "
    "providers' published list prices and are estimates, not a "
    "bill; calls with no input/output token split, or a model "
    "with no published price, are unpriced, never guessed. "
    "Error steps are compared by position: a new error is one "
    "run B has where run A had none, a fixed error the reverse.\n\n"
    "Diff:\n{diff}\n\n"
    "Run A transcript (this session):\n{transcript_a}\n\n"
    "Run B transcript (other run):\n{transcript_b}"
)

SUMMARIZE_RETENTION_PROMPT = (
    "Summarise the retention state of this blackbox session: how "
    "much history is still in the log, what earlier compactions "
    "already removed (and what the checkpoint certifies about "
    "them), and what the default retention rules would remove "
    "next. The projections below are dry runs of `blackbox "
    "compact --keep-days {keep_days}` and `blackbox compact "
    "--keep-n {keep_n}` — nothing has been removed by asking "
    "for this prompt. A checkpoint certifies only how many "
    "records were removed and the head hash up to which they "
    "existed; the removed records themselves are gone and "
    "cannot be re-verified, so say what would be lost before "
    "recommending a compaction, and whether the log looks safe "
    "to compact or worth keeping whole.\n\n"
    "Retention summary:\n{retention}\n\n"
    "Session transcript:\n{transcript}"
)

SUMMARIZE_AGGREGATES_PROMPT = (
    "Summarise this blackbox session at a glance: how many steps "
    "it took, how long it ran, which tools it used and how often, "
    "what each model cost in tokens and USD, which steps look "
    "like errors, and which step was the slowest. The totals "
    "below are the session aggregates from `blackbox aggregates` "
    "— read-only, computed from the log, so asking for this "
    "prompt changes nothing. The totals count steps only: the "
    "summary record is bookkeeping and the checkpoint certifies "
    "records an earlier compaction removed, so compacted-away "
    "records and sampled-out steps are reported separately — "
    "they are not in these totals, and the removed records are "
    "gone and cannot be re-counted. Prices are the providers' "
    "published list prices and are estimates, not a bill; calls "
    "with no input/output token split, or a model with no "
    "published price, are unpriced, never guessed. The slowest "
    "step is the step with the longest time until the next step, "
    "not necessarily the step that did the most work.\n\n"
    "Session totals:\n{aggregates}\n\n"
    "Session transcript:\n{transcript}"
)


def _usd(cost):
    """A USD amount with no trailing zeros: 0.0049, not 0.004900."""
    return ("%.6f" % cost).rstrip("0").rstrip(".")


def _tree_lines(nodes, depth, lines):
    """A trace forest as indented lines, like `blackbox trace`."""
    for node in nodes:
        lines.append("%s[%d] %s" % ("  " * depth, node["seq"],
                                    node["label"]))
        _tree_lines(node["children"], depth + 1, lines)


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
        if method == "resources/templates/list":
            return _result(msg_id,
                           {"resourceTemplates": RESOURCE_TEMPLATES})
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

    def _model_usage(self, records, default_model=None):
        """Per-model token usage for a list of records.

        The shared `model_usage` from replay.py — same rules as
        the Recorder's summary, so the cost summary, the diff
        prompt and the diff resource always agree on a model's
        tokens.
        """
        return model_usage(records, default_model)

    def _cost_lines(self):
        """Per-model token usage and USD cost, as readable lines.

        Computed from the recorded model_call records, with the same
        rules as the Recorder's summary: a plain int token count has
        no input/output split and is not priced, and an unknown model
        costs None, never a guessed number.
        """
        usage = self._model_usage(load_records(self.log_path),
                                  default_model=self.recorder.model)
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
                parts.append("$%s" % _usd(cost))
                total += cost
                priced_any = True
            elif entry["input_tokens"] or entry["output_tokens"]:
                parts.append("cost unknown: no published price "
                             "for this model")
            else:
                parts.append("not priced: no input/output split")
            lines.append("%s: %s" % (model, ", ".join(parts)))
        if priced_any:
            lines.append("Total priced cost: $%s" % _usd(total))
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

    def _sampling_lines(self):
        """The tail-sampling decision for the session, as readable
        lines.

        The same decision `blackbox tail-sample` makes with its
        defaults (keep rate 0.5, a slow span over 5 s or a cost
        over $1 keeps the whole session), with a fixed seed (0) so
        the summary is reproducible. The summary record is skipped
        — it is bookkeeping, not a span. The decision describes the
        export only: sampled-out spans are still in the log.
        """
        records = load_records(self.log_path)
        steps = [r for r in records if r["type"] != "summary"]
        if not steps:
            return "No steps recorded yet."
        kept, decision = tail_sample_records(
            records, keep_rate=0.5, seed=0)
        kept_seqs = {r["seq"] for r in kept}
        lines = []
        if decision["decision"] == "kept_all":
            lines.append(
                "Decision: kept_all — a trigger fired (%s), so every "
                "span would be kept"
                % ", ".join(decision["reasons"]))
        else:
            lines.append(
                "Decision: sampled — no trigger fired, so the session "
                "would be sampled down at keep rate 0.5 (seed 0, "
                "reproducible)")
        lines.append(
            "Kept %d of %d spans, %d sampled out"
            % (decision["kept_spans"], decision["total_spans"],
               decision["dropped_spans"]))
        if decision["max_span_ms"] is not None:
            lines.append("Longest span: %.1f ms (slow threshold 5000 ms)"
                         % decision["max_span_ms"])
        if decision["total_cost_usd"] is not None:
            lines.append("Total priced cost: $%.6f (high-cost threshold "
                         "$1.00)" % decision["total_cost_usd"])
        else:
            lines.append("Total priced cost: unpriced (no model call "
                         "with an input/output split and a known price)")
        if decision["parent_gaps"]:
            lines.append(
                "Parent gaps: %d kept span(s) whose chain parent was "
                "sampled out" % decision["parent_gaps"])
        dropped = [r for r in steps if r["seq"] not in kept_seqs]
        if not dropped:
            lines.append("Nothing sampled out: every span was kept.")
        else:
            lines.append("Sampled out (%d):" % len(dropped))
            by_label = {}
            for record in dropped:
                payload = record.get("payload") or {}
                label = record["type"]
                if record["type"] == "model_call":
                    model = payload.get("model")
                    if model:
                        label = "model_call (%s)" % model
                elif record["type"] in ("tool_call", "tool_result"):
                    tool = payload.get("tool")
                    if tool:
                        label = "%s (%s)" % (record["type"], tool)
                by_label[label] = by_label.get(label, 0) + 1
            for label in sorted(by_label):
                lines.append("  %s: %d" % (label, by_label[label]))
        return "\n".join(lines)

    def _trace_lines(self):
        """The session's reconstructed causal trace, as readable
        lines.

        Step count and span-type breakdown, the slowest steps (a
        step's duration is the time until the next step, as in the
        tail-sampling decision), the error steps, the parent gaps,
        and per-tool / per-model cost and token totals. The parent
        gaps are causal, not chain, gaps: a span the tail-sample
        defaults (keep rate 0.5, seed 0) would keep, whose causal
        parent from trace.link_parents would be sampled out — in a
        sampled export that span would dangle without its parent.
        A kept-all trace (a trigger fired) has no gaps. The trace
        tree itself is printed last, so the shape is visible, not
        just the counts.
        """
        records = load_records(self.log_path)
        steps = [r for r in records if r["type"] != "summary"]
        if not steps:
            return "No steps recorded yet."
        parents = link_parents(records)
        roots = build_trace(records)
        lines = []
        lines.append("Trace: %d steps, %d root(s), max depth %d"
                     % (len(steps), len(roots),
                        max(_tree_depth(r) for r in roots)))
        by_type = {}
        for record in steps:
            by_type[record["type"]] = by_type.get(record["type"], 0) + 1
        lines.append("By type: %s" % ", ".join(
            "%s: %d" % (t, by_type[t]) for t in sorted(by_type)))
        durations = _span_durations_ms(records)
        timed = sorted(
            ((d, r) for d, r in zip(durations, steps) if d is not None),
            key=lambda pair: pair[0], reverse=True)[:3]
        if not timed:
            lines.append("Slowest steps: no timing available.")
        else:
            lines.append("Slowest steps:")
            for duration, record in timed:
                lines.append("  step %d: %s — %.1f ms"
                             % (record["seq"], _label(record), duration))
        errors = [r for r in steps if _is_error(r)]
        if not errors:
            lines.append("Error steps: none.")
        else:
            lines.append("Error steps:")
            for record in errors:
                payload = record.get("payload") or {}
                detail = ""
                if "error" in payload:
                    detail = ": %s" % _short(str(payload["error"]))
                lines.append("  step %d: %s%s"
                             % (record["seq"], _label(record), detail))
        kept, decision = tail_sample_records(
            records, keep_rate=0.5, seed=0)
        kept_seqs = {r["seq"] for r in kept}
        gaps = []
        for record in steps:
            parent = parents.get(record["seq"])
            if (record["seq"] in kept_seqs and parent is not None
                    and parent not in kept_seqs):
                gaps.append((record, parent))
        if decision["decision"] == "kept_all":
            lines.append(
                "Parent gaps: none — the trace would be kept whole "
                "(trigger: %s), so no parent is sampled out"
                % ", ".join(decision["reasons"]))
        elif not gaps:
            lines.append(
                "Parent gaps: none — every kept span's causal parent "
                "is kept too.")
        else:
            lines.append(
                "Parent gaps: %d kept span(s) whose causal parent "
                "would be sampled out (keep rate 0.5, seed 0):"
                % len(gaps))
            for record, parent in gaps:
                lines.append("  step %d: %s (causal parent step %d)"
                             % (record["seq"], _label(record), parent))
        lines.append("Cost and tokens (per model):")
        lines.append(self._cost_lines())
        tool_counts = {}
        for record in steps:
            if record["type"] != "tool_call":
                continue
            tool = (record.get("payload") or {}).get("tool") or "?"
            tool_counts[tool] = tool_counts.get(tool, 0) + 1
        if not tool_counts:
            lines.append("Tool calls: none.")
        else:
            lines.append("Tool calls:")
            for tool in sorted(tool_counts):
                lines.append("  %s: %d" % (tool, tool_counts[tool]))
        lines.append("Trace tree:")
        _tree_lines(roots, 0, lines)
        return "\n".join(lines)

    def _diff_lines(self, other_path):
        """What changed between this session (run A) and another
        run (run B), as readable lines.

        Position-by-position, like `blackbox diff` (and using the
        same `diff_records`): a step at the same position with a
        different payload is changed, extra steps in run B are
        added, steps missing from run B are removed. Token and
        cost deltas are per model, with the same pricing rules as
        the cost summary — a plain int token count has no
        input/output split and is not priced, an unknown model
        costs None, never a guessed number. Error steps are
        compared by position too: an error run B has where run A
        had none is new, one run A had that run B does not is
        fixed. The summary record is skipped in both runs — it is
        bookkeeping, not a step.
        """
        records_a = load_records(self.log_path)
        records_b = load_records(other_path)
        steps_a = [r for r in records_a if r["type"] != "summary"]
        steps_b = [r for r in records_b if r["type"] != "summary"]
        if not steps_a and not steps_b:
            return "No steps in either run."
        findings = diff_records(records_a, records_b)
        lines = []
        lines.append(
            "Run A (this session): %d steps, %d tokens; run B "
            "(other run): %d steps, %d tokens (token diff %+d)"
            % (len(steps_a), findings["tokens_a"],
               len(steps_b), findings["tokens_b"],
               findings["tokens_b"] - findings["tokens_a"]))
        if (not findings["changed"] and not findings["added"]
                and not findings["removed"]):
            lines.append("IDENTICAL: %d steps match" % len(steps_a))
        for i in findings["changed"]:
            lines.append("CHANGED step %d:" % i)
            lines.append("  A: %s" % format_record(steps_a[i])
                         .replace("\n", "\n  "))
            lines.append("  B: %s" % format_record(steps_b[i])
                         .replace("\n", "\n  "))
        for i in findings["added"]:
            lines.append("ADDED step %d (only in run B): %s"
                         % (i, format_record(steps_b[i])
                            .replace("\n", " ")))
        for i in findings["removed"]:
            lines.append("REMOVED step %d (only in run A): %s"
                         % (i, format_record(steps_a[i])
                            .replace("\n", " ")))
        usage_a = self._model_usage(
            records_a, default_model=self.recorder.model)
        usage_b = self._model_usage(records_b)
        models = sorted(set(usage_a) | set(usage_b))
        if models:
            lines.append("Cost and token deltas (per model):")
            empty = {"input_tokens": 0, "output_tokens": 0,
                     "tokens": 0}
            for model in models:
                entry_a = usage_a.get(model, empty)
                entry_b = usage_b.get(model, empty)
                tokens_a = (entry_a["input_tokens"]
                            + entry_a["output_tokens"]
                            + entry_a["tokens"])
                tokens_b = (entry_b["input_tokens"]
                            + entry_b["output_tokens"]
                            + entry_b["tokens"])
                cost_a = (cost_for(model, entry_a["input_tokens"],
                                   entry_a["output_tokens"])
                          if entry_a["input_tokens"]
                          or entry_a["output_tokens"] else None)
                cost_b = (cost_for(model, entry_b["input_tokens"],
                                   entry_b["output_tokens"])
                          if entry_b["input_tokens"]
                          or entry_b["output_tokens"] else None)
                cost_text_a = ("$%s" % _usd(cost_a)
                               if cost_a is not None else "unpriced")
                cost_text_b = ("$%s" % _usd(cost_b)
                               if cost_b is not None else "unpriced")
                if cost_a is not None and cost_b is not None:
                    delta = cost_b - cost_a
                    if delta >= 0:
                        delta_text = "+$%s" % _usd(delta)
                    else:
                        delta_text = "-$%s" % _usd(-delta)
                    cost_text = "%s -> %s (%s)" % (
                        cost_text_a, cost_text_b, delta_text)
                else:
                    cost_text = "%s -> %s (cost delta unknown: " \
                        "unpriced in one run)" % (cost_text_a,
                                                  cost_text_b)
                lines.append(
                    "  %s: %d tokens -> %d tokens (%+d), %s"
                    % (model, tokens_a, tokens_b, tokens_b - tokens_a,
                       cost_text))
        new_errors = []
        fixed_errors = []
        for i in range(max(len(steps_a), len(steps_b))):
            error_a = i < len(steps_a) and _is_error(steps_a[i])
            error_b = i < len(steps_b) and _is_error(steps_b[i])
            if error_b and not error_a:
                new_errors.append(steps_b[i])
            elif error_a and not error_b:
                fixed_errors.append(steps_a[i])
        errors_a = [r for r in steps_a if _is_error(r)]
        errors_b = [r for r in steps_b if _is_error(r)]
        if not errors_a and not errors_b:
            lines.append("Error steps: none in either run.")
        else:
            if new_errors:
                lines.append("New errors (in run B, not run A):")
                for record in new_errors:
                    lines.append("  step %d: %s"
                                 % (record["seq"], _label(record)))
            if fixed_errors:
                lines.append("Errors fixed (in run A, not run B):")
                for record in fixed_errors:
                    lines.append("  step %d: %s"
                                 % (record["seq"], _label(record)))
            if not new_errors and not fixed_errors:
                lines.append("Error steps: same in both runs.")
        return "\n".join(lines)

    def _retention_lines(self, keep_days=30, keep_n=1000):
        """The session's retention / compaction state, as readable
        lines.

        Whether an earlier compaction already removed records
        (and what its checkpoint certifies), how much history is
        still in the log and how old it is, and what the default
        retention rules would remove next — as dry runs of
        `compact_log`, so asking for the summary removes nothing.
        The checkpoint and the summary record are bookkeeping, not
        steps; the projections count steps only.
        """
        records = load_records(self.log_path)
        checkpoint = (records[0]
                      if records and records[0].get("type") == "checkpoint"
                      else None)
        steps = [r for r in records
                 if r["type"] not in ("summary", "checkpoint")]
        if not steps and checkpoint is None:
            return "No steps recorded yet."
        lines = []
        if checkpoint is not None:
            payload = checkpoint.get("payload") or {}
            head = payload.get("head_hash") or "?"
            lines.append(
                "Already compacted: %d record(s) were removed by "
                "earlier compaction(s), certified by a checkpoint "
                "(head hash %s..., cutoff %s, %s kept at the time)"
                % (payload.get("compacted_count", 0), head[:16],
                   payload.get("cutoff_ts") or "no date cutoff",
                   payload.get("kept_count", "?")))
            if payload.get("original_events") is not None:
                lines.append(
                    "  The checkpoint carries the dropped run's "
                    "totals: %s events, %s tokens"
                    % (payload["original_events"],
                       payload.get("original_total_tokens", "?")))
        else:
            lines.append(
                "Never compacted: no checkpoint record — the full "
                "history is still in the log.")
        if steps:
            oldest = _parse_ts(steps[0]["ts"], steps[0]["seq"])
            newest = _parse_ts(steps[-1]["ts"], steps[-1]["seq"])
            now = datetime.now(timezone.utc)
            lines.append(
                "Log now: %d step(s), oldest %s (%.1f days ago), "
                "newest %s, spanning %.1f day(s)"
                % (len(steps), steps[0]["ts"],
                   (now - oldest).total_seconds() / 86400,
                   steps[-1]["ts"],
                   (newest - oldest).total_seconds() / 86400))
        else:
            lines.append("Log now: no steps left — only the "
                         "checkpoint.")
        lines.append("Retention projections (dry run — nothing is "
                     "removed):")
        for label, kwargs in (
                ("--keep-days %d" % keep_days,
                 {"keep_days": keep_days}),
                ("--keep-n %d" % keep_n, {"keep_n": keep_n})):
            result = compact_log(self.log_path, dry_run=True, **kwargs)
            if not result["changed"]:
                lines.append("  %s: would remove nothing (%d step(s) "
                             "kept)" % (label, result["kept"]))
            else:
                lines.append(
                    "  %s: would remove %d record(s), keep %d; the "
                    "new checkpoint would certify head hash %s..."
                    % (label, result["compacted"], result["kept"],
                       (result["head_hash"] or "")[:16]))
        lines.append(
            "Removed records are gone: the checkpoint certifies "
            "only how many were removed and the head hash up to "
            "which they existed — it cannot re-verify them.")
        return "\n".join(lines)

    def _aggregates_lines(self):
        """The session's totals at a glance, as readable lines.

        The same report `blackbox aggregates` prints
        (`format_aggregates` over `session_aggregates`): step count
        and duration, per-type and per-tool counts, per-model
        calls/tokens/cost, error steps, the slowest step, and how
        many records an earlier compaction removed or sampling
        dropped. Read-only — asking for the summary removes
        nothing.
        """
        records = load_records(self.log_path)
        agg = session_aggregates(records)
        if not agg["steps"] and not agg["compacted_count"]:
            return "No steps recorded yet."
        return format_aggregates(agg)

    def _read_diff(self, msg_id, uri, other_path):
        """The session-vs-other-run diff as a resource read.

        One implementation for both ways of naming the other run:
        the blackbox://session/diff?other_path=... resource and the
        blackbox://session/diff/{other_path} template. Both reuse
        `session_diff` from replay.py — the same diff the
        summarize_diff prompt summarises — so the template, the
        prompt and the resource always agree, and an unpriced call
        costs None, never a guessed number.
        """
        try:
            records_a = load_records(self.log_path)
            records_b = load_records(other_path)
        except (OSError, ValueError) as exc:
            return _error(msg_id, INVALID_PARAMS,
                          "cannot read the other run's log %r: %s"
                          % (other_path, exc))
        diff = session_diff(
            records_a, records_b,
            default_model_a=self.recorder.model)
        text = json.dumps(diff, indent=2, sort_keys=True)
        return _result(msg_id, {"contents": [
            {"uri": uri, "mimeType": "application/json",
             "text": text}]})

    def _read_trace_subtree(self, msg_id, uri, seq_text):
        """One branch of the session's causal trace, as a resource
        read.

        One implementation for the blackbox://session/trace/{seq}
        template: the subtree rooted at step seq, from
        `trace_subtree` in trace.py — the same node
        `build_trace` builds for the whole session — so the
        template and the blackbox://session/trace resource always
        agree. A step number that is not an integer, or a step the
        session does not have, is an error.
        """
        try:
            seq = int(unquote(seq_text))
        except ValueError:
            return _error(
                msg_id, INVALID_PARAMS,
                "blackbox://session/trace/{seq} needs a step number, "
                "e.g. blackbox://session/trace/2")
        records = load_records(self.log_path)
        node = trace_subtree(records, seq)
        if node is None:
            return _error(msg_id, INVALID_PARAMS,
                          "no step %d in this session" % seq)
        text = json.dumps(node, indent=2, sort_keys=True)
        return _result(msg_id, {"contents": [
            {"uri": uri, "mimeType": "application/json",
             "text": text}]})

    def _read_aggregates_slice(self, msg_id, uri, kind, name_text):
        """One model's or one tool's cost totals, as a resource
        read.

        One implementation for both aggregates templates:
        blackbox://session/aggregates/model/{model} and
        blackbox://session/aggregates/tool/{tool}. Both reuse
        `cost_totals` from aggregates.py — the same totals the
        `blackbox costs` report prints — so the template and the
        report always agree, and an unpriced call costs None,
        never a guessed number. A model or tool the session does
        not have, or an empty name, is an error.
        """
        name = unquote(name_text)
        if not name:
            return _error(
                msg_id, INVALID_PARAMS,
                "blackbox://session/aggregates/%s/{%s} needs a %s "
                "name, e.g. blackbox://session/aggregates/%s/%s"
                % (kind, kind, kind, kind,
                   "gpt-5" if kind == "model" else "search"))
        records = load_records(self.log_path)
        totals = cost_totals(records)
        section = ("per_model" if kind == "model" else "per_tool")
        entry = totals[section].get(name)
        if entry is None:
            return _error(msg_id, INVALID_PARAMS,
                          "no %s %r in this session" % (kind, name))
        result = {kind: name}
        result.update(entry)
        text = json.dumps(result, indent=2, sort_keys=True)
        return _result(msg_id, {"contents": [
            {"uri": uri, "mimeType": "application/json",
             "text": text}]})

    def _read_record(self, msg_id, uri, seq_text):
        """One step's raw record, as a resource read.

        One implementation for the blackbox://session/record/{seq}
        template: the record for step seq, from `load_records` —
        the same record the blackbox://session.json resource
        carries for that step — so the template and the resource
        always agree. A step number that is not an integer, or a
        step the session does not have, is an error.
        """
        try:
            seq = int(unquote(seq_text))
        except ValueError:
            return _error(
                msg_id, INVALID_PARAMS,
                "blackbox://session/record/{seq} needs a step number, "
                "e.g. blackbox://session/record/2")
        records = load_records(self.log_path)
        record = next((r for r in records if r["seq"] == seq), None)
        if record is None:
            return _error(msg_id, INVALID_PARAMS,
                          "no step %d in this session" % seq)
        text = json.dumps(record, indent=2, sort_keys=True)
        return _result(msg_id, {"contents": [
            {"uri": uri, "mimeType": "application/json",
             "text": text}]})

    def _read_transcript_step(self, msg_id, uri, seq_text):
        """One step's formatted transcript line, as a resource read.

        One implementation for the
        blackbox://session/transcript/{seq} template: the
        `format_record` line for step seq — the same line the
        blackbox://session resource carries for that step — so
        the template and the resource always agree. A step number
        that is not an integer, or a step the session does not
        have, is an error.
        """
        try:
            seq = int(unquote(seq_text))
        except ValueError:
            return _error(
                msg_id, INVALID_PARAMS,
                "blackbox://session/transcript/{seq} needs a step "
                "number, e.g. blackbox://session/transcript/2")
        records = load_records(self.log_path)
        record = next((r for r in records if r["seq"] == seq), None)
        if record is None:
            return _error(msg_id, INVALID_PARAMS,
                          "no step %d in this session" % seq)
        text = format_record(record)
        return _result(msg_id, {"contents": [
            {"uri": uri, "mimeType": "text/plain",
             "text": text}]})

    def _read_span(self, msg_id, uri, seq_text):
        """One step as a single OTLP span, as a resource read.

        One implementation for the blackbox://session/span/{seq}
        template: the span `session_span` in otel.py builds from
        the record — the same span builder the full `blackbox
        otel` export uses — so the template agrees with the
        export, the transcript and the raw record. The span id
        is the step number, the span name the step kind, start/end
        times come from the record's timestamps, and a model_call
        with an input/output split and a known price carries its
        USD cost — an unpriced call carries none, never a guessed
        number. A step number that is not an integer, a step the
        session does not have, or a step with no usable timestamp,
        is an error.
        """
        try:
            seq = int(unquote(seq_text))
        except ValueError:
            return _error(
                msg_id, INVALID_PARAMS,
                "blackbox://session/span/{seq} needs a step number, "
                "e.g. blackbox://session/span/2")
        records = load_records(self.log_path)
        try:
            span = session_span(records, seq)
        except ValueError as exc:
            return _error(msg_id, INVALID_PARAMS, str(exc))
        if span is None:
            return _error(msg_id, INVALID_PARAMS,
                          "no step %d in this session" % seq)
        text = json.dumps(span, indent=2, sort_keys=True)
        return _result(msg_id, {"contents": [
            {"uri": uri, "mimeType": "application/json",
             "text": text}]})

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
        if uri == "blackbox://session/aggregates":
            records = load_records(self.log_path)
            agg = session_aggregates(records)
            text = json.dumps(agg, indent=2, sort_keys=True)
            return _result(msg_id, {"contents": [
                {"uri": uri, "mimeType": "application/json",
                 "text": text}]})
        if uri == "blackbox://session/trace":
            records = load_records(self.log_path)
            trace = session_trace(records)
            text = json.dumps(trace, indent=2, sort_keys=True)
            return _result(msg_id, {"contents": [
                {"uri": uri, "mimeType": "application/json",
                 "text": text}]})
        if uri.startswith("blackbox://session/aggregates/model/"):
            # The resource template
            # blackbox://session/aggregates/model/{model}: the
            # per-model totals for one model — the same entry
            # `cost_totals` computes for the `blackbox costs`
            # report. The name is the whole remainder of the URI,
            # percent-decoded, so a model name with a slash in it
            # still names one model.
            name_text = uri[len("blackbox://session/aggregates/model/"):]
            return self._read_aggregates_slice(msg_id, uri, "model",
                                               name_text)
        if uri.startswith("blackbox://session/aggregates/tool/"):
            # The resource template
            # blackbox://session/aggregates/tool/{tool}: the
            # per-tool totals for one tool — the same entry
            # `cost_totals` computes for the `blackbox costs`
            # report. The name is the whole remainder of the URI,
            # percent-decoded, so a tool name with a slash in it
            # still names one tool.
            name_text = uri[len("blackbox://session/aggregates/tool/"):]
            return self._read_aggregates_slice(msg_id, uri, "tool",
                                               name_text)
        if uri.startswith("blackbox://session/record/"):
            # The resource template blackbox://session/record/{seq}:
            # one step's raw record as JSON, the same record the
            # blackbox://session.json resource carries for that step.
            seq_text = uri[len("blackbox://session/record/"):]
            if not seq_text:
                return _error(
                    msg_id, INVALID_PARAMS,
                    "blackbox://session/record/{seq} needs a step "
                    "number, e.g. blackbox://session/record/2")
            return self._read_record(msg_id, uri, seq_text)
        if uri.startswith("blackbox://session/transcript/"):
            # The resource template
            # blackbox://session/transcript/{seq}: one step's
            # formatted transcript line, the same line the
            # blackbox://session resource carries for that step.
            seq_text = uri[len("blackbox://session/transcript/"):]
            if not seq_text:
                return _error(
                    msg_id, INVALID_PARAMS,
                    "blackbox://session/transcript/{seq} needs a "
                    "step number, e.g. "
                    "blackbox://session/transcript/2")
            return self._read_transcript_step(msg_id, uri, seq_text)
        if uri.startswith("blackbox://session/span/"):
            # The resource template blackbox://session/span/{seq}:
            # one step as a single OTLP span — the same span the
            # `blackbox otel` export carries for that step, with
            # the step number as the span id and the step kind as
            # the span name.
            seq_text = uri[len("blackbox://session/span/"):]
            if not seq_text:
                return _error(
                    msg_id, INVALID_PARAMS,
                    "blackbox://session/span/{seq} needs a step "
                    "number, e.g. blackbox://session/span/2")
            return self._read_span(msg_id, uri, seq_text)
        if uri.startswith("blackbox://session/trace/"):
            # The resource template blackbox://session/trace/{seq}:
            # the subtree rooted at step seq, the same node the
            # trace resource carries for that step.
            seq_text = uri[len("blackbox://session/trace/"):]
            if not seq_text:
                return _error(
                    msg_id, INVALID_PARAMS,
                    "blackbox://session/trace/{seq} needs a step "
                    "number, e.g. blackbox://session/trace/2")
            return self._read_trace_subtree(msg_id, uri, seq_text)
        if (uri == "blackbox://session/diff"
                or uri.startswith("blackbox://session/diff?")):
            # A resource read takes no arguments, so the other
            # run's log path rides in the URI as a query
            # parameter: blackbox://session/diff?other_path=...
            other_path = parse_qs(
                urlparse(uri).query).get("other_path", [None])[0]
            if not other_path:
                return _error(
                    msg_id, INVALID_PARAMS,
                    "blackbox://session/diff needs an 'other_path' "
                    "query parameter: the path of the second run's "
                    "log, e.g. blackbox://session/diff"
                    "?other_path=data/run-b.jsonl")
            return self._read_diff(msg_id, uri, other_path)
        if uri.startswith("blackbox://session/diff/"):
            # The resource template blackbox://session/diff/
            # {other_path}: the other run's log path is the URI
            # path after the prefix, percent-decoded — an absolute
            # path keeps its leading slash, so
            # blackbox://session/diff//tmp/run-b.jsonl names
            # /tmp/run-b.jsonl.
            other_path = unquote(
                uri[len("blackbox://session/diff/"):])
            if not other_path:
                return _error(
                    msg_id, INVALID_PARAMS,
                    "blackbox://session/diff/{other_path} needs the "
                    "other run's log path in the URI, e.g. "
                    "blackbox://session/diff/data/run-b.jsonl")
            return self._read_diff(msg_id, uri, other_path)
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
        if name == "summarize_sampling":
            text = SUMMARIZE_SAMPLING_PROMPT.format(
                sampling=self._sampling_lines(),
                transcript=self._transcript())
            return _result(msg_id, {
                "description": description,
                "messages": [{"role": "user",
                              "content": {"type": "text", "text": text}}],
            })
        if name == "summarize_trace":
            text = SUMMARIZE_TRACE_PROMPT.format(
                trace=self._trace_lines(),
                transcript=self._transcript())
            return _result(msg_id, {
                "description": description,
                "messages": [{"role": "user",
                              "content": {"type": "text", "text": text}}],
            })
        if name == "summarize_diff":
            other_path = arguments.get("other_path")
            if (not isinstance(other_path, str) or not other_path):
                return _error(
                    msg_id, INVALID_PARAMS,
                    "summarize_diff needs an 'other_path' argument: "
                    "the path of the second run's log")
            try:
                diff_text = self._diff_lines(other_path)
                other_transcript = "\n".join(
                    format_record(r)
                    for r in load_records(other_path))
            except (OSError, ValueError) as exc:
                return _error(
                    msg_id, INVALID_PARAMS,
                    "cannot read run B (%s): %s" % (other_path, exc))
            text = SUMMARIZE_DIFF_PROMPT.format(
                diff=diff_text,
                transcript_a=self._transcript(),
                transcript_b=other_transcript)
            return _result(msg_id, {
                "description": description,
                "messages": [{"role": "user",
                              "content": {"type": "text", "text": text}}],
            })
        if name == "summarize_retention":
            keep_days = arguments.get("keep_days", 30)
            keep_n = arguments.get("keep_n", 1000)
            for label, value in (("keep_days", keep_days),
                                 ("keep_n", keep_n)):
                if (not isinstance(value, int) or isinstance(value, bool)
                        or value < 0):
                    return _error(
                        msg_id, INVALID_PARAMS,
                        "%s must be a non-negative integer" % label)
            try:
                retention = self._retention_lines(
                    keep_days=keep_days, keep_n=keep_n)
            except (OSError, ValueError) as exc:
                return _error(msg_id, INVALID_PARAMS,
                              "cannot read session: %s" % exc)
            text = SUMMARIZE_RETENTION_PROMPT.format(
                keep_days=keep_days, keep_n=keep_n,
                retention=retention, transcript=self._transcript())
            return _result(msg_id, {
                "description": description,
                "messages": [{"role": "user",
                              "content": {"type": "text", "text": text}}],
            })
        if name == "summarize_aggregates":
            try:
                aggregates = self._aggregates_lines()
            except (OSError, ValueError) as exc:
                return _error(msg_id, INVALID_PARAMS,
                              "cannot read session: %s" % exc)
            text = SUMMARIZE_AGGREGATES_PROMPT.format(
                aggregates=aggregates, transcript=self._transcript())
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
