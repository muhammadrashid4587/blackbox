"""CLI: python -m blackbox verify|replay|diff|mcp|otel|metrics|tail-sample|export|query|trace <logfile> ...

    python -m blackbox verify data/demo.jsonl
    python -m blackbox replay data/demo.jsonl
    python -m blackbox diff data/run-a.jsonl data/run-b.jsonl
    python -m blackbox mcp data/run.jsonl   # MCP server on stdio
    python -m blackbox record data/run.jsonl --type tool_call \
        --payload '{"tool": "search", "args": {"query": "mcp"}}'
    printf '%s\n' '{"type": "tool_call", "payload": {"tool": "search"}}' \
        | python -m blackbox record data/run.jsonl --stdin
    python -m blackbox record data/run.jsonl --file steps.jsonl
    python -m blackbox span data/run.jsonl 0  # one step as an OTLP span
    python -m blackbox watch data/run.jsonl  # tail a live log
    python -m blackbox watch data/run.jsonl --type tool_call --errors-only
    python -m blackbox watch data/run.jsonl --since 2026-11-24T01:00:00Z --until 2026-11-24T02:00:00Z
    python -m blackbox watch data/run.jsonl --last 10m  # recent steps, then live
    python -m blackbox otel data/run.jsonl --out spans.json
    python -m blackbox metrics data/run.jsonl --out metrics.json
    python -m blackbox tail-sample data/run.jsonl --out spans.json
    python -m blackbox export data/run.jsonl --live  # POST spans to a collector
    python -m blackbox query data/run.jsonl --model gpt-5 --min-cost 0.01
    python -m blackbox trace data/run.jsonl  # causal parent-child tree
    python -m blackbox compact data/run.jsonl --keep-days 30  # retention
    python -m blackbox aggregates data/run.jsonl  # per-session totals
    python -m blackbox costs data/run.jsonl  # per-model/per-tool cost totals
    python -m blackbox compact data/run.jsonl --keep-n 1000

Any command takes a .jsonl log or a .db SQLite store.
"""

import argparse
import json
import os
import sys

from .aggregates import aggregates_log, cost_totals_log
from .compact import compact_log
from .live_export import DEFAULT_OTLP_ENDPOINT, OtlpHttpExporter
from .mcp_server import serve_stdio
from .otel import export_otlp, export_otlp_metrics, format_span, session_span
from .query import query_log
from .recorder import Recorder, _normalize_ts
from .replay import diff, load_records, replay
from .tail_sampling import (DEFAULT_MAX_COST_USD, DEFAULT_SLOW_MS,
                            export_tail_sampled_otlp)
from .trace import trace_log
from .verify import verify_cli
from .watch import watch_log


def _parse_record_tokens(text):
    """Parse a --tokens value: a JSON int, or a JSON object of
    non-negative int counts (an input/output split).

    Raises ValueError with a plain message on anything else — a
    token count is never guessed from a bad value.
    """
    try:
        tokens = json.loads(text)
    except json.JSONDecodeError:
        raise ValueError("--tokens must be a JSON int or object, "
                         "e.g. 120 or '{\"input\": 1200, "
                         "\"output\": 340}'") from None
    if isinstance(tokens, bool):
        raise ValueError("--tokens must be a JSON int or object")
    if isinstance(tokens, int):
        if tokens < 0:
            raise ValueError("--tokens must not be negative")
        return tokens
    if isinstance(tokens, dict):
        for key, value in tokens.items():
            if (not isinstance(key, str) or isinstance(value, bool)
                    or not isinstance(value, int) or value < 0):
                raise ValueError(
                    "--tokens object values must be non-negative ints")
        return tokens
    raise ValueError("--tokens must be a JSON int or object, "
                     "e.g. 120 or '{\"input\": 1200, \"output\": 340}'")


def _record_task_model(args):
    """Keep the log's existing task/model in the summary the
    Recorder writes on close, unless the flags override them."""
    task, model = args.task, args.model
    if (task is None or model is None) and os.path.exists(args.logfile):
        try:
            existing = load_records(args.logfile)
        except (FileNotFoundError, ValueError, OSError):
            existing = []
        if existing and existing[-1].get("type") == "summary":
            summary = existing[-1].get("payload") or {}
            if task is None:
                task = summary.get("task")
            if model is None:
                model = summary.get("model")
    return task, model


def _parse_bulk_steps(text, source):
    """Parse bulk steps for `blackbox record --stdin` / `--file`:
    one JSON object per line, or a single JSON array of step
    objects. Each step is {"type": ..., "payload": {...}, "tokens":
    ..., "ts": ...}; payload, tokens and ts are optional. `source`
    is "--stdin" or "--file" and names the input in error messages.
    Raises ValueError with a plain message on anything else — a bad
    step is never guessed at, and the caller validates every step
    before opening the log, so a bad batch leaves the log untouched.
    """
    stripped = text.strip()
    if not stripped:
        raise ValueError("%s: no steps %s"
                         % (source, "in the file"
                            if source == "--file" else "on stdin"))
    if stripped.startswith("["):
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            raise ValueError("%s: %s must be JSON steps, one "
                             "per line or a JSON array"
                             % (source, "file"
                                if source == "--file" else "stdin")) \
                from None
        if not isinstance(parsed, list):
            raise ValueError("%s: %s must be JSON steps, one "
                             "per line or a JSON array"
                             % (source, "file"
                                if source == "--file" else "stdin"))
    else:
        parsed = []
        for lineno, line in enumerate(text.splitlines(), 1):
            if not line.strip():
                continue
            try:
                parsed.append(json.loads(line))
            except json.JSONDecodeError:
                raise ValueError("%s line %d: not a JSON object"
                                 % (source, lineno)) from None
    if not parsed:
        raise ValueError("%s: no steps %s"
                         % (source, "in the file"
                            if source == "--file" else "on stdin"))
    steps = []
    for index, step in enumerate(parsed):
        where = "%s step %d" % (source, index + 1)
        if not isinstance(step, dict):
            raise ValueError("%s: must be a JSON object" % where)
        event_type = step.get("type")
        if (not event_type or not isinstance(event_type, str)
                or event_type in ("summary", "checkpoint")):
            raise ValueError("%s: type must name a step type, not "
                             "'summary' or 'checkpoint'" % where)
        payload = step.get("payload", {})
        if not isinstance(payload, dict):
            raise ValueError("%s: payload must be a JSON object" % where)
        tokens = step.get("tokens")
        if tokens is not None:
            try:
                tokens = _parse_record_tokens(json.dumps(tokens))
            except ValueError as exc:
                raise ValueError("%s: %s" % (where, exc)) from None
        ts = step.get("ts")
        if ts is not None:
            try:
                _normalize_ts(ts)
            except ValueError as exc:
                raise ValueError("%s: %s" % (where, exc)) from None
        steps.append({"type": event_type, "payload": payload,
                      "tokens": tokens, "ts": ts})
    return steps


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="blackbox",
        description="Flight recorder for AI agents: verify, replay, diff.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_verify = sub.add_parser("verify", help="check a log's hash chain")
    p_verify.add_argument("logfile")

    p_replay = sub.add_parser("replay", help="print a log as a transcript")
    p_replay.add_argument("logfile")

    p_diff = sub.add_parser("diff", help="compare two runs of the same task")
    p_diff.add_argument("log_a")
    p_diff.add_argument("log_b")

    p_mcp = sub.add_parser("mcp", help="serve blackbox over MCP on stdio")
    p_mcp.add_argument("logfile")
    p_mcp.add_argument("--task")
    p_mcp.add_argument("--model")

    p_record = sub.add_parser(
        "record", help="append one step to a log from the shell: the "
        "same record the Recorder writes — redacted, hash-chained, "
        "fsynced — so a step can be recorded without the MCP server "
        "or a Python script")
    p_record.add_argument("logfile")
    p_record.add_argument("--type",
                          help="step type, e.g. model_call, tool_call, "
                          "tool_result (required unless --stdin or "
                          "--file is given)")
    p_record.add_argument("--payload", default=None,
                          help="step payload as a JSON object "
                          "(default: {})")
    p_record.add_argument("--stdin", action="store_true",
                          help="read steps from stdin instead of the "
                          "flags: one JSON object per line, or a JSON "
                          "array of step objects — each step carries "
                          "its own type, payload, tokens and ts, so a "
                          "shell pipeline can log a whole run without "
                          "one process per step")
    p_record.add_argument("--file", default=None, metavar="PATH",
                          help="read steps from this file instead of "
                          "the flags: the same step JSON as --stdin — "
                          "one JSON object per line, or a JSON array "
                          "of step objects — for runs too big to pipe")
    p_record.add_argument("--tokens",
                          help="token count as a JSON int, or an "
                          "input/output split as a JSON object, e.g. "
                          "'{\"input\": 1200, \"output\": 340}'")
    p_record.add_argument("--ts",
                          help="ISO 8601 timestamp for the step "
                          "(default: now); must carry a timezone, "
                          "e.g. 2026-11-24T01:00:00Z")
    p_record.add_argument("--task",
                          help="task name for the session summary "
                          "(default: the log's existing task, if any)")
    p_record.add_argument("--model",
                          help="default model for the session summary "
                          "(default: the log's existing model, if any)")

    p_span = sub.add_parser(
        "span", help="print one step as an OTLP span: the same span "
        "the blackbox://session/span/{seq} MCP resource template "
        "serves, so the span is readable without an MCP client")
    p_span.add_argument("logfile")
    p_span.add_argument("seq", type=int, help="step number to print")
    p_span.add_argument("--json", action="store_true", dest="as_json",
                        help="print the span as JSON instead of a "
                        "report")

    p_watch = sub.add_parser(
        "watch", help="tail a live log and print new steps as they "
        "land, in the same format as replay: follow a running "
        "agent's flight recorder in real time")
    p_watch.add_argument("logfile")
    p_watch.add_argument("--from-start", action="store_true",
                         help="print the existing steps first, then "
                         "follow new ones (default: start at the end "
                         "and print only new steps)")
    p_watch.add_argument("--poll-interval", type=float, default=0.2,
                         help="seconds between checks for new steps "
                         "(default: %(default)s)")
    p_watch.add_argument("--type", action="append", dest="types",
                         help="step type to keep, e.g. tool_call "
                         "(repeatable; the summary is kept only when "
                         "--type summary is given)")
    p_watch.add_argument("--errors-only", action="store_true",
                         help="keep only steps that look like errors "
                         "(the same rule as query --errors-only)")
    p_watch.add_argument("--model",
                         help="keep only model_call steps for one "
                         "model, e.g. gpt-5 (case-insensitive; the "
                         "same rule as query --model)")
    p_watch.add_argument("--tool",
                         help="keep only tool_call / tool_result "
                         "steps for one tool, e.g. search "
                         "(case-insensitive; the same rule as "
                         "query --tool)")
    p_watch.add_argument("--min-tokens", type=int, default=None,
                         help="keep records with at least this many "
                         "tokens (the same rule as query --min-tokens)")
    p_watch.add_argument("--min-cost", type=float, default=None,
                         help="keep model_call records costing at least "
                         "this many USD (unpriced calls never match; "
                         "the same rule as query --min-cost)")
    p_watch.add_argument("--since", help="keep records at or after this "
                         "ISO timestamp (with a timezone, e.g. "
                         "2026-11-24T01:00:00Z; the same window as "
                         "query --since)")
    p_watch.add_argument("--until", help="keep records at or before this "
                         "ISO timestamp (with a timezone, e.g. "
                         "2026-11-24T01:00:00Z; the same window as "
                         "query --until)")
    p_watch.add_argument("--last", help="keep records from the last "
                         "this long, e.g. 10m, 2h, 1d, 1h30m "
                         "(cannot be combined with --since or --until)")

    p_otel = sub.add_parser("otel", help="export a log as OTLP/JSON spans")
    p_otel.add_argument("logfile")
    p_otel.add_argument("--out", help="write JSON here (default: stdout)")
    p_otel.add_argument("--service-name", default="blackbox")
    p_otel.add_argument("--causal", action="store_true",
                        help="parent spans from the causal trace "
                        "(tool_result -> tool_call, tool_call -> "
                        "model_call) instead of the hash chain; "
                        "same tree as `blackbox trace`")
    p_otel.add_argument("--last", help="keep steps from the last "
                        "this long, e.g. 10m, 2h, 1d, 1h30m "
                        "(the same relative window as query --last)")
    p_otel.add_argument("--since", help="keep steps at or after "
                        "this ISO timestamp (with a timezone, e.g. "
                        "2026-11-24T01:00:00Z; the same window as "
                        "query --since)")
    p_otel.add_argument("--until", help="keep steps at or before "
                        "this ISO timestamp (with a timezone, e.g. "
                        "2026-11-24T01:00:00Z; the same window as "
                        "query --until)")

    p_metrics = sub.add_parser(
        "metrics", help="export a log as OTLP/JSON metrics (per-model "
        "calls, tokens, latency, cost; per-tool calls)")
    p_metrics.add_argument("logfile")
    p_metrics.add_argument("--out", help="write JSON here (default: stdout)")
    p_metrics.add_argument("--service-name", default="blackbox")
    p_metrics.add_argument("--last", help="keep steps from the last "
                           "this long, e.g. 10m, 2h, 1d, 1h30m "
                           "(the same relative window as query --last)")
    p_metrics.add_argument("--since", help="keep steps at or after "
                           "this ISO timestamp (with a timezone, e.g. "
                           "2026-11-24T01:00:00Z; the same window as "
                           "query --since)")
    p_metrics.add_argument("--until", help="keep steps at or before "
                           "this ISO timestamp (with a timezone, e.g. "
                           "2026-11-24T01:00:00Z; the same window as "
                           "query --until)")

    p_tail = sub.add_parser(
        "tail-sample", help="export a log as tail-sampled OTLP/JSON "
        "spans: keep every span of a trace with an error, a slow span "
        "or high cost, sample the rest; the decision is recorded in "
        "the export summary")
    p_tail.add_argument("logfile")
    p_tail.add_argument("--out", help="write JSON here (default: stdout)")
    p_tail.add_argument("--service-name", default="blackbox")
    p_tail.add_argument("--keep-rate", type=float, default=0.5,
                        help="probability of keeping a span of a boring "
                        "trace (default: %(default)s)")
    p_tail.add_argument("--seed", type=int, default=None,
                        help="seed the keep/drop choice (default: random)")
    p_tail.add_argument("--slow-ms", type=float, default=DEFAULT_SLOW_MS,
                        help="a span slower than this keeps the whole "
                        "trace (default: %(default)s ms)")
    p_tail.add_argument("--max-cost-usd", type=float,
                        default=DEFAULT_MAX_COST_USD,
                        help="a trace costing more than this keeps the "
                        "whole trace (default: $%(default)s)")
    p_tail.add_argument("--causal", action="store_true",
                        help="keep a sampled span's causal ancestors "
                        "too, and take span parents from the causal "
                        "trace instead of the hash chain, so the "
                        "sampled trace stays connected (same tree as "
                        "`blackbox trace`)")

    p_export = sub.add_parser(
        "export", help="export a log as OTLP/JSON, to a file or live to a collector")
    p_export.add_argument("logfile")
    p_export.add_argument("--out", help="write JSON here (default: stdout)")
    p_export.add_argument("--service-name", default="blackbox")
    p_export.add_argument("--causal", action="store_true",
                          help="parent spans from the causal trace "
                          "instead of the hash chain (same tree as "
                          "`blackbox trace`); applies to the file "
                          "export and to --live")
    p_export.add_argument("--live", action="store_true",
                          help="POST the spans to an OTLP/HTTP collector")
    p_export.add_argument("--otlp-endpoint", default=DEFAULT_OTLP_ENDPOINT,
                          help="collector URL (default: %(default)s)")
    p_export.add_argument("--max-retries", type=int, default=3)
    p_export.add_argument("--timeout", type=float, default=10.0)
    p_export.add_argument("--last", help="keep steps from the last "
                          "this long, e.g. 10m, 2h, 1d, 1h30m "
                          "(the same relative window as query --last)")
    p_export.add_argument("--since", help="keep steps at or after "
                          "this ISO timestamp (with a timezone, e.g. "
                          "2026-11-24T01:00:00Z; the same window as "
                          "query --since)")
    p_export.add_argument("--until", help="keep steps at or before "
                          "this ISO timestamp (with a timezone, e.g. "
                          "2026-11-24T01:00:00Z; the same window as "
                          "query --until)")

    p_query = sub.add_parser(
        "query", help="print the steps of a log matching filters "
        "(--type, --model, --tool, --min-tokens, --min-cost, "
        "--errors-only, --since/--until, --last); filters combine "
        "with AND")
    p_query.add_argument("logfile")
    p_query.add_argument("--type", action="append", dest="types",
                         help="event type to keep (repeatable)")
    p_query.add_argument("--model", help="model_call records for this model")
    p_query.add_argument("--tool",
                         help="tool_call/tool_result records for this tool")
    p_query.add_argument("--min-tokens", type=int, default=None,
                         help="keep records with at least this many tokens")
    p_query.add_argument("--min-cost", type=float, default=None,
                         help="keep model_call records costing at least "
                         "this many USD (unpriced calls never match)")
    p_query.add_argument("--errors-only", action="store_true",
                         help="keep only steps that look like errors")
    p_query.add_argument("--since", help="keep records at or after this "
                         "ISO timestamp")
    p_query.add_argument("--until", help="keep records at or before this "
                         "ISO timestamp")
    p_query.add_argument("--last", help="keep records from the last "
                         "this long, e.g. 10m, 2h, 1d, 1h30m "
                         "(cannot be combined with --since or --until; "
                         "the same relative window as watch --last)")
    p_query.add_argument("--json", action="store_true", dest="as_json",
                         help="print each match as one raw JSON record "
                         "per line instead of a transcript")
    p_query.add_argument("--count", action="store_true",
                         help="print only the number of matches")

    p_trace = sub.add_parser(
        "trace", help="print a log as a causal parent-child tree: "
        "tool_results hang off their tool_calls, tool_calls off the "
        "model_call that decided them")
    p_trace.add_argument("logfile")
    p_trace.add_argument("--json", action="store_true", dest="as_json",
                         help="print the nested tree as JSON instead of "
                         "an indented tree")

    p_aggregates = sub.add_parser(
        "aggregates", help="print per-session totals: steps, duration, "
        "per-model tokens and cost, per-tool calls, errors, slowest step")
    p_aggregates.add_argument("logfile")
    p_aggregates.add_argument("--json", action="store_true", dest="as_json",
                              help="print the aggregates as JSON instead "
                              "of a report")
    p_aggregates.add_argument("--last", help="keep steps from the last "
                              "this long, e.g. 10m, 2h, 1d, 1h30m "
                              "(the same relative window as query --last)")
    p_aggregates.add_argument("--since", help="keep steps at or after "
                              "this ISO timestamp (with a timezone, e.g. "
                              "2026-11-24T01:00:00Z; the same window as "
                              "query --since)")
    p_aggregates.add_argument("--until", help="keep steps at or before "
                              "this ISO timestamp (with a timezone, e.g. "
                              "2026-11-24T01:00:00Z; the same window as "
                              "query --until)")

    p_costs = sub.add_parser(
        "costs", help="print cost totals broken down per model and per "
        "tool: the model spend that produced each tool call is charged "
        "to that tool")
    p_costs.add_argument("logfile")
    p_costs.add_argument("--json", action="store_true", dest="as_json",
                         help="print the cost totals as JSON instead "
                         "of a report")
    p_costs.add_argument("--last", help="keep steps from the last "
                         "this long, e.g. 10m, 2h, 1d, 1h30m "
                         "(the same relative window as query --last)")
    p_costs.add_argument("--since", help="keep steps at or after "
                         "this ISO timestamp (with a timezone, e.g. "
                         "2026-11-24T01:00:00Z; the same window as "
                         "query --since)")
    p_costs.add_argument("--until", help="keep steps at or before "
                         "this ISO timestamp (with a timezone, e.g. "
                         "2026-11-24T01:00:00Z; the same window as "
                         "query --until)")

    p_compact = sub.add_parser(
        "compact", help="retention: drop old records and anchor them "
        "with a hash-chain checkpoint, so verify still passes on the "
        "compacted log")
    p_compact.add_argument("logfile")
    p_compact.add_argument("--keep-days", type=float, default=None,
                           help="keep records from the last N days")
    p_compact.add_argument("--keep-n", type=int, default=None,
                           help="keep the newest N records")
    p_compact.add_argument("--dry-run", action="store_true",
                           help="report what would be removed without "
                           "writing anything")

    args = parser.parse_args(argv)
    if args.command == "verify":
        return verify_cli(args.logfile)
    if args.command == "replay":
        replay(args.logfile)
        return 0
    if args.command == "diff":
        diff(args.log_a, args.log_b)
        return 0
    if args.command == "mcp":
        serve_stdio(args.logfile, task=args.task, model=args.model)
        return 0
    if args.command == "record":
        steps = None
        if args.file is not None:
            # --file carries its own type/payload/tokens/ts per step,
            # so it must not be mixed with --stdin or the
            # single-step flags.
            if (args.stdin or args.type is not None
                    or args.payload is not None or args.tokens is not None
                    or args.ts is not None):
                print("record: --file cannot be combined with --stdin, "
                      "--type, --payload, --tokens or --ts — each step "
                      "carries its own", file=sys.stderr)
                return 2
            try:
                with open(args.file, encoding="utf-8") as f:
                    text = f.read()
            except OSError as exc:
                print("record: cannot read --file %s: %s"
                      % (args.file, exc.strerror or exc), file=sys.stderr)
                return 2
            try:
                steps = _parse_bulk_steps(text, "--file")
            except ValueError as exc:
                print("record: %s" % exc, file=sys.stderr)
                return 2
        elif args.stdin:
            # --stdin carries its own type/payload/tokens/ts per step,
            # so the single-step flags must not be mixed in.
            if (args.type is not None or args.payload is not None
                    or args.tokens is not None or args.ts is not None):
                print("record: --stdin cannot be combined with --type, "
                      "--payload, --tokens or --ts — each step carries "
                      "its own", file=sys.stderr)
                return 2
            try:
                steps = _parse_bulk_steps(sys.stdin.read(), "--stdin")
            except ValueError as exc:
                print("record: %s" % exc, file=sys.stderr)
                return 2
        if steps is not None:
            task, model = _record_task_model(args)
            try:
                with Recorder(args.logfile, task=task,
                              model=model) as rec:
                    seqs = [rec.record(step["type"], step["payload"],
                                       tokens=step["tokens"],
                                       ts=step["ts"])
                            for step in steps]
            except ValueError as exc:
                print("record: %s" % exc, file=sys.stderr)
                return 2
            seqs = [s for s in seqs if s is not None]
            if not seqs:
                print("recorded 0 steps (all sampled out)")
            elif len(seqs) == 1:
                print("recorded step %d (%s)"
                      % (seqs[0], steps[0]["type"]))
            else:
                print("recorded %d steps (seqs %d-%d)"
                      % (len(seqs), seqs[0], seqs[-1]))
            return 0
        if args.type is None:
            parser.error("--type is required unless --stdin or --file "
                         "is given")
        # Validate everything before opening the log: opening a
        # Recorder resumes it, which drops a trailing summary, so a
        # failed record must not get that far.
        if not args.type or args.type in ("summary", "checkpoint"):
            print("record: --type must name a step type, not 'summary' "
                  "or 'checkpoint'", file=sys.stderr)
            return 2
        try:
            payload = json.loads(args.payload
                                 if args.payload is not None else "{}")
        except json.JSONDecodeError:
            print("record: --payload must be a JSON object",
                  file=sys.stderr)
            return 2
        if not isinstance(payload, dict):
            print("record: --payload must be a JSON object",
                  file=sys.stderr)
            return 2
        try:
            tokens = (_parse_record_tokens(args.tokens)
                      if args.tokens is not None else None)
            if args.ts is not None:
                _normalize_ts(args.ts)
        except ValueError as exc:
            print("record: %s" % exc, file=sys.stderr)
            return 2
        task, model = _record_task_model(args)
        try:
            with Recorder(args.logfile, task=task, model=model) as rec:
                seq = rec.record(args.type, payload, tokens=tokens,
                                 ts=args.ts)
        except ValueError as exc:
            print("record: %s" % exc, file=sys.stderr)
            return 2
        print("recorded step %d (%s)" % (seq, args.type))
        return 0
    if args.command == "span":
        try:
            records = load_records(args.logfile)
        except FileNotFoundError:
            print("span: file not found: %s" % args.logfile,
                  file=sys.stderr)
            return 1
        try:
            span = session_span(records, args.seq)
        except ValueError as exc:
            print("span: %s" % exc, file=sys.stderr)
            return 2
        if span is None:
            print("span: no step %d in %s" % (args.seq, args.logfile),
                  file=sys.stderr)
            return 2
        if args.as_json:
            sys.stdout.write(json.dumps(span, indent=2) + "\n")
        else:
            sys.stdout.write(format_span(span))
        return 0
    if args.command == "watch":
        if args.poll_interval <= 0:
            print("watch: --poll-interval must be positive",
                  file=sys.stderr)
            return 2
        try:
            watch_log(args.logfile, from_start=args.from_start,
                      poll_interval=args.poll_interval,
                      types=args.types, errors_only=args.errors_only,
                      model=args.model, tool=args.tool,
                      min_tokens=args.min_tokens,
                      min_cost=args.min_cost,
                      since=args.since, until=args.until,
                      last=args.last)
        except KeyboardInterrupt:
            # Ctrl-C stops the tail cleanly, exit code 0.
            return 0
        except ValueError as exc:
            # A bad --since / --until / --last value, --since after
            # --until, or --last combined with --since / --until.
            print("watch: %s" % exc, file=sys.stderr)
            return 2
        except OSError as exc:
            print("watch: %s" % (exc.strerror or exc), file=sys.stderr)
            return 1
        return 0
    if args.command == "otel":
        try:
            data = export_otlp(args.logfile, args.service_name,
                               causal=args.causal, last=args.last,
                               since=args.since, until=args.until)
        except ValueError as exc:
            # A bad --last value, a bad --since / --until value,
            # --since after --until, or --last combined with --since /
            # --until.
            print("otel: %s" % exc, file=sys.stderr)
            return 2
        text = json.dumps(data, indent=2) + "\n"
        if args.out:
            with open(args.out, "w", encoding="utf-8") as f:
                f.write(text)
        else:
            sys.stdout.write(text)
        return 0
    if args.command == "metrics":
        try:
            data = export_otlp_metrics(args.logfile, args.service_name,
                                       last=args.last, since=args.since,
                                       until=args.until)
        except ValueError as exc:
            # A bad --last value, a bad --since / --until value,
            # --since after --until, or --last combined with --since /
            # --until.
            print("metrics: %s" % exc, file=sys.stderr)
            return 2
        text = json.dumps(data, indent=2) + "\n"
        if args.out:
            with open(args.out, "w", encoding="utf-8") as f:
                f.write(text)
        else:
            sys.stdout.write(text)
        return 0
    if args.command == "tail-sample":
        try:
            data = export_tail_sampled_otlp(
                args.logfile, keep_rate=args.keep_rate, seed=args.seed,
                slow_ms=args.slow_ms, max_cost_usd=args.max_cost_usd,
                service_name=args.service_name, causal=args.causal)
        except ValueError as exc:
            print("tail-sample: %s" % exc, file=sys.stderr)
            return 2
        text = json.dumps(data, indent=2) + "\n"
        if args.out:
            with open(args.out, "w", encoding="utf-8") as f:
                f.write(text)
            decision = data["tail_sampling"]
            reasons = ", ".join(decision["reasons"]) or "no trigger fired"
            extra = ""
            if decision["causal_ancestors_kept"]:
                extra = (", +%d causal ancestors"
                         % decision["causal_ancestors_kept"])
            if decision["parent_gaps"]:
                extra += (", %d parent gap%s"
                          % (decision["parent_gaps"],
                             "s" if decision["parent_gaps"] != 1 else ""))
            print("tail-sample: %s (%s) — kept %d of %d spans%s"
                  % (decision["decision"], reasons,
                     decision["kept_spans"], decision["total_spans"],
                     extra))
        else:
            sys.stdout.write(text)
        return 0
    if args.command == "export":
        try:
            if args.live:
                exporter = OtlpHttpExporter(
                    args.otlp_endpoint, timeout=args.timeout,
                    max_retries=args.max_retries,
                    service_name=args.service_name)
                ok = exporter.export_log(args.logfile, causal=args.causal,
                                         last=args.last, since=args.since,
                                         until=args.until)
                if ok:
                    print("exported %d spans to %s"
                          % (exporter.exported_spans, args.otlp_endpoint))
                    return 0
                print("export failed: %s" % exporter.last_error,
                      file=sys.stderr)
                return 1
            data = export_otlp(args.logfile, args.service_name,
                               causal=args.causal, last=args.last,
                               since=args.since, until=args.until)
        except ValueError as exc:
            # A bad --last value, a bad --since / --until value,
            # --since after --until, or --last combined with --since /
            # --until.
            print("export: %s" % exc, file=sys.stderr)
            return 2
        text = json.dumps(data, indent=2) + "\n"
        if args.out:
            with open(args.out, "w", encoding="utf-8") as f:
                f.write(text)
        else:
            sys.stdout.write(text)
        return 0
    if args.command == "query":
        try:
            query_log(args.logfile, types=args.types, model=args.model,
                      tool=args.tool, min_tokens=args.min_tokens,
                      min_cost=args.min_cost, errors_only=args.errors_only,
                      since=args.since, until=args.until, last=args.last,
                      as_json=args.as_json, count_only=args.count)
        except ValueError as exc:
            # A bad --last value, or --last combined with --since /
            # --until.
            print("query: %s" % exc, file=sys.stderr)
            return 2
        return 0
    if args.command == "trace":
        trace_log(args.logfile, as_json=args.as_json)
        return 0
    if args.command == "aggregates":
        try:
            aggregates_log(args.logfile, as_json=args.as_json,
                           last=args.last, since=args.since,
                           until=args.until)
        except ValueError as exc:
            # A bad --last value, a bad --since / --until value,
            # --since after --until, or --last combined with --since /
            # --until.
            print("aggregates: %s" % exc, file=sys.stderr)
            return 2
        return 0
    if args.command == "costs":
        try:
            cost_totals_log(args.logfile, as_json=args.as_json,
                            last=args.last, since=args.since,
                            until=args.until)
        except ValueError as exc:
            # A bad --last value, a bad --since / --until value,
            # --since after --until, or --last combined with --since /
            # --until.
            print("costs: %s" % exc, file=sys.stderr)
            return 2
        return 0
    if args.command == "compact":
        try:
            result = compact_log(args.logfile, keep_days=args.keep_days,
                                 keep_n=args.keep_n,
                                 dry_run=args.dry_run)
        except FileNotFoundError:
            print("compact: file not found: %s" % args.logfile,
                  file=sys.stderr)
            return 1
        except ValueError as exc:
            print("compact: %s" % exc, file=sys.stderr)
            return 2
        if not result["changed"]:
            print("compact: nothing to remove — kept %d records"
                  % result["kept"])
            return 0
        action = "would remove" if args.dry_run else "removed"
        print("compact: %s %d records, kept %d; checkpoint %s "
              "(head %s)"
              % (action, result["compacted"], result["kept"],
                 (result["checkpoint_hash"] or "n/a")[:12],
                 (result["head_hash"] or "")[:12]))
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
