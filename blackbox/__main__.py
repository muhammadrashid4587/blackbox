"""CLI: python -m blackbox verify|replay|diff|mcp|otel|metrics|tail-sample|export <logfile> ...

    python -m blackbox verify data/demo.jsonl
    python -m blackbox replay data/demo.jsonl
    python -m blackbox diff data/run-a.jsonl data/run-b.jsonl
    python -m blackbox mcp data/run.jsonl   # MCP server on stdio
    python -m blackbox otel data/run.jsonl --out spans.json
    python -m blackbox metrics data/run.jsonl --out metrics.json
    python -m blackbox tail-sample data/run.jsonl --out spans.json
    python -m blackbox export data/run.jsonl --live  # POST spans to a collector
    python -m blackbox query data/run.jsonl --model gpt-5 --min-cost 0.01

Any command takes a .jsonl log or a .db SQLite store.
"""

import argparse
import json
import sys

from .live_export import DEFAULT_OTLP_ENDPOINT, OtlpHttpExporter
from .mcp_server import serve_stdio
from .otel import export_otlp, export_otlp_metrics
from .query import query_log
from .replay import diff, replay
from .tail_sampling import (DEFAULT_MAX_COST_USD, DEFAULT_SLOW_MS,
                            export_tail_sampled_otlp)
from .verify import verify_cli


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

    p_otel = sub.add_parser("otel", help="export a log as OTLP/JSON spans")
    p_otel.add_argument("logfile")
    p_otel.add_argument("--out", help="write JSON here (default: stdout)")
    p_otel.add_argument("--service-name", default="blackbox")

    p_metrics = sub.add_parser(
        "metrics", help="export a log as OTLP/JSON metrics (per-model "
        "calls, tokens, latency, cost; per-tool calls)")
    p_metrics.add_argument("logfile")
    p_metrics.add_argument("--out", help="write JSON here (default: stdout)")
    p_metrics.add_argument("--service-name", default="blackbox")

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

    p_export = sub.add_parser(
        "export", help="export a log as OTLP/JSON, to a file or live to a collector")
    p_export.add_argument("logfile")
    p_export.add_argument("--out", help="write JSON here (default: stdout)")
    p_export.add_argument("--service-name", default="blackbox")
    p_export.add_argument("--live", action="store_true",
                          help="POST the spans to an OTLP/HTTP collector")
    p_export.add_argument("--otlp-endpoint", default=DEFAULT_OTLP_ENDPOINT,
                          help="collector URL (default: %(default)s)")
    p_export.add_argument("--max-retries", type=int, default=3)
    p_export.add_argument("--timeout", type=float, default=10.0)

    p_query = sub.add_parser(
        "query", help="print the steps of a log matching filters "
        "(--type, --model, --tool, --min-tokens, --min-cost, "
        "--errors-only, --since/--until); filters combine with AND")
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
    p_query.add_argument("--json", action="store_true", dest="as_json",
                         help="print each match as one raw JSON record "
                         "per line instead of a transcript")
    p_query.add_argument("--count", action="store_true",
                         help="print only the number of matches")

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
    if args.command == "otel":
        data = export_otlp(args.logfile, args.service_name)
        text = json.dumps(data, indent=2) + "\n"
        if args.out:
            with open(args.out, "w", encoding="utf-8") as f:
                f.write(text)
        else:
            sys.stdout.write(text)
        return 0
    if args.command == "metrics":
        data = export_otlp_metrics(args.logfile, args.service_name)
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
                service_name=args.service_name)
        except ValueError as exc:
            print("tail-sample: %s" % exc, file=sys.stderr)
            return 2
        text = json.dumps(data, indent=2) + "\n"
        if args.out:
            with open(args.out, "w", encoding="utf-8") as f:
                f.write(text)
            decision = data["tail_sampling"]
            reasons = ", ".join(decision["reasons"]) or "no trigger fired"
            print("tail-sample: %s (%s) — kept %d of %d spans"
                  % (decision["decision"], reasons,
                     decision["kept_spans"], decision["total_spans"]))
        else:
            sys.stdout.write(text)
        return 0
    if args.command == "export":
        if args.live:
            exporter = OtlpHttpExporter(
                args.otlp_endpoint, timeout=args.timeout,
                max_retries=args.max_retries,
                service_name=args.service_name)
            ok = exporter.export_log(args.logfile)
            if ok:
                print("exported %d spans to %s"
                      % (exporter.exported_spans, args.otlp_endpoint))
                return 0
            print("export failed: %s" % exporter.last_error,
                  file=sys.stderr)
            return 1
        data = export_otlp(args.logfile, args.service_name)
        text = json.dumps(data, indent=2) + "\n"
        if args.out:
            with open(args.out, "w", encoding="utf-8") as f:
                f.write(text)
        else:
            sys.stdout.write(text)
        return 0
    if args.command == "query":
        query_log(args.logfile, types=args.types, model=args.model,
                  tool=args.tool, min_tokens=args.min_tokens,
                  min_cost=args.min_cost, errors_only=args.errors_only,
                  since=args.since, until=args.until,
                  as_json=args.as_json, count_only=args.count)
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
