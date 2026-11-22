"""CLI: python -m blackbox verify|replay|diff|mcp|otel <logfile> ...

    python -m blackbox verify data/demo.jsonl
    python -m blackbox replay data/demo.jsonl
    python -m blackbox diff data/run-a.jsonl data/run-b.jsonl
    python -m blackbox mcp data/run.jsonl   # MCP server on stdio
    python -m blackbox otel data/run.jsonl --out spans.json

Any command takes a .jsonl log or a .db SQLite store.
"""

import argparse
import json
import sys

from .mcp_server import serve_stdio
from .otel import export_otlp
from .replay import diff, replay
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
    return 2


if __name__ == "__main__":
    sys.exit(main())
