"""CLI: python -m blackbox verify|replay|diff|mcp <logfile> ...

    python -m blackbox verify data/demo.jsonl
    python -m blackbox replay data/demo.jsonl
    python -m blackbox diff data/run-a.jsonl data/run-b.jsonl
    python -m blackbox mcp data/run.jsonl   # MCP server on stdio

Any command takes a .jsonl log or a .db SQLite store.
"""

import argparse
import sys

from .mcp_server import serve_stdio
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
    return 2


if __name__ == "__main__":
    sys.exit(main())
