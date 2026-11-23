"""Replay a log as a readable transcript, and diff two runs.

Replay prints one line per step, in order, with the payload rendered
compactly so a run can be read back like a transcript:

    [0] 2026-11-21T...Z model_call  model=demo-model tokens=120
        prompt: Plan how to research flight recorders.
    [1] 2026-11-21T...Z tool_call  search(query="flight recorder AI agents")
    ...

Diff compares two runs of the same task, position by position, and
reports changed steps (same position, different payload), steps added
in the second run, and steps removed from the first. Token totals for
both runs are printed at the end.
"""

import json

from .recorder import _tokens_total
from .sqlite_store import is_sqlite_path, load_sqlite_records


def load_records(path):
    """Read a log into a list of record dicts, in order.

    Works for both backends: a .db/.sqlite/.sqlite3 path reads the
    SQLite store, anything else reads a .jsonl log.
    """
    if is_sqlite_path(path):
        return load_sqlite_records(path)
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def _short(value, limit=120):
    text = value if isinstance(value, str) else json.dumps(value)
    return text if len(text) <= limit else text[: limit - 3] + "..."


def format_record(record):
    """One record as human-readable text."""
    seq = record["seq"]
    ts = record["ts"]
    event_type = record["type"]
    payload = record.get("payload") or {}
    tokens = record.get("tokens")
    header = "[%d] %s %s" % (seq, ts, event_type)
    if event_type == "model_call":
        model = payload.get("model", "?")
        header += "  model=%s" % model
        if tokens is not None:
            header += " tokens=%s" % tokens
        lines = [header]
        if "prompt" in payload:
            lines.append("    prompt: %s" % _short(payload["prompt"]))
        return "\n".join(lines)
    if event_type == "tool_call":
        tool = payload.get("tool", "?")
        args = payload.get("args", {})
        args_text = ", ".join("%s=%s" % (k, _short(v, 60))
                              for k, v in args.items())
        return "%s  %s(%s)" % (header, tool, args_text)
    if event_type == "tool_result":
        tool = payload.get("tool", "?")
        result = payload.get("result", "")
        return "%s  %s -> %s" % (header, tool, _short(result))
    if event_type == "checkpoint":
        lines = [header]
        lines.append("    compacted %s records up to %s, kept %s"
                     % (payload.get("compacted_count"),
                        (payload.get("head_hash") or "")[:12],
                        payload.get("kept_count")))
        return "\n".join(lines)
    if event_type == "summary":
        lines = [header]
        lines.append("    events: %s  tokens: %s  %s -> %s" % (
            payload.get("events"), payload.get("total_tokens"),
            payload.get("started"), payload.get("ended")))
        if payload.get("cost_usd") is not None:
            lines.append("    cost: $%.6f" % payload["cost_usd"])
            for model, entry in sorted(
                    (payload.get("cost_by_model") or {}).items()):
                cost = entry.get("cost_usd")
                cost_text = "$%.6f" % cost if cost is not None else "unknown price"
                lines.append("      %s: %d in / %d out / %d tokens -> %s" % (
                    model, entry.get("input_tokens", 0),
                    entry.get("output_tokens", 0), entry.get("tokens", 0),
                    cost_text))
        return "\n".join(lines)
    # Any other event type: print the payload as compact JSON.
    return "%s  %s" % (header, _short(payload))


def replay(path):
    """Print the transcript of a log. Returns the records."""
    records = load_records(path)
    for record in records:
        print(format_record(record))
    return records


def _step_signature(record):
    """What identifies a step for diffing: its type, payload and tokens.
    Tokens are included: a run that spent twice the tokens is a
    different run, even if the payloads match."""
    return (record["type"],
            json.dumps(record.get("payload") or {}, sort_keys=True),
            json.dumps(record.get("tokens"), sort_keys=True))


def diff_records(records_a, records_b):
    """Compare two lists of records, position by position.

    The summary record is bookkeeping, not a step, so it is skipped
    in both runs. Returns the findings: the changed/added/removed
    step positions and the token totals of both runs. No printing:
    `diff()` prints the same findings, and the MCP summarize_diff
    prompt formats them into a prompt instead.
    """
    steps_a = [r for r in records_a if r["type"] != "summary"]
    steps_b = [r for r in records_b if r["type"] != "summary"]
    changed = []
    added = []
    removed = []
    for i in range(min(len(steps_a), len(steps_b))):
        sig_a = _step_signature(steps_a[i])
        sig_b = _step_signature(steps_b[i])
        if sig_a != sig_b:
            changed.append(i)
    if len(steps_b) > len(steps_a):
        added = list(range(len(steps_a), len(steps_b)))
    if len(steps_a) > len(steps_b):
        removed = list(range(len(steps_b), len(steps_a)))

    def _total(records):
        return sum(_tokens_total(r.get("tokens")) for r in records)

    return {"changed": changed, "added": added, "removed": removed,
            "tokens_a": _total(steps_a), "tokens_b": _total(steps_b)}


def diff(path_a, path_b):
    """Compare two runs. Returns a dict of findings, and prints a report."""
    records_a = load_records(path_a)
    records_b = load_records(path_b)
    findings = diff_records(records_a, records_b)
    steps_a = [r for r in records_a if r["type"] != "summary"]
    steps_b = [r for r in records_b if r["type"] != "summary"]
    changed = findings["changed"]
    added = findings["added"]
    removed = findings["removed"]
    tokens_a = findings["tokens_a"]
    tokens_b = findings["tokens_b"]

    for i in changed:
        print("CHANGED step %d:" % i)
        print("  A: %s" % format_record(steps_a[i]).replace("\n", "\n  "))
        print("  B: %s" % format_record(steps_b[i]).replace("\n", "\n  "))
    for i in added:
        print("ADDED step %d: %s" % (i, format_record(steps_b[i]).replace("\n", " ")))
    for i in removed:
        print("REMOVED step %d: %s" % (i, format_record(steps_a[i]).replace("\n", " ")))
    if not changed and not added and not removed:
        print("IDENTICAL: %d steps match" % len(steps_a))
    print("tokens: A=%d  B=%d  (diff %+d)" % (tokens_a, tokens_b, tokens_b - tokens_a))

    return {"changed": changed, "added": added, "removed": removed,
            "tokens_a": tokens_a, "tokens_b": tokens_b}
