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

`replay` takes the same --since / --until / --last window as
`blackbox query` and `blackbox trace`, through the shared
`_window_records` — a transcript can be narrowed to a window the
same way the trace can. `diff` takes the same window too, applied
to both runs before they are compared, so two runs can be compared
over the same window.
"""

import json

from .costs import cost_for
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


def session_transcript_step(records, seq):
    """One step's formatted transcript line, or None if the session
    has no record with that seq.

    The line the blackbox://session/transcript/{seq} MCP resource
    template serves: `format_record` over the record — the same
    line the blackbox://session resource carries for that step
    and `blackbox replay` prints for it — so the template, the
    resource and the replay always agree. A model_call line
    includes its prompt line; a tool_call line its arguments;
    a tool_result line its result. Returns None when no record
    has that seq.
    """
    record = next((r for r in records if r["seq"] == seq), None)
    if record is None:
        return None
    return format_record(record)


def replay(path, last=None, since=None, until=None, now=None):
    """Print the transcript of a log. Returns the records.

    last is a relative window — a duration string like "10m" or a
    timedelta — keeping only steps from the last that long, as a
    lower bound of now minus the duration (now defaults to the
    current UTC time, and is a parameter so tests can fix it); a
    value that is not a duration raises ValueError. since / until
    are ISO 8601 timestamps with a timezone, keeping only steps in
    the closed window [since, until], the same window as `blackbox
    query --since` / `--until`; a value that does not parse or
    carries no timezone raises ValueError, as does --since after
    --until, and --last cannot be combined with --since or --until.
    The window is applied through the shared `_window_records`,
    before anything is printed, so a step whose timestamp does not
    parse, or carries no timezone, never matches, and the summary
    and checkpoint records are kept untouched — they are
    bookkeeping about the whole log, not just the window. Works on
    .jsonl logs and .db stores.
    """
    # Deferred: aggregates imports load_records from this module,
    # so importing it at the top would be circular.
    from .aggregates import _window_records
    records = _window_records(load_records(path), last=last,
                              since=since, until=until, now=now)
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


def model_usage(records, default_model=None):
    """Per-model token usage for a list of records.

    Same rules as the Recorder's summary: a plain int token count
    has no input/output split and is not priced, and an unknown
    model costs None, never a guessed number. A model_call with no
    model in its payload falls back to `default_model` (the
    session's model, when there is one). This is the same
    computation the MCP server's cost summary uses, so the prompt
    and the diff resource always agree on a model's tokens.
    """
    usage = {}
    for record in records:
        if record.get("type") != "model_call":
            continue
        model = ((record.get("payload") or {}).get("model")
                 or default_model)
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
    return usage


def _is_error(record):
    """A step that looks like an error: an error step, a tool result
    carrying an error, or a payload with status "error".

    Mirrors the same rule in query.py and tail_sampling.py (replay
    cannot import from either — both import this module — so the
    three-line rule lives here too)."""
    if record["type"] == "error":
        return True
    payload = record.get("payload") or {}
    if "error" in payload:
        return True
    return payload.get("status") == "error"


def session_diff(records_a, records_b, default_model_a=None,
                 default_model_b=None):
    """What changed between two runs, as JSON-able data.

    The same comparison `blackbox diff` prints and the
    `summarize_diff` MCP prompt summarises, as a dict: step counts
    and token totals for both runs, changed steps (same position,
    different type/payload/tokens, with both steps included),
    steps added in run B and removed from run A, per-model token
    and cost deltas (cost None where a model is unpriced in either
    run — a plain int token count has no input/output split, an
    unknown model has no published price, never a guessed number),
    and new vs fixed errors (an error run B has where run A had
    none, and one run A had that run B does not, compared by
    position). The summary record is skipped in both runs — it is
    bookkeeping, not a step. Read-only — it removes or changes
    nothing.
    """
    findings = diff_records(records_a, records_b)
    steps_a = [r for r in records_a if r["type"] != "summary"]
    steps_b = [r for r in records_b if r["type"] != "summary"]
    changed = [{"position": i, "step_a": steps_a[i],
                "step_b": steps_b[i]}
               for i in findings["changed"]]
    added = [steps_b[i] for i in findings["added"]]
    removed = [steps_a[i] for i in findings["removed"]]
    usage_a = model_usage(records_a, default_model_a)
    usage_b = model_usage(records_b, default_model_b)
    per_model = {}
    for model in sorted(set(usage_a) | set(usage_b)):
        entry_a = usage_a.get(model, {"input_tokens": 0,
                                      "output_tokens": 0,
                                      "tokens": 0})
        entry_b = usage_b.get(model, {"input_tokens": 0,
                                      "output_tokens": 0,
                                      "tokens": 0})
        tokens_a = (entry_a["input_tokens"]
                    + entry_a["output_tokens"] + entry_a["tokens"])
        tokens_b = (entry_b["input_tokens"]
                    + entry_b["output_tokens"] + entry_b["tokens"])
        cost_a = (cost_for(model, entry_a["input_tokens"],
                           entry_a["output_tokens"])
                  if entry_a["input_tokens"] or entry_a["output_tokens"]
                  else None)
        cost_b = (cost_for(model, entry_b["input_tokens"],
                           entry_b["output_tokens"])
                  if entry_b["input_tokens"] or entry_b["output_tokens"]
                  else None)
        cost_delta = (cost_b - cost_a
                      if cost_a is not None and cost_b is not None
                      else None)
        per_model[model] = {"tokens_a": tokens_a,
                            "tokens_b": tokens_b,
                            "tokens_delta": tokens_b - tokens_a,
                            "cost_usd_a": cost_a,
                            "cost_usd_b": cost_b,
                            "cost_usd_delta": cost_delta}
    new_errors = []
    fixed_errors = []
    for i in range(max(len(steps_a), len(steps_b))):
        error_a = i < len(steps_a) and _is_error(steps_a[i])
        error_b = i < len(steps_b) and _is_error(steps_b[i])
        if error_b and not error_a:
            new_errors.append(steps_b[i])
        elif error_a and not error_b:
            fixed_errors.append(steps_a[i])
    return {"run_a": {"steps": len(steps_a),
                      "tokens": findings["tokens_a"]},
            "run_b": {"steps": len(steps_b),
                      "tokens": findings["tokens_b"]},
            "tokens_diff": findings["tokens_b"] - findings["tokens_a"],
            "identical": (not findings["changed"]
                          and not findings["added"]
                          and not findings["removed"]),
            "changed": changed,
            "added": added,
            "removed": removed,
            "per_model": per_model,
            "new_errors": new_errors,
            "fixed_errors": fixed_errors,
            "errors_a": sum(1 for r in steps_a if _is_error(r)),
            "errors_b": sum(1 for r in steps_b if _is_error(r))}


def diff(path_a, path_b, last=None, since=None, until=None, now=None):
    """Compare two runs. Returns a dict of findings, and prints a report.

    last is a relative window — a duration string like "10m" or a
    timedelta — keeping only steps from the last that long, as a
    lower bound of now minus the duration (now defaults to the
    current UTC time, and is a parameter so tests can fix it); a
    value that is not a duration raises ValueError. since / until
    are ISO 8601 timestamps with a timezone, keeping only steps in
    the closed window [since, until], the same window as `blackbox
    query --since` / `--until`; a value that does not parse or
    carries no timezone raises ValueError, as does --since after
    --until, and --last cannot be combined with --since or --until.
    The window is applied to both runs through the shared
    `_window_records`, before the runs are compared position by
    position, so a step outside the window cannot make the runs
    look different, and a step whose timestamp does not parse, or
    carries no timezone, never matches. The summary and checkpoint
    records are kept untouched — they are bookkeeping about the
    whole log, not just the window. Works on .jsonl logs and .db
    stores.
    """
    # Deferred: aggregates imports load_records from this module,
    # so importing it at the top would be circular.
    from .aggregates import _window_records
    records_a = _window_records(load_records(path_a), last=last,
                                since=since, until=until, now=now)
    records_b = _window_records(load_records(path_b), last=last,
                                since=since, until=until, now=now)
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
