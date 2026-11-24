"""Session aggregates: per-session totals for one recorded run.

    python -m blackbox aggregates data/run.jsonl

A session is more than its steps in order: how long it ran, how many
tokens and dollars each model burned, which tools it used, and how
many steps look like errors. `session_aggregates` computes all of
that from the records, read-only — it removes or changes nothing —
so a later MCP prompt can hand the client the session at a glance
instead of the whole transcript. It works on .jsonl logs and .db
stores, and on compacted logs too: a checkpoint's compacted count
is reported separately, since the removed records are gone and
cannot be re-aggregated. --last keeps only steps from the last N
seconds/minutes/hours/days (e.g. 10m), the same relative window as
`blackbox query --last`, and --since / --until keep only steps in
an absolute window, the same window as `blackbox query
--since` / `--until`, so a finished session's totals can be
narrowed to the recent past or to an absolute window the same way
its steps can.
"""

import json
from datetime import datetime

from .costs import cost_for
from .query import (_is_error, _parse_bound, _parse_record_ts,
                    _record_cost, query_records)
from .recorder import _tokens_total
from .replay import load_records, model_usage
from .trace import link_parents


def _parse_ts(ts):
    """An ISO timestamp to a datetime, or None if it does not parse."""
    if not isinstance(ts, str):
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        return None


def _window_records(records, last=None, since=None, until=None,
                    now=None):
    """The records inside a --last relative window or a --since /
    --until absolute window.

    Steps are filtered through the shared `query_records` filter for
    --last, so the duration parsing and the timestamp strictness are
    the same as `blackbox query --last`: a value that is not a
    duration raises ValueError, and a step whose timestamp does not
    parse, or carries no timezone, never matches. --since / --until
    are parsed by the shared `_parse_bound`, the same way `blackbox
    watch` parses them, and compared as datetimes through
    `_parse_record_ts`, so a record stamped with fractional seconds
    compares correctly against a whole-second bound and a step whose
    timestamp does not parse, or carries no timezone, never matches.
    --last cannot be combined with --since or --until, because it
    sets the lower bound itself, and --since after --until raises
    ValueError, the same conflict rules as `blackbox query --last`.
    The summary and checkpoint records are kept untouched — they are
    bookkeeping about the whole log, so `sampled_out` and
    `compacted_count` still describe the log, not just the window —
    and the original order is preserved, so started/ended are still
    the window's first and last steps. With no window the records
    come back unchanged.
    """
    if last is not None and (since is not None or until is not None):
        raise ValueError("--last cannot be combined with --since or "
                         "--until — it sets the lower bound itself")
    if last is None and since is None and until is None:
        return records
    steps = [r for r in records
             if r["type"] not in ("summary", "checkpoint")]
    if last is not None:
        kept = query_records(steps, last=last, now=now)
    else:
        since_dt = (_parse_bound(since, "--since")
                    if since is not None else None)
        until_dt = (_parse_bound(until, "--until")
                    if until is not None else None)
        if (since_dt is not None and until_dt is not None
                and since_dt > until_dt):
            raise ValueError("--since must be at or before --until")
        kept = []
        for record in steps:
            ts = _parse_record_ts(record.get("ts"))
            if ts is None:
                continue
            if since_dt is not None and ts < since_dt:
                continue
            if until_dt is not None and ts > until_dt:
                continue
            kept.append(record)
    kept_ids = {id(record) for record in kept}
    return [r for r in records
            if r["type"] in ("summary", "checkpoint")
            or id(r) in kept_ids]


def session_aggregates(records):
    """Per-session totals for a list of records.

    Steps are every record except the summary and any checkpoint:
    the summary is bookkeeping about the run, the checkpoint
    certifies records that were compacted away. Returns a dict:
    step count, start/end timestamps, duration in seconds (None
    when the timestamps do not parse), per-type and per-tool counts,
    per-model calls/tokens/cost (cost None for a model with no known
    price or no input/output split, never a guessed number), total
    tokens and priced cost, error steps, the slowest step (time
    until the next step), and how many records an earlier compaction
    removed, if any.
    """
    steps = [r for r in records
             if r["type"] not in ("summary", "checkpoint")]
    by_type = {}
    by_tool = {}
    by_model = {}
    total_tokens = 0
    total_cost = 0.0
    unpriced_model_calls = 0
    error_seqs = []
    for record in steps:
        event_type = record["type"]
        by_type[event_type] = by_type.get(event_type, 0) + 1
        total_tokens += _tokens_total(record.get("tokens"))
        payload = record.get("payload") or {}
        if event_type == "tool_call" and payload.get("tool"):
            tool = payload["tool"]
            by_tool[tool] = by_tool.get(tool, 0) + 1
        if event_type == "model_call":
            model = payload.get("model") or "?"
            entry = by_model.setdefault(
                model, {"calls": 0, "input_tokens": 0,
                        "output_tokens": 0, "tokens": 0,
                        "cost_usd": None})
            entry["calls"] += 1
            tokens = record.get("tokens")
            if isinstance(tokens, dict):
                if isinstance(tokens.get("input"), int):
                    entry["input_tokens"] += tokens["input"]
                if isinstance(tokens.get("output"), int):
                    entry["output_tokens"] += tokens["output"]
            elif isinstance(tokens, int):
                entry["tokens"] += tokens
            cost = _record_cost(record)
            if cost is None:
                unpriced_model_calls += 1
            else:
                total_cost += cost
        if _is_error(record):
            error_seqs.append(record["seq"])
    for model, entry in by_model.items():
        if entry["input_tokens"] or entry["output_tokens"]:
            entry["cost_usd"] = cost_for(
                model, entry["input_tokens"], entry["output_tokens"])

    started = steps[0]["ts"] if steps else None
    ended = steps[-1]["ts"] if steps else None
    duration_seconds = None
    start_dt = _parse_ts(started)
    end_dt = _parse_ts(ended)
    if start_dt is not None and end_dt is not None:
        duration_seconds = round((end_dt - start_dt).total_seconds(), 3)

    slowest_step = None
    for i in range(len(steps) - 1):
        this_dt = _parse_ts(steps[i]["ts"])
        next_dt = _parse_ts(steps[i + 1]["ts"])
        if this_dt is None or next_dt is None:
            continue
        latency = round((next_dt - this_dt).total_seconds(), 3)
        if slowest_step is None or latency > slowest_step["latency_seconds"]:
            slowest_step = {"seq": steps[i]["seq"],
                            "type": steps[i]["type"],
                            "latency_seconds": latency}

    compacted = 0
    for record in records:
        if record["type"] == "checkpoint":
            compacted += (record.get("payload") or {}).get(
                "compacted_count") or 0
    sampled_out = 0
    for record in records:
        if record["type"] == "summary":
            sampled_out = (record.get("payload") or {}).get(
                "sampled_out") or 0

    return {"steps": len(steps),
            "started": started,
            "ended": ended,
            "duration_seconds": duration_seconds,
            "by_type": by_type,
            "by_tool": by_tool,
            "by_model": by_model,
            "total_tokens": total_tokens,
            "total_cost_usd": round(total_cost, 6),
            "unpriced_model_calls": unpriced_model_calls,
            "errors": len(error_seqs),
            "error_seqs": error_seqs,
            "slowest_step": slowest_step,
            "compacted_count": compacted,
            "sampled_out": sampled_out}


def cost_totals(records, default_model=None):
    """Cost totals broken down per model and per tool, as JSON-able data.

    Per model reuses `model_usage` from replay.py and `cost_for`, so
    it agrees with the cost summary, the diff prompt and the diff
    resource: a plain int token count has no input/output split and
    is not priced, and an unknown model costs None, never a guessed
    number. Per tool attributes a model call to the tool it decided
    to invoke, using the same causal rule as `link_parents` in
    trace.py — a tool_call hangs off the most recent model_call, so
    the model spend that produced a tool call is charged to that
    tool, in full. A model call invoking several tools is counted
    in full under each of them, so per-tool totals can exceed the
    session total; use per_model for totals that sum. A tool with
    no invoking model call (an orphaned tool_call, e.g. at the
    start of a resumed log) is still counted, with no tokens and
    cost None. Per-tool cost sums the priced attributions only and
    reports how many attributed calls were unpriced, the same
    pattern as the session total — cost None means no attributed
    call had a price. The summary record is not a step and is
    skipped. Read-only — it removes or changes nothing.
    """
    usage = model_usage(records, default_model)
    per_model = {}
    for model, entry in usage.items():
        cost = None
        if entry["input_tokens"] or entry["output_tokens"]:
            cost = cost_for(model, entry["input_tokens"],
                            entry["output_tokens"])
        per_model[model] = {"calls": 0,
                            "input_tokens": entry["input_tokens"],
                            "output_tokens": entry["output_tokens"],
                            "tokens": entry["tokens"],
                            "cost_usd": cost}
    # A model_call's children in the causal trace are the tool_calls
    # it decided to invoke: invert the parent map, keeping only
    # tool_call children.
    parents = link_parents(records)
    by_seq = {r["seq"]: r for r in records
              if r["type"] != "summary"}
    invoked = {}
    for record in records:
        if record["type"] != "tool_call":
            continue
        parent = parents.get(record["seq"])
        if parent is not None and parent in by_seq:
            invoked.setdefault(parent, []).append(record)
    per_tool = {}
    total_cost = 0.0
    unpriced_model_calls = 0
    for record in records:
        if record["type"] == "tool_call":
            tool = (record.get("payload") or {}).get("tool") or "?"
            per_tool.setdefault(
                tool, {"calls": 0, "input_tokens": 0,
                       "output_tokens": 0, "tokens": 0,
                       "cost_usd": None, "unpriced_calls": 0})
            per_tool[tool]["calls"] += 1
            continue
        if record["type"] != "model_call":
            continue
        model = ((record.get("payload") or {}).get("model")
                 or default_model)
        if not model or model not in per_model:
            continue
        per_model[model]["calls"] += 1
        # Price this call with the resolved model, not _record_cost:
        # a model_call with no model in its payload still counts
        # under default_model in per_model, so its cost must use
        # the same model here or the total would disagree.
        tokens = record.get("tokens")
        cost = None
        if isinstance(tokens, dict):
            cost = cost_for(model, tokens.get("input"),
                            tokens.get("output"))
        if cost is None:
            unpriced_model_calls += 1
        else:
            total_cost += cost
        for tool_call in invoked.get(record["seq"], []):
            tool = (tool_call.get("payload") or {}).get("tool") or "?"
            entry = per_tool.setdefault(
                tool, {"calls": 0, "input_tokens": 0,
                       "output_tokens": 0, "tokens": 0,
                       "cost_usd": None, "unpriced_calls": 0})
            if isinstance(tokens, dict):
                if isinstance(tokens.get("input"), int):
                    entry["input_tokens"] += tokens["input"]
                if isinstance(tokens.get("output"), int):
                    entry["output_tokens"] += tokens["output"]
            elif isinstance(tokens, int):
                entry["tokens"] += tokens
            if cost is None:
                entry["unpriced_calls"] += 1
            else:
                entry["cost_usd"] = round(
                    (entry["cost_usd"] or 0.0) + cost, 6)
    return {"per_model": per_model,
            "per_tool": per_tool,
            "total_cost_usd": round(total_cost, 6),
            "unpriced_model_calls": unpriced_model_calls}


def format_cost_totals(totals):
    """A cost totals dict as a short human-readable report.

    The same report `cost_totals_log` prints: the priced session
    total and how many model calls were unpriced, then the per-model
    and per-tool breakdowns. Returns the report as one string."""
    lines = []
    lines.append("cost totals: $%.6f priced (%d unpriced model calls)"
                 % (totals["total_cost_usd"],
                    totals["unpriced_model_calls"]))
    lines.append("per model:")
    for model in sorted(totals["per_model"]):
        entry = totals["per_model"][model]
        cost = entry["cost_usd"]
        cost_text = "$%.6f" % cost if cost is not None else "unpriced"
        lines.append("  %s: %d calls, %d in / %d out / %d tokens -> %s"
                     % (model, entry["calls"], entry["input_tokens"],
                        entry["output_tokens"], entry["tokens"],
                        cost_text))
    lines.append("per tool (cost of the model call that invoked it):")
    for tool in sorted(totals["per_tool"]):
        entry = totals["per_tool"][tool]
        cost = entry["cost_usd"]
        cost_text = "$%.6f" % cost if cost is not None else "unpriced"
        if entry["unpriced_calls"]:
            cost_text += " (%d unpriced)" % entry["unpriced_calls"]
        lines.append("  %s: %d calls, %d in / %d out / %d tokens -> %s"
                     % (tool, entry["calls"], entry["input_tokens"],
                        entry["output_tokens"], entry["tokens"],
                        cost_text))
    return "\n".join(lines)


def cost_totals_log(path, as_json=False, last=None, since=None,
                    until=None, now=None):
    """Compute a log's cost totals and print them.

    --json prints the cost totals dict as JSON; otherwise a short
    human-readable report. last is a relative window — a duration
    string like "10m" or a timedelta — keeping only steps from the
    last that long, as a lower bound of now minus the duration (now
    defaults to the current UTC time, and is a parameter so tests can
    fix it); a value that is not a duration raises ValueError.
    since / until are ISO 8601 timestamps with a timezone, keeping
    only steps in the closed window [since, until], the same window
    as `blackbox query --since` / `--until`; a value that does not
    parse or carries no timezone raises ValueError, as does --since
    after --until, and --last cannot be combined with --since or
    --until. The summary and checkpoint records are kept, so
    `sampled_out` and `compacted_count` still describe the whole log.
    Returns the cost totals dict."""
    records = load_records(path)
    records = _window_records(records, last=last, since=since,
                              until=until, now=now)
    totals = cost_totals(records)
    if as_json:
        print(json.dumps(totals, indent=2, sort_keys=True))
        return totals
    print(format_cost_totals(totals))
    return totals


def format_aggregates(agg):
    """A session aggregates dict as a short human-readable report.

    The same report `aggregates_log` prints: step count and
    duration, total tokens and priced cost, per-type and per-tool
    counts, per-model calls/tokens/cost, error steps, the slowest
    step, and how many records an earlier compaction removed or
    sampling dropped. Returns the report as one string."""
    lines = []
    first = "session: %d steps" % agg["steps"]
    if agg["duration_seconds"] is not None:
        first += " in %.3fs" % agg["duration_seconds"]
    first += "  (%s -> %s)" % (agg["started"], agg["ended"])
    lines.append(first)
    lines.append("tokens: %d  cost: $%.6f (%d unpriced model calls)"
                 % (agg["total_tokens"], agg["total_cost_usd"],
                    agg["unpriced_model_calls"]))
    for event_type in sorted(agg["by_type"]):
        lines.append("  %s: %d" % (event_type, agg["by_type"][event_type]))
    for tool in sorted(agg["by_tool"]):
        lines.append("  tool %s: %d calls" % (tool, agg["by_tool"][tool]))
    for model in sorted(agg["by_model"]):
        entry = agg["by_model"][model]
        cost = entry["cost_usd"]
        cost_text = "$%.6f" % cost if cost is not None else "unpriced"
        lines.append("  model %s: %d calls, %d in / %d out / %d tokens "
                     "-> %s"
                     % (model, entry["calls"], entry["input_tokens"],
                        entry["output_tokens"], entry["tokens"],
                        cost_text))
    if agg["errors"]:
        lines.append("errors: %d (seqs %s)"
                     % (agg["errors"],
                        ", ".join(str(s) for s in agg["error_seqs"])))
    if agg["slowest_step"] is not None:
        slowest = agg["slowest_step"]
        lines.append("slowest step: seq %d (%s), %.3fs until the next "
                     "step" % (slowest["seq"], slowest["type"],
                               slowest["latency_seconds"]))
    if agg["compacted_count"]:
        lines.append("compacted away earlier: %d records (not in these "
                     "totals)" % agg["compacted_count"])
    if agg["sampled_out"]:
        lines.append("sampled out: %d steps (not in these totals)"
                     % agg["sampled_out"])
    return "\n".join(lines)


def aggregates_log(path, as_json=False, last=None, since=None,
                   until=None, now=None):
    """Compute a log's session aggregates and print them.

    --json prints the aggregates dict as JSON; otherwise a short
    human-readable report. last is a relative window — a duration
    string like "10m" or a timedelta — keeping only steps from the
    last that long, as a lower bound of now minus the duration (now
    defaults to the current UTC time, and is a parameter so tests can
    fix it); a value that is not a duration raises ValueError.
    since / until are ISO 8601 timestamps with a timezone, keeping
    only steps in the closed window [since, until], the same window
    as `blackbox query --since` / `--until`; a value that does not
    parse or carries no timezone raises ValueError, as does --since
    after --until, and --last cannot be combined with --since or
    --until. The summary and checkpoint records are kept, so
    `sampled_out` and `compacted_count` still describe the whole log.
    Returns the aggregates dict."""
    records = load_records(path)
    records = _window_records(records, last=last, since=since,
                              until=until, now=now)
    agg = session_aggregates(records)
    if as_json:
        print(json.dumps(agg, indent=2, sort_keys=True))
        return agg
    print(format_aggregates(agg))
    return agg
