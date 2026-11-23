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
cannot be re-aggregated.
"""

import json
from datetime import datetime

from .costs import cost_for
from .query import _is_error, _record_cost
from .recorder import _tokens_total
from .replay import load_records


def _parse_ts(ts):
    """An ISO timestamp to a datetime, or None if it does not parse."""
    if not isinstance(ts, str):
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        return None


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


def aggregates_log(path, as_json=False):
    """Compute a log's session aggregates and print them.

    --json prints the aggregates dict as JSON; otherwise a short
    human-readable report. Returns the aggregates dict."""
    records = load_records(path)
    agg = session_aggregates(records)
    if as_json:
        print(json.dumps(agg, indent=2, sort_keys=True))
        return agg
    print("session: %d steps" % agg["steps"], end="")
    if agg["duration_seconds"] is not None:
        print(" in %.3fs" % agg["duration_seconds"], end="")
    print("  (%s -> %s)" % (agg["started"], agg["ended"]))
    print("tokens: %d  cost: $%.6f (%d unpriced model calls)"
          % (agg["total_tokens"], agg["total_cost_usd"],
             agg["unpriced_model_calls"]))
    for event_type in sorted(agg["by_type"]):
        print("  %s: %d" % (event_type, agg["by_type"][event_type]))
    for tool in sorted(agg["by_tool"]):
        print("  tool %s: %d calls" % (tool, agg["by_tool"][tool]))
    for model in sorted(agg["by_model"]):
        entry = agg["by_model"][model]
        cost = entry["cost_usd"]
        cost_text = "$%.6f" % cost if cost is not None else "unpriced"
        print("  model %s: %d calls, %d in / %d out / %d tokens -> %s"
              % (model, entry["calls"], entry["input_tokens"],
                 entry["output_tokens"], entry["tokens"], cost_text))
    if agg["errors"]:
        print("errors: %d (seqs %s)"
              % (agg["errors"], ", ".join(str(s) for s in agg["error_seqs"])))
    if agg["slowest_step"] is not None:
        slowest = agg["slowest_step"]
        print("slowest step: seq %d (%s), %.3fs until the next step"
              % (slowest["seq"], slowest["type"],
                 slowest["latency_seconds"]))
    if agg["compacted_count"]:
        print("compacted away earlier: %d records (not in these totals)"
              % agg["compacted_count"])
    if agg["sampled_out"]:
        print("sampled out: %d steps (not in these totals)"
              % agg["sampled_out"])
    return agg
