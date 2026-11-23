"""Query a recorded session: filter its steps and print the matches.

    python -m blackbox query data/run.jsonl --model gpt-5 --min-cost 0.01
    python -m blackbox query data/run.jsonl --tool search --errors-only
    python -m blackbox query data/run.jsonl --type model_call --json

Filters combine with AND: a record matches only if it passes every
filter given. The summary record is never a match unless
--type summary is given explicitly - a query is about steps, not the
summary. Cost is per model_call, from the published list prices in
costs.py: a call with a plain int token count (no input/output split)
or an unknown model has no cost, so it never matches --min-cost.
"""

import json

from .costs import cost_for
from .recorder import _tokens_total
from .replay import format_record, load_records


def _record_cost(record):
    """USD cost of one record, or None if it has no price.

    Only model_call records with an input/output token split and a
    model in the pricing table have a cost; everything else is None,
    never a guessed number."""
    if record["type"] != "model_call":
        return None
    tokens = record.get("tokens")
    if not isinstance(tokens, dict):
        return None
    model = (record.get("payload") or {}).get("model")
    if model is None:
        return None
    return cost_for(model, tokens.get("input"), tokens.get("output"))


def _is_error(record):
    """Whether a step looks like an error: an explicit error field, a
    status of "error", or a tool result that mentions an error."""
    payload = record.get("payload") or {}
    if payload.get("error"):
        return True
    if payload.get("status") == "error":
        return True
    result = payload.get("result")
    return isinstance(result, str) and "error" in result.lower()


def query_records(records, types=None, model=None, tool=None,
                  min_tokens=None, min_cost=None, errors_only=False,
                  since=None, until=None):
    """Return the records matching every filter given.

    types is a list of event types to keep; model matches model_call
    records by payload model (case-insensitive); tool matches
    tool_call/tool_result records by payload tool (case-insensitive);
    min_tokens keeps records with at least that many tokens;
    min_cost keeps model_call records costing at least that many USD;
    errors_only keeps steps that look like errors; since/until keep
    records whose ISO timestamp falls in [since, until]. The summary
    record is excluded unless "summary" is in types.
    """
    matched = []
    for record in records:
        if record["type"] == "summary" and not (types and "summary" in types):
            continue
        if types and record["type"] not in types:
            continue
        payload = record.get("payload") or {}
        if model is not None:
            if record["type"] != "model_call":
                continue
            if str(payload.get("model", "")).lower() != model.lower():
                continue
        if tool is not None:
            if record["type"] not in ("tool_call", "tool_result"):
                continue
            if str(payload.get("tool", "")).lower() != tool.lower():
                continue
        if min_tokens is not None and _tokens_total(record.get("tokens")) < min_tokens:
            continue
        if min_cost is not None:
            cost = _record_cost(record)
            if cost is None or cost < min_cost:
                continue
        if errors_only and not _is_error(record):
            continue
        if since is not None and record["ts"] < since:
            continue
        if until is not None and record["ts"] > until:
            continue
        matched.append(record)
    return matched


def query_log(path, types=None, model=None, tool=None, min_tokens=None,
              min_cost=None, errors_only=False, since=None, until=None,
              as_json=False, count_only=False):
    """Query a log file and print the matches. Returns the matched records.

    By default each match is printed as a transcript line, followed by
    a footer with the match count, token total and priced cost total.
    With as_json each match is printed as one raw JSON record per line
    (no footer). With count_only only the count is printed.
    """
    records = load_records(path)
    steps = [r for r in records if r["type"] != "summary"]
    matched = query_records(records, types=types, model=model, tool=tool,
                            min_tokens=min_tokens, min_cost=min_cost,
                            errors_only=errors_only, since=since, until=until)
    if count_only:
        print(len(matched))
        return matched
    if as_json:
        for record in matched:
            print(json.dumps(record, sort_keys=True))
        return matched
    for record in matched:
        print(format_record(record))
    total_tokens = sum(_tokens_total(r.get("tokens")) for r in matched)
    costs = [_record_cost(r) for r in matched]
    priced = [c for c in costs if c is not None]
    unpriced = sum(1 for r, c in zip(matched, costs)
                   if r["type"] == "model_call" and c is None)
    footer = "matched: %d of %d records, %d tokens" % (
        len(matched), len(steps), total_tokens)
    if priced:
        footer += ", $%.6f cost" % sum(priced)
    if unpriced:
        footer += " (%d unpriced)" % unpriced
    print(footer)
    return matched
