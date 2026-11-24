"""Tail-based sampling: decide what to keep after the full trace is known.

Head-based sampling (Recorder's sample_rate) drops steps as they are
recorded, before anyone knows whether the run mattered. Tail sampling
waits until the session is over, looks at the whole trace, and then
decides: a trace that contains an error, a slow span or a high cost
is interesting, so every span in it is kept. A boring trace is
sampled down at keep_rate, with a seeded RNG so the choice is
reproducible.

A blackbox session is one trace, so the decision is per session:
interesting sessions export whole, boring ones export a subset. The
decision is recorded in the export summary (the "tail_sampling" key
of export_tail_sampled_otlp), so an exported file is honest about
what it kept and why: the decision, the triggers that fired, the
rate, and how many spans were kept and dropped.

Spans keep their original ids: span ids come from the record hashes,
so a kept span still points at exactly the record it came from, and
the parent of a kept span still names the previous record in the
chain even when that record was dropped. Dropped spans are simply
absent from the export; nothing is rewritten.

Causal mode (`causal=True`, `blackbox tail-sample --causal`): a
sampled trace stays connected. When the RNG keeps a span, its
causal ancestors (from trace.link_parents over the log's steps)
are kept too, so a tool_result never dangles without the tool_call it
answers and a tool_call never dangles without the model_call that
decided it. The exported spans then also take their parentSpanId
from the same full-log causal tree (as `blackbox otel --causal`
does), so every non-root kept span's parent is a span in the
export. Ancestors pulled in this way are counted separately in the
decision (`causal_ancestors_kept`): the export stays honest about
which spans the RNG chose and which were added to keep the tree
connected. Kept-all traces need no closure — everything is kept
already — and the default (non-causal) export is unchanged.

Parent gaps (plain mode): a kept span whose chain parent was
dropped keeps its parentSpanId naming the real (absent) parent
and is marked with a `blackbox.parent_sampled_out` attribute (see
otel.records_to_spans), so the gap is visible instead of a silent
dangle. The decision counts them as `parent_gaps`: 0 for kept-all
traces and for causal sampling, where closure guarantees every
kept span's parent is in the export.

Time windows (`last` / `since` / `until`, `blackbox tail-sample
--last` / `--since` / `--until`): the export can be narrowed to a
window the same way the full export can, through the shared
`_window_records` from aggregates.py, so a bad bound, a reversed
window, or --last combined with --since / --until is an error.
The window applies before the tail decision, so the triggers are
judged on the windowed steps only — an error outside the window
does not keep the trace whole — and the summary and checkpoint
records are kept untouched, so the last kept step still runs to
the summary timestamp.
"""

import json
import random

from .aggregates import _window_records
from .costs import cost_for
from .otel import SERVICE_NAME, export_otlp, records_to_spans
from .query import _parse_record_ts
from .replay import load_records
from .trace import link_parents

DEFAULT_SLOW_MS = 5000
DEFAULT_MAX_COST_USD = 1.0


def _is_error(record):
    """A step that looks like an error: an error step, a tool result
    carrying an error, or a payload with status "error"."""
    if record["type"] == "error":
        return True
    payload = record.get("payload") or {}
    if "error" in payload:
        return True
    return payload.get("status") == "error"


def _span_durations_ms(records):
    """Each step's duration in ms: time until the next step (or the
    summary, for the last step). A step with no next timestamp has
    no duration. Returns a list aligned with the steps. The summary
    timestamp comes from the last summary record, not assumed to be
    the last record, and a step whose timestamp does not parse, or
    carries no timezone, contributes no duration instead of crashing
    the export."""
    steps = [r for r in records
             if r["type"] not in ("summary", "checkpoint")]
    summary_ts = None
    for record in reversed(records):
        if record["type"] == "summary":
            summary_ts = _parse_record_ts(record.get("ts"))
            break
    durations = []
    for i, record in enumerate(steps):
        if i + 1 < len(steps):
            end = _parse_record_ts(steps[i + 1].get("ts"))
        else:
            end = summary_ts
        start = _parse_record_ts(record.get("ts"))
        if end is None or start is None:
            durations.append(None)
        else:
            durations.append((end - start).total_seconds() * 1000)
    return durations


def _total_cost_usd(records):
    """The session's priced cost in USD, or None when no model call
    can be priced (no input/output split, or an unknown model)."""
    total = 0.0
    priced = False
    for record in records:
        if record["type"] != "model_call":
            continue
        tokens = record.get("tokens")
        if not isinstance(tokens, dict):
            continue
        model = (record.get("payload") or {}).get("model")
        cost = cost_for(model, tokens.get("input"), tokens.get("output"))
        if cost is not None:
            total += cost
            priced = True
    return round(total, 6) if priced else None


def tail_sample_records(records, keep_rate=0.5, seed=None,
                        slow_ms=DEFAULT_SLOW_MS,
                        max_cost_usd=DEFAULT_MAX_COST_USD, causal=False):
    """Decide which steps of a finished session to keep.

    Returns (kept_records, decision). The summary record, if any, is
    never a span and is not in kept_records; it is only used to time
    the last step. The decision dict records the outcome:

        decision: "kept_all" when a trigger fired, else "sampled"
        reasons: the triggers that fired ("error", "slow_span",
            "high_cost"), empty when the trace was sampled down
        keep_rate: the configured rate for boring traces
        total_spans / kept_spans / dropped_spans: the counts
        max_span_ms: the longest step duration, None with no timing
        total_cost_usd: the priced cost, None when nothing is priced
        causal: whether causal ancestor closure was on
        causal_ancestors_kept: spans added by that closure (0 when
            causal is off, or when the trace was kept whole)
        parent_gaps: kept spans whose parent was sampled out (0
            when the trace was kept whole, or when causal closure
            kept every kept span's parent)

    With causal=True and a sampled trace, a span the RNG keeps
    pulls in its causal ancestors (trace.link_parents over the
    log's steps, so a checkpoint never becomes a parent), so the
    kept spans form a connected tree instead of a set of
    orphans. The RNG choice itself is unchanged: the causal kept
    set is a superset of the plain kept set for the same seed.
    Without causal, a kept span whose chain parent was dropped is
    counted in parent_gaps and marked in the export with a
    blackbox.parent_sampled_out attribute (see records_to_spans),
    so the sampled trace never pretends to be connected when it
    is not.
    """
    if not 0 < keep_rate <= 1.0:
        raise ValueError("keep_rate must be in (0, 1]")
    # A checkpoint is bookkeeping about records compacted away, not
    # a step: it is not counted in total_spans, never kept or
    # dropped, and never ends a step's latency (its 09:00 timestamp
    # used to end the last step's span with a negative gap).
    steps = [r for r in records
             if r["type"] not in ("summary", "checkpoint")]
    durations = _span_durations_ms(records)
    total_cost = _total_cost_usd(records)
    timed = [d for d in durations if d is not None]
    max_span_ms = round(max(timed), 3) if timed else None

    reasons = []
    if any(_is_error(r) for r in steps):
        reasons.append("error")
    if max_span_ms is not None and max_span_ms > slow_ms:
        reasons.append("slow_span")
    if total_cost is not None and total_cost > max_cost_usd:
        reasons.append("high_cost")

    ancestors_kept = 0
    parent_gaps = 0
    if reasons:
        kept = list(steps)
        decision = "kept_all"
    else:
        rng = random.Random(seed)
        kept = [r for r in steps if rng.random() < keep_rate]
        decision = "sampled"
        if causal:
            # Parents over the log's steps only, so a checkpoint
            # never becomes a span's parent.
            parents = link_parents(steps)
            by_seq = {r["seq"]: r for r in steps}
            order = {r["seq"]: i for i, r in enumerate(steps)}
            kept_seqs = {r["seq"] for r in kept}
            added = []
            for record in kept:
                seq = parents.get(record["seq"])
                while seq is not None and seq not in kept_seqs:
                    kept_seqs.add(seq)
                    added.append(by_seq[seq])
                    seq = parents.get(seq)
            ancestors_kept = len(added)
            if added:
                kept = sorted(kept + added,
                              key=lambda r: order[r["seq"]])
        else:
            # Chain parent is the previous step: a kept span whose
            # predecessor was dropped is a parent gap. The first
            # step is a root and has no parent to lose.
            kept_seqs = {r["seq"] for r in kept}
            for i, record in enumerate(steps):
                if (i > 0 and record["seq"] in kept_seqs
                        and steps[i - 1]["seq"] not in kept_seqs):
                    parent_gaps += 1

    return kept, {
        "decision": decision,
        "reasons": reasons,
        "keep_rate": keep_rate,
        "total_spans": len(steps),
        "kept_spans": len(kept),
        "dropped_spans": len(steps) - len(kept),
        "max_span_ms": max_span_ms,
        "total_cost_usd": total_cost,
        "causal": causal,
        "causal_ancestors_kept": ancestors_kept,
        "parent_gaps": parent_gaps,
    }


def export_tail_sampled_otlp(path, keep_rate=0.5, seed=None,
                             slow_ms=DEFAULT_SLOW_MS,
                             max_cost_usd=DEFAULT_MAX_COST_USD,
                             service_name=SERVICE_NAME, causal=False,
                             last=None, since=None, until=None, now=None):
    """A log file as a tail-sampled OTLP/JSON dict.

    Works on .jsonl and .db logs. The shape is the same resourceSpans
    export as export_otlp, but with only the kept spans, plus a
    top-level "tail_sampling" key holding the decision dict, so the
    export records what was kept and why.

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
    The window is the shared `_window_records` from aggregates.py,
    so the summary and checkpoint records are kept untouched and a
    step whose timestamp does not parse, or carries no timezone,
    never matches. The window applies before the tail decision, so
    the error / slow-span / high-cost triggers are judged on the
    windowed steps only — an error outside the window does not keep
    the trace whole — and the last kept step runs to the summary
    timestamp, as in the unwindowed export.

    With causal=True, the kept spans are closed under their causal
    ancestors (see tail_sample_records) and their parentSpanIds come
    from the causal tree of the full log, so every non-root kept
    span's parent is a span in the export. The parents are taken
    from the full log on purpose: recomputing them from the sampled
    subset could pair a result with a different call than the full
    trace does. Without causal, spans keep chain parents as before,
    and a kept span whose chain parent was dropped is marked with
    a blackbox.parent_sampled_out attribute and counted as
    parent_gaps in the decision, so the gap is honest.
    """
    records = load_records(path)
    records = _window_records(records, last=last, since=since,
                              until=until, now=now)
    kept, decision = tail_sample_records(
        records, keep_rate=keep_rate, seed=seed, slow_ms=slow_ms,
        max_cost_usd=max_cost_usd, causal=causal)
    # The summary timestamp comes from the last summary record, not
    # assumed to be the last record: a checkpoint or a step can
    # follow it in the log.
    summary = None
    for record in reversed(records):
        if record["type"] == "summary":
            summary = record
            break
    parents = link_parents(records) if causal else None
    spans = records_to_spans(kept + ([summary] if summary else []),
                             service_name, causal=causal, parents=parents)
    data = export_otlp(path, service_name)
    data["resourceSpans"][0]["scopeSpans"][0]["spans"] = spans
    data["tail_sampling"] = decision
    return data


def export_tail_sampled_otlp_file(path, out_path, keep_rate=0.5,
                                  seed=None, slow_ms=DEFAULT_SLOW_MS,
                                  max_cost_usd=DEFAULT_MAX_COST_USD,
                                  service_name=SERVICE_NAME, causal=False,
                                  last=None, since=None, until=None,
                                  now=None):
    """Write a log's tail-sampled OTLP/JSON export to a file.
    Returns the dict. last / since / until narrow the export to a
    window, the same window as `export_tail_sampled_otlp`."""
    data = export_tail_sampled_otlp(
        path, keep_rate=keep_rate, seed=seed, slow_ms=slow_ms,
        max_cost_usd=max_cost_usd, service_name=service_name,
        causal=causal, last=last, since=since, until=until, now=now)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")
    return data
