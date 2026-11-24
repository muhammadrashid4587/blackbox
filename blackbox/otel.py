"""Export a recorded session as OpenTelemetry spans (OTLP/JSON).

Each recorded step becomes one span in a single trace:

- trace_id: the first 32 hex chars of the first record's hash, so every
  span in a session shares one trace, anchored in the hash chain.
- span_id: the first 16 hex chars of the record's own hash, so the
  span is tied to exactly one tamper-evident record.
- parent_span_id: the first 16 hex chars of the record's prev_hash,
  omitted for the first record (the root). The span tree is the
  chain, in order.
- start time: the record's timestamp. End time: the next step's
  timestamp (or the summary's, for the last step), so a span covers
  the time until the next thing the agent did; 1 ms if there is no
  next timestamp.
- attributes: the record's seq, type, hash, prev_hash, tokens, the
  full payload as JSON, plus type-specific ones (model and token
  split for model_call, tool name for tool_call/tool_result).

The output is the standard OTLP/JSON shape (resourceSpans /
scopeSpans / spans), so it can be handed to any collector that
accepts OTLP JSON, or inspected as plain JSON. No dependency: the
OpenTelemetry SDK is not required, the exporter only builds dicts.

The summary record is not exported as a span: it describes the run,
not a step. Round-tripping is lossless for the steps: spans_to_records
rebuilds each step's seq, timestamp, type, payload, tokens and hashes
from the span attributes, so an exported trace can be checked back
against the log it came from.

Single-step spans: `session_span` renders one step as a single span
— the span the blackbox://session/span/{seq} MCP resource template
serves — reusing `record_to_span`, with the step number as the span
id and the step kind as the span name, and a `blackbox.cost_usd`
attribute on a priced model_call (an input/output token split and a
known price), never a guessed cost.

Causal parents: `records_to_spans(..., causal=True)` (or
`blackbox otel --causal` / `blackbox export --causal`) sets each
span's parentSpanId from `trace.link_parents` instead of the hash
chain, so the exported trace reconstructs the same parent-child
tree as `blackbox trace`: a tool_result hangs off the tool_call it
answers, a tool_call off the model_call that decided it, a
model_call off the previous step. Spans in a causal export carry a
`blackbox.causal_parent` attribute, so the reconstruction is visible
in the export itself. The default export is unchanged (chain
parents). The limits are the same as the trace reconstruction's:
the links are inferred from step order, not recorded, so a
tool_result for a tool called twice in a row pairs with the earlier
call by FIFO order, and an orphaned result falls back to the
previous step.

Parent gaps: when spans are exported from a subset of the log (a
tail-sampled export), a kept span whose parent was dropped carries
a `blackbox.parent_sampled_out` attribute, so the gap is visible
instead of a silent dangle — its parentSpanId still names the real
dropped parent. A full log has no gaps, so a full export is
unchanged.

Metrics (records_to_metrics / export_otlp_metrics) are the same
session as OTLP Sum metrics: per-model call counts, input/output
token totals, latency (time until the next step, in ms) and cost in
USD where the model has a known price, plus per-tool call counts.
A step with no next timestamp (last step, no summary) contributes
no latency. Costs are estimates from costs.py, never guesses for
unknown models: an unpriced model simply has no cost data point.
"""

import json
from datetime import datetime, timedelta, timezone

from . import __version__
from .costs import cost_for
from .replay import load_records
from .trace import link_parents

SERVICE_NAME = "blackbox"
SCOPE_NAME = "blackbox"
_MIN_SPAN = timedelta(milliseconds=1)

# Sentinel for record_to_span: parent from the hash chain (the
# default), as opposed to an explicit span id or None (a root).
_CHAIN_PARENT = object()


def _parse_ts(ts):
    """Parse a blackbox ISO timestamp (with trailing Z) to datetime."""
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))


def _nanos(dt):
    """Unix nanoseconds, exact: float seconds * 1e9 loses microseconds."""
    epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
    delta = dt - epoch
    seconds = delta.days * 86400 + delta.seconds
    return str(seconds * 1_000_000_000 + delta.microseconds * 1000)


def _attr(key, value):
    """One OTLP attribute; int/bool/float/str map to their value kinds."""
    if isinstance(value, bool):
        return {"key": key, "value": {"boolValue": value}}
    if isinstance(value, float):
        return {"key": key, "value": {"doubleValue": value}}
    if isinstance(value, int):
        return {"key": key, "value": {"intValue": str(value)}}
    return {"key": key, "value": {"stringValue": str(value)}}


def _tokens_total(tokens):
    if tokens is None:
        return None
    if isinstance(tokens, dict):
        return sum(v for v in tokens.values() if isinstance(v, int))
    if isinstance(tokens, int):
        return tokens
    return None


def record_to_span(record, trace_id, end=None, parent_span_id=_CHAIN_PARENT,
                   causal=False, parent_sampled_out=False):
    """One record as an OTLP span dict. `end` is the span end time.

    `parent_span_id` overrides the chain parent: a span id string
    for an explicit parent, None for a root (no parentSpanId). Left
    at the default, the parent is the previous record's hash, as
    before. `causal` marks the span as carrying a reconstructed
    parent (see records_to_spans). `parent_sampled_out` marks a span
    whose parent was dropped by sampling: its parentSpanId still
    names the dropped parent (the id is real, the span is just not
    in this export), and a `blackbox.parent_sampled_out` attribute
    says so, so the gap is visible instead of a silent dangle.
    """
    payload = record.get("payload") or {}
    event_type = record["type"]
    start = _parse_ts(record["ts"])
    if end is None:
        end = start + _MIN_SPAN
    if event_type == "model_call":
        name = "model_call %s" % payload.get("model", "?")
    elif event_type in ("tool_call", "tool_result"):
        name = "%s %s" % (event_type, payload.get("tool", "?"))
    else:
        name = event_type

    attributes = [
        _attr("blackbox.seq", record["seq"]),
        _attr("blackbox.type", event_type),
        _attr("blackbox.hash", record["hash"]),
        _attr("blackbox.prev_hash", record["prev_hash"]),
        _attr("blackbox.payload", json.dumps(payload, sort_keys=True)),
    ]
    tokens = record.get("tokens")
    total = _tokens_total(tokens)
    if total is not None:
        attributes.append(_attr("blackbox.tokens", total))
    if isinstance(tokens, dict):
        if isinstance(tokens.get("input"), int):
            attributes.append(_attr("gen_ai.usage.input_tokens",
                                    tokens["input"]))
        if isinstance(tokens.get("output"), int):
            attributes.append(_attr("gen_ai.usage.output_tokens",
                                    tokens["output"]))
    if event_type == "model_call" and payload.get("model"):
        attributes.append(_attr("gen_ai.request.model", payload["model"]))
    if event_type in ("tool_call", "tool_result") and payload.get("tool"):
        attributes.append(_attr("tool.name", payload["tool"]))
    if causal:
        attributes.append(_attr("blackbox.causal_parent", True))
    if parent_sampled_out:
        attributes.append(_attr("blackbox.parent_sampled_out", True))

    span = {
        "traceId": trace_id,
        "spanId": record["hash"][:16],
        "name": name,
        "kind": 1,  # SPAN_KIND_INTERNAL
        "startTimeUnixNano": _nanos(start),
        "endTimeUnixNano": _nanos(end),
        "attributes": attributes,
        "status": {},
    }
    if parent_span_id is _CHAIN_PARENT:
        if record["prev_hash"] != "0" * 64:
            span["parentSpanId"] = record["prev_hash"][:16]
    elif parent_span_id is not None:
        span["parentSpanId"] = parent_span_id
    return span


def records_to_spans(records, service_name=SERVICE_NAME, causal=False,
                     parents=None):
    """A list of records as OTLP span dicts (summary excluded).

    With causal=True, each span's parent comes from
    trace.link_parents — the same parent-child tree `blackbox trace`
    prints — instead of the previous record in the hash chain. The
    default (causal=False) keeps the chain parents.

    `parents` is an optional seq -> parent-seq map computed from a
    fuller log than `records` (a tail-sampled export passes the
    full log's map): with causal=True it is used as-is instead of
    recomputing parents from the records given, so a kept span's
    parent is the parent the full trace assigns it. A parent seq
    absent from `records` still yields a root span, as before.

    A kept span whose parent is not in `records` — dropped by
    sampling, in either mode — is marked with a
    `blackbox.parent_sampled_out` attribute, so the export is
    honest about the gap: the parentSpanId still names the real
    (absent) parent instead of silently dangling. A full log has
    no gaps, so a full export is unchanged.
    """
    steps = [r for r in records if r["type"] != "summary"]
    if not steps:
        return []
    trace_id = steps[0]["hash"][:32]
    if causal and parents is None:
        parents = link_parents(records)
    by_seq = {r["seq"]: r for r in steps} if causal else None
    kept_ids = {r["hash"][:16] for r in steps}
    spans = []
    for i, record in enumerate(steps):
        if i + 1 < len(steps):
            end = _parse_ts(steps[i + 1]["ts"])
        elif records and records[-1]["type"] == "summary":
            end = _parse_ts(records[-1]["ts"])
        else:
            end = None
        if causal:
            parent_seq = parents.get(record["seq"])
            parent_span_id = (by_seq[parent_seq]["hash"][:16]
                              if parent_seq is not None
                              and parent_seq in by_seq else None)
            spans.append(record_to_span(
                record, trace_id, end=end,
                parent_span_id=parent_span_id, causal=True,
                parent_sampled_out=(parent_seq is not None
                                    and parent_seq not in by_seq)))
        else:
            gap = (record["prev_hash"] != "0" * 64
                   and record["prev_hash"][:16] not in kept_ids)
            spans.append(record_to_span(record, trace_id, end=end,
                                        parent_sampled_out=gap))
    return spans


def session_span(records, seq):
    """One step as a single OTLP span dict, or None if the session has
    no step with that seq.

    The span for the blackbox://session/span/{seq} MCP resource
    template: one recorded step rendered as a single OTLP span, so a
    client can read one step as a span without exporting the whole
    trace. It reuses `record_to_span` — the same span builder the
    full `blackbox otel` export uses — so the span agrees with the
    export, with the transcript (`format_record` over the same
    record) and with the raw record
    (`blackbox://session/record/{seq}`).

    Two things differ from the full export, both by design: the span
    id is the step number itself (16 hex chars, zero-padded), so the
    span is addressable by seq, and the span name is the step kind
    alone (e.g. "model_call"), not the export's "model_call gpt-5"
    label. The span still carries the record's hash in
    `blackbox.hash`, so it ties to exactly one tamper-evident
    record, and its parentSpanId is the chain parent, as in the
    export.

    Start time is the record's timestamp; end time is the next
    step's timestamp, or the summary's for the last step — the same
    rule as `records_to_spans`. A step whose timestamp is missing
    or unparseable raises ValueError: a span with no honest start
    or end time is not a span, and times are never invented. A
    model_call with an input/output token split and a known price
    carries a `blackbox.cost_usd` attribute; an unpriced call
    carries none — cost None is never guessed.
    """
    steps = [r for r in records if r["type"] != "summary"]
    record = next((r for r in steps if r["seq"] == seq), None)
    if record is None:
        return None
    if "ts" not in record:
        raise ValueError("step %d has no timestamp" % seq)
    try:
        _parse_ts(record["ts"])
    except ValueError:
        raise ValueError("step %d has no usable timestamp" % seq)
    index = steps.index(record)
    try:
        if index + 1 < len(steps):
            end = _parse_ts(steps[index + 1]["ts"])
        elif records and records[-1]["type"] == "summary":
            end = _parse_ts(records[-1]["ts"])
        else:
            end = None
    except (KeyError, ValueError):
        raise ValueError("step %d has no usable end timestamp" % seq)
    span = record_to_span(record, steps[0]["hash"][:32], end=end)
    span["spanId"] = "%016x" % seq
    span["name"] = record["type"]
    payload = record.get("payload") or {}
    if record["type"] == "model_call":
        tokens = record.get("tokens")
        if isinstance(tokens, dict):
            cost = cost_for(payload.get("model"),
                            tokens.get("input"), tokens.get("output"))
            if cost is not None:
                span["attributes"].append(
                    _attr("blackbox.cost_usd", cost))
    return span


def _iso_from_nanos(nanos):
    """Unix nanoseconds back to a blackbox ISO timestamp (trailing Z)."""
    seconds, ns = divmod(int(nanos), 1_000_000_000)
    dt = (datetime.fromtimestamp(seconds, timezone.utc)
          + timedelta(microseconds=ns // 1000))
    return dt.isoformat().replace("+00:00", "Z")


def format_span(span):
    """One OTLP span dict as readable lines: name and span id, trace
    and parent ids, start/end times and duration, then one
    `key: value` line per attribute. The human-readable form of the
    `blackbox span` CLI command; `--json` prints the dict instead.
    """
    lines = ["%s (span %s)" % (span["name"], span["spanId"])]
    lines.append("  trace: %s" % span["traceId"])
    if "parentSpanId" in span:
        lines.append("  parent: %s" % span["parentSpanId"])
    start = int(span["startTimeUnixNano"])
    end = int(span["endTimeUnixNano"])
    lines.append("  start: %s" % _iso_from_nanos(start))
    lines.append("  end: %s" % _iso_from_nanos(end))
    lines.append("  duration: %.3f s" % ((end - start) / 1e9))
    for attr in span.get("attributes", []):
        value = attr["value"]
        if "intValue" in value:
            rendered = value["intValue"]
        elif "doubleValue" in value:
            rendered = repr(value["doubleValue"])
        elif "boolValue" in value:
            rendered = "true" if value["boolValue"] else "false"
        else:
            rendered = value.get("stringValue", "")
        lines.append("  %s: %s" % (attr["key"], rendered))
    return "\n".join(lines) + "\n"


def _sum_metric(name, points, unit=None, monotonic=True):
    """One OTLP Sum metric from (attributes, value) points.

    int values become asInt, float values asDouble. Cumulative
    temporality: the numbers cover the whole recorded session.
    """
    data_points = []
    for attrs, value in points:
        point = {"attributes": [_attr(k, v) for k, v in attrs]}
        if isinstance(value, float):
            point["asDouble"] = value
        else:
            point["asInt"] = str(value)
        data_points.append(point)
    metric = {
        "name": name,
        "sum": {
            "dataPoints": data_points,
            "aggregationTemporality": 2,  # AGGREGATION_TEMPORALITY_CUMULATIVE
            "isMonotonic": monotonic,
        },
    }
    if unit:
        metric["unit"] = unit
    return metric


def records_to_metrics(records, service_name=SERVICE_NAME):
    """A list of records as OTLP Sum metric dicts (summary excluded).

    One metric per signal, one data point per model or tool. An
    empty session (no steps) yields no metrics.
    """
    steps = [r for r in records if r["type"] != "summary"]
    if not steps:
        return []
    summary_ts = None
    if records and records[-1]["type"] == "summary":
        summary_ts = _parse_ts(records[-1]["ts"])

    calls = {}
    input_tokens = {}
    output_tokens = {}
    latency_ms = {}
    cost_usd = {}
    tool_calls = {}
    for i, record in enumerate(steps):
        payload = record.get("payload") or {}
        if record["type"] == "model_call":
            model = payload.get("model", "?")
            calls[model] = calls.get(model, 0) + 1
            tokens = record.get("tokens")
            if isinstance(tokens, dict):
                if isinstance(tokens.get("input"), int):
                    input_tokens[model] = (input_tokens.get(model, 0)
                                           + tokens["input"])
                if isinstance(tokens.get("output"), int):
                    output_tokens[model] = (output_tokens.get(model, 0)
                                            + tokens["output"])
                cost = cost_for(model, tokens.get("input"),
                                tokens.get("output"))
                if cost is not None:
                    cost_usd[model] = round(
                        cost_usd.get(model, 0.0) + cost, 6)
            if i + 1 < len(steps):
                end = _parse_ts(steps[i + 1]["ts"])
            else:
                end = summary_ts
            if end is not None:
                start = _parse_ts(record["ts"])
                ms = (end - start).total_seconds() * 1000
                latency_ms[model] = latency_ms.get(model, 0.0) + ms
        elif record["type"] == "tool_call":
            tool = payload.get("tool", "?")
            tool_calls[tool] = tool_calls.get(tool, 0) + 1

    metrics = []
    if calls:
        def model_attrs(m):
            return [("gen_ai.request.model", m)]

        metrics.append(_sum_metric(
            "blackbox.model_calls",
            [(model_attrs(m), n) for m, n in sorted(calls.items())],
            unit="1"))
        if input_tokens:
            metrics.append(_sum_metric(
                "blackbox.model_input_tokens",
                [(model_attrs(m), n)
                 for m, n in sorted(input_tokens.items())], unit="1"))
        if output_tokens:
            metrics.append(_sum_metric(
                "blackbox.model_output_tokens",
                [(model_attrs(m), n)
                 for m, n in sorted(output_tokens.items())], unit="1"))
        if cost_usd:
            metrics.append(_sum_metric(
                "blackbox.model_cost_usd",
                [(model_attrs(m), c)
                 for m, c in sorted(cost_usd.items())], unit="USD"))
        if latency_ms:
            metrics.append(_sum_metric(
                "blackbox.model_latency_ms",
                [(model_attrs(m), round(ms, 3))
                 for m, ms in sorted(latency_ms.items())],
                unit="ms", monotonic=False))
    if tool_calls:
        metrics.append(_sum_metric(
            "blackbox.tool_calls",
            [( [("tool.name", t)], n)
             for t, n in sorted(tool_calls.items())], unit="1"))
    return metrics


def export_otlp_metrics(path, service_name=SERVICE_NAME):
    """A log file as an OTLP/JSON metrics dict. Works on .jsonl and
    .db logs. The shape is resourceMetrics / scopeMetrics / metrics,
    so it can be posted to the same collector as the spans."""
    records = load_records(path)
    return {
        "resourceMetrics": [{
            "resource": {"attributes": [
                _attr("service.name", service_name),
            ]},
            "scopeMetrics": [{
                "scope": {"name": SCOPE_NAME, "version": __version__},
                "metrics": records_to_metrics(records, service_name),
            }],
        }],
    }


def export_otlp_metrics_file(path, out_path, service_name=SERVICE_NAME):
    """Write a log's OTLP/JSON metrics export to a file. Returns the
    dict."""
    data = export_otlp_metrics(path, service_name)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")
    return data


def export_otlp(path, service_name=SERVICE_NAME, causal=False):
    """A log file as an OTLP/JSON dict. Works on .jsonl and .db logs.

    With causal=True, span parents come from the causal trace
    reconstruction instead of the hash chain (see records_to_spans).
    """
    records = load_records(path)
    return {
        "resourceSpans": [{
            "resource": {"attributes": [
                _attr("service.name", service_name),
            ]},
            "scopeSpans": [{
                "scope": {"name": SCOPE_NAME, "version": __version__},
                "spans": records_to_spans(records, service_name,
                                          causal=causal),
            }],
        }],
    }


def export_otlp_file(path, out_path, service_name=SERVICE_NAME,
                     causal=False):
    """Write a log's OTLP/JSON export to a file. Returns the dict."""
    data = export_otlp(path, service_name, causal=causal)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")
    return data


def spans_to_records(spans):
    """Rebuild the recorded steps from exported span dicts.

    The inverse of records_to_spans: each span's attributes carry the
    record's seq, timestamp, type, payload, tokens and hashes, so the
    steps come back exactly. Spans are returned in order.
    """
    records = []
    for span in spans:
        attrs = {}
        for attr in span.get("attributes", []):
            value = attr["value"]
            if "intValue" in value:
                attrs[attr["key"]] = int(value["intValue"])
            elif "doubleValue" in value:
                attrs[attr["key"]] = value["doubleValue"]
            elif "boolValue" in value:
                attrs[attr["key"]] = value["boolValue"]
            else:
                attrs[attr["key"]] = value.get("stringValue", "")
        start = int(span["startTimeUnixNano"])
        ts = (datetime.fromtimestamp(start // 1_000_000_000, timezone.utc)
              + timedelta(microseconds=(start % 1_000_000_000) // 1000))
        ts = ts.isoformat().replace("+00:00", "Z")
        tokens = attrs.get("blackbox.tokens")
        if ("gen_ai.usage.input_tokens" in attrs
                or "gen_ai.usage.output_tokens" in attrs):
            # A model_call recorded with a tokens dict: rebuild the split.
            tokens = {}
            if "gen_ai.usage.input_tokens" in attrs:
                tokens["input"] = attrs["gen_ai.usage.input_tokens"]
            if "gen_ai.usage.output_tokens" in attrs:
                tokens["output"] = attrs["gen_ai.usage.output_tokens"]
        records.append({
            "seq": attrs["blackbox.seq"],
            "ts": ts,
            "type": attrs["blackbox.type"],
            "payload": json.loads(attrs["blackbox.payload"]),
            "tokens": tokens,
            "prev_hash": attrs["blackbox.prev_hash"],
            "hash": attrs["blackbox.hash"],
        })
    return records
