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

SERVICE_NAME = "blackbox"
SCOPE_NAME = "blackbox"
_MIN_SPAN = timedelta(milliseconds=1)


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
    """One OTLP attribute; int/bool/str map to their value kinds."""
    if isinstance(value, bool):
        return {"key": key, "value": {"boolValue": value}}
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


def record_to_span(record, trace_id, end=None):
    """One record as an OTLP span dict. `end` is the span end time."""
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
    if record["prev_hash"] != "0" * 64:
        span["parentSpanId"] = record["prev_hash"][:16]
    return span


def records_to_spans(records, service_name=SERVICE_NAME):
    """A list of records as OTLP span dicts (summary excluded)."""
    steps = [r for r in records if r["type"] != "summary"]
    if not steps:
        return []
    trace_id = steps[0]["hash"][:32]
    spans = []
    for i, record in enumerate(steps):
        if i + 1 < len(steps):
            end = _parse_ts(steps[i + 1]["ts"])
        elif records and records[-1]["type"] == "summary":
            end = _parse_ts(records[-1]["ts"])
        else:
            end = None
        spans.append(record_to_span(record, trace_id, end=end))
    return spans


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


def export_otlp(path, service_name=SERVICE_NAME):
    """A log file as an OTLP/JSON dict. Works on .jsonl and .db logs."""
    records = load_records(path)
    return {
        "resourceSpans": [{
            "resource": {"attributes": [
                _attr("service.name", service_name),
            ]},
            "scopeSpans": [{
                "scope": {"name": SCOPE_NAME, "version": __version__},
                "spans": records_to_spans(records, service_name),
            }],
        }],
    }


def export_otlp_file(path, out_path, service_name=SERVICE_NAME):
    """Write a log's OTLP/JSON export to a file. Returns the dict."""
    data = export_otlp(path, service_name)
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
