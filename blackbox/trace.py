"""Causal span links: rebuild the parent-child trace of a session.

The log is a flat, hash-chained list of steps, in order. The OTLP
export keeps that shape: each span's parent is the previous record.
That is honest about the chain, but it hides the causal shape of an
agent loop, which is a tree:

    model_call ──> tool_call ──> tool_result ──> model_call ──> ...

`link_parents` reconstructs that tree from the steps, with three
rules, in order:

- a tool_result is the child of the most recent unmatched tool_call
  for the same tool (calls are matched first-in first-out, so two
  parallel calls to the same tool pair with their results in order);
- a tool_call is the child of the most recent model_call (the model
  that decided to call the tool);
- anything else is the child of the previous step.

A rule that finds nothing falls back to the previous step, and the
first step is a root — so an orphaned tool_result (no matching call
in the log, e.g. a resumed log or a truncated one) still lands
somewhere sensible instead of being dropped. The links are a
reconstruction, a best reading of the order, not a claim about what
the agent intended: the log itself records no parent pointers, and
a tool_result only names its tool, so a result for a tool called
twice in a row pairs with the earlier call by FIFO order.

`build_trace` returns the tree as nested dicts, one per step, each
with `seq`, `type`, a short `label` (model or tool name), and
`children`. `session_trace` is the same trace as a JSON-able dict —
step count, per-type counts, slowest and error steps, parent gaps,
and the tree — for the `blackbox://session/trace` MCP resource.
`trace_log` prints it, indented, or as JSON with `--json`. The
summary record is not a step and is never a node.
Works on .jsonl logs and .db stores.
"""

import json

from .replay import load_records


def _label(record):
    """Short label for a step: model or tool name where there is one."""
    payload = record.get("payload") or {}
    if record["type"] == "model_call" and payload.get("model"):
        return "model_call %s" % payload["model"]
    if record["type"] in ("tool_call", "tool_result") and payload.get("tool"):
        return "%s %s" % (record["type"], payload["tool"])
    return record["type"]


def link_parents(records):
    """Map each step's seq to its parent step's seq (None for a root).

    The summary record is excluded. See the module docstring for the
    rules and their limits.
    """
    steps = [r for r in records if r["type"] != "summary"]
    parents = {}
    # seqs of tool_calls per tool, in order, not yet matched to a result.
    pending_calls = {}
    last_model_call = None
    prev_seq = None
    for record in steps:
        seq = record["seq"]
        event_type = record["type"]
        payload = record.get("payload") or {}
        parent = None
        if event_type == "tool_result":
            tool = payload.get("tool")
            pending = pending_calls.get(tool) if tool else None
            if pending:
                parent = pending.pop(0)
            elif prev_seq is not None:
                parent = prev_seq
        elif event_type == "tool_call":
            if last_model_call is not None:
                parent = last_model_call
            elif prev_seq is not None:
                parent = prev_seq
            tool = payload.get("tool")
            if tool:
                pending_calls.setdefault(tool, []).append(seq)
        elif event_type == "model_call":
            # The model reacts to the previous step: usually a
            # tool_result it is reading, otherwise the step before it.
            if prev_seq is not None:
                parent = prev_seq
            last_model_call = seq
        elif prev_seq is not None:
            parent = prev_seq
        parents[seq] = parent
        prev_seq = seq
    return parents


def build_trace(records):
    """A session as a forest of nested step dicts, in log order.

    Each node is {"seq", "type", "label", "children"}; roots are the
    steps with no parent. Children are in log order.
    """
    steps = [r for r in records if r["type"] != "summary"]
    parents = link_parents(records)
    nodes = {}
    for record in steps:
        nodes[record["seq"]] = {
            "seq": record["seq"],
            "type": record["type"],
            "label": _label(record),
            "children": [],
        }
    roots = []
    for record in steps:
        node = nodes[record["seq"]]
        parent = parents[record["seq"]]
        if parent is None or parent not in nodes:
            roots.append(node)
        else:
            nodes[parent]["children"].append(node)
    return roots


def _tree_depth(node):
    """Depth of a trace-tree node: 1 for a leaf, more below."""
    if not node["children"]:
        return 1
    return 1 + max(_tree_depth(child) for child in node["children"])


def session_trace(records):
    """A session's reconstructed causal trace as JSON-able data.

    The same trace `blackbox trace` prints and the `summarize_trace`
    MCP prompt summarises, as a dict: step count, root count and max
    depth, per-type counts, the slowest steps (a step's latency is
    the time until the next step, in seconds), the error steps, the
    parent gaps under the tail-sample defaults (keep rate 0.5, seed
    0 — a kept span whose causal parent would be sampled out), and
    the nested parent-child tree from `build_trace`. The summary
    record is not a step and never appears. Read-only — it removes
    or changes nothing.
    """
    # Deferred: tail_sampling imports link_parents from this module,
    # so importing it at the top would be circular. These are the
    # same helpers the summarize_trace prompt uses, so the resource
    # and the prompt always report the same numbers.
    from .tail_sampling import (_is_error, _span_durations_ms,
                                tail_sample_records)
    steps = [r for r in records if r["type"] != "summary"]
    roots = build_trace(records)
    parents = link_parents(records)
    by_type = {}
    for record in steps:
        by_type[record["type"]] = by_type.get(record["type"], 0) + 1
    durations = _span_durations_ms(records)
    timed = sorted(
        ((d, r) for d, r in zip(durations, steps) if d is not None),
        key=lambda pair: pair[0], reverse=True)[:3]
    slowest_steps = [{"seq": record["seq"],
                      "type": record["type"],
                      "label": _label(record),
                      "latency_seconds": round(duration / 1000, 3)}
                     for duration, record in timed]
    error_steps = []
    for record in steps:
        if not _is_error(record):
            continue
        payload = record.get("payload") or {}
        error_steps.append({
            "seq": record["seq"],
            "type": record["type"],
            "label": _label(record),
            "error": (str(payload["error"]) if "error" in payload
                      else None),
        })
    kept, decision = tail_sample_records(
        records, keep_rate=0.5, seed=0)
    kept_seqs = {r["seq"] for r in kept}
    parent_gaps = [{"seq": record["seq"],
                    "parent_seq": parents[record["seq"]]}
                   for record in steps
                   if (record["seq"] in kept_seqs
                       and parents.get(record["seq"]) is not None
                       and parents[record["seq"]] not in kept_seqs)]
    return {"steps": len(steps),
            "roots": len(roots),
            "max_depth": max((_tree_depth(r) for r in roots),
                             default=0),
            "by_type": by_type,
            "slowest_steps": slowest_steps,
            "errors": len(error_steps),
            "error_seqs": [s["seq"] for s in error_steps],
            "error_steps": error_steps,
            "tail_sample_decision": decision["decision"],
            "tail_sample_reasons": decision["reasons"],
            "parent_gaps": parent_gaps,
            "tree": roots}


def _print_node(node, depth, lines):
    lines.append("%s[%d] %s" % ("  " * depth, node["seq"], node["label"]))
    for child in node["children"]:
        _print_node(child, depth + 1, lines)


def trace_log(path, as_json=False):
    """Print a log's causal trace: indented tree, or JSON with
    as_json. Works on .jsonl logs and .db stores."""
    records = load_records(path)
    roots = build_trace(records)
    if as_json:
        print(json.dumps(roots, indent=2))
        return
    lines = []
    for root in roots:
        _print_node(root, 0, lines)
    print("\n".join(lines))
