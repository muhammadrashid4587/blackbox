# blackbox

A flight recorder for AI agents.

Every step an agent takes — model calls, tool calls with arguments, tool
results, timestamps, token counts — is written to a tamper-evident
append-only log you can verify and replay.

## Why

When an agent does something wrong, the question is always the same:
what exactly did it do, in what order, with what inputs? Chat
transcripts don't answer that — they are a summary, not a record.
blackbox keeps the record: one JSON line per step, hash-chained so any
later edit is detectable, with secrets redacted before anything is
written.

It is a small local tool, not a platform. It has no server, no account,
no dependencies — pure Python 3 standard library. The optional SQLite
backend is the stdlib `sqlite3` module, still no server.

## Quickstart

```bash
# Record a run (see scripts/demo.py for a scripted example)
make demo

# Check the log: OK, or the exact seq of the first broken record
python -m blackbox verify data/demo.jsonl
# OK: 9 records, hash chain intact

# Read the run back as a transcript
python -m blackbox replay data/demo.jsonl

# Compare two runs of the same task
python -m blackbox diff data/run-a.jsonl data/run-b.jsonl
```

Record from your own code:

```python
from blackbox import Recorder

with Recorder("data/run.jsonl", task="research topic", model="gpt-x") as rec:
    rec.record("model_call", {"model": "gpt-x", "prompt": "..."}, tokens=120)
    rec.record("tool_call", {"tool": "search", "args": {"query": "..."}})
    rec.record("tool_result", {"tool": "search", "result": "3 results"})
```

`close()` (or exiting the `with` block) writes a summary record with
the event count, start/end time and total tokens. If the file already
exists, recording resumes after its last record without breaking the
chain.

SQLite backend: pass a `.db` path instead of `.jsonl` and records go
to a SQLite table instead of a text log. Same API, same records, same
hash chain — `verify`, `replay` and `diff` work on either, and `diff`
can compare a `.jsonl` run against a `.db` run:

```python
with Recorder("data/run.db", task="research topic") as rec:
    rec.record("model_call", {"model": "gpt-x", "prompt": "..."}, tokens=120)
```

```bash
python -m blackbox verify data/run.db
# OK: 4 records, hash chain intact

python -m blackbox replay data/run.db

# Query runs with plain SQL
sqlite3 data/run.db "SELECT type, COUNT(*) FROM records GROUP BY type"
```

## How it works

**Hash chain.** Each record carries `prev_hash` (the previous record's
hash, 64 zeros for the first) and `hash`, a SHA-256 over the canonical
JSON of everything else in the record. Edit any record and its hash
stops matching; every record after it stops matching too. `verify`
replays the chain from scratch and reports the first broken seq, and
checks the summary's event count, so deleted records are caught as
well. A torn final line from a crashed write fails to parse and is
reported at its position.

**Redaction before write.** Secrets are replaced before a record is
hashed or written, so the log itself never contains them. A value is
redacted if its key is a sensitive word (`api_key`, `token`, `password`,
`secret`, `authorization`, `private_key`, `access_key`, `session`,
`cookie`, `bearer`, … — matched on whole words, so a metadata key like
`total_tokens` is not redacted) or if the value matches a known secret
pattern (`sk-…` keys, `Bearer` tokens, GitHub PATs, AWS access key ids,
Slack tokens).

**Durability.** Every record is flushed and fsynced as it is written
(JSONL), or committed with `synchronous=FULL` in WAL journal mode
(SQLite). A flight recorder that loses the last seconds of a crashed
run is not much use.

**SQLite backend.** A `.db` (or `.sqlite`/`.sqlite3`) path selects the
SQLite store: one `records` table with the same fields as the JSONL
record (`seq`, `ts`, `type`, `payload`, `tokens`, `prev_hash`,
`hash`), payload and tokens stored as JSON text. The hash is computed
over the same canonical JSON as the JSONL record, so the chain is
identical and a record verifies the same way in either backend.
`close()` checkpoints the WAL into the `.db` file, so the file alone
holds the whole run.

**Cost tracking.** Each `model_call` accrues its tokens against its
model, and `close()` writes `cost_usd` and a per-model breakdown into
the summary. Pass tokens as a dict to get a price — a plain int count
has no input/output split, so it is counted but not priced:

```python
rec.record("model_call", {"model": "gpt-5", "prompt": "..."},
           tokens={"input": 1200, "output": 340})
# summary: cost_usd 0.0049, cost_by_model {"gpt-5": {"input_tokens": 1200,
#   "output_tokens": 340, "tokens": 0, "cost_usd": 0.0049}}
```

Prices are the providers' published list prices per 1M tokens,
checked 2026-11-22, for gpt-5 / gpt-5-mini / gpt-5-nano, Claude
Opus 4.1 / 4.5, Sonnet 4.5, Haiku 4.5, and Gemini 2.5 Pro / Flash /
Flash-Lite. They change often, so treat them as an estimate:
`register_pricing("gpt-5", 1.25, 10.00)` overrides a rate, and an
unknown model costs `None`, never a guessed number. `replay` prints
the run's cost under the summary.

**MCP server.** `python -m blackbox mcp data/run.jsonl` starts a
minimal MCP server on stdio (newline-delimited JSON-RPC 2.0, no
dependencies), so any MCP client can record through blackbox. It
exposes three tools: `blackbox_record` (append one step — same
redaction, hash chain and fsync as the library), `blackbox_verify`
(check the chain) and `blackbox_replay` (read the transcript back).
The session is also readable as five resources:
`blackbox://session` (the transcript), `blackbox://session.json`
(the raw records, hash chain included),
`blackbox://session/aggregates` (the session totals as JSON —
step count and duration, per-type and per-tool counts, per-model
tokens and cost, error steps, the slowest step, and how many
records an earlier compaction removed or sampling dropped, the
same totals `blackbox aggregates` prints), and
`blackbox://session/trace` (the reconstructed causal trace as
JSON — step count, span-type breakdown, the slowest steps, error
steps, parent gaps, and the parent-child tree, the same trace
`blackbox trace` prints) — all read from the
same log the tools write to, so a client always sees the current
session. A fifth resource, `blackbox://session/diff`, exposes
what changed between this session and another run as JSON —
changed, added and removed steps (position by position, like
`blackbox diff`), per-model token and cost deltas, and new vs
fixed errors, the same diff the `summarize_diff` prompt
summarises; read it as
`blackbox://session/diff?other_path=<path to the other run's
log>` — a resource read takes no arguments, so the other run's
path rides in the URI as a query parameter, and reading the
resource without it is an error. The same diff is also a resource
template, `blackbox://session/diff/{other_path}`, listed by
`resources/templates/list`: a client expands the template with the
other run's log path — `blackbox://session/diff/data/run-b.jsonl`
— so the other run is named in the URI itself instead of a query
string, and the template, the `summarize_diff` prompt and the
resource all read the same diff, so they always agree.
A second resource template, `blackbox://session/trace/{seq}`,
exposes one branch of the session's causal trace: the subtree
rooted at step seq — the same node the
`blackbox://session/trace` resource carries for that step, with
its children in log order — so a client can read one branch
without the whole tree, e.g. `blackbox://session/trace/2`.
Reading it with a step number that is not an integer, or a step
the session does not have, is an error.
One prompt template, `analyze_session`, hands the client a prompt
for analysing the session, with an optional `focus` argument
(`errors`, `cost`, `tool usage`, or anything else); the prompt
includes the transcript and the chain-verification result. A second
prompt, `summarize_costs`, hands the client a prompt for the
session's cost: per-model token usage and USD from the published
list prices, with unpriced calls (no input/output split, or an
unknown model) listed as unpriced, never guessed. A third prompt,
`summarize_anomalies`, hands the client a prompt for the session's
suspicious steps: error results, repeated identical tool calls,
tool calls with no result, and steps where a secret was redacted
before writing. The findings are heuristics computed from the log,
and the prompt says to treat them as leads, not verdicts — a
repeated call may be a legitimate retry. A fourth prompt,
`summarize_sampling`, hands the client a prompt for the session's
tail-sampling decisions: kept vs sampled-out counts, which
triggers fired, parent gaps, and a per-model / per-type breakdown
of the sampled-out spans, computed with the tail-sample defaults
(keep rate 0.5, seed 0 so the decision is reproducible). The prompt
says the sampled-out spans are still in the log — the decision
describes what an export would keep, it never deletes steps. A
fifth prompt, `summarize_trace`, hands the client a prompt for
the session's reconstructed causal trace: step count, span-type
breakdown, slowest steps, error steps, parent gaps, and per-tool
/ per-model cost and token totals, with the trace tree itself
printed so the parent-child shape is visible, not just the
counts. A parent gap here is a kept span whose causal parent
would be sampled out by the tail-sample defaults, so a sampled
export would leave it dangling. A sixth prompt, `summarize_diff`,
hands the client a prompt for what changed between this session
and another run: changed, added and removed steps (position by
position, like `blackbox diff`), per-model token and cost deltas,
and new vs fixed errors, with both transcripts included so the
client can see the change in context. It takes the other run's
log path as its `other_path` argument (.jsonl or .db).
A seventh prompt, `summarize_retention`, hands the client a
prompt for the session's retention state: what earlier
compactions already removed (and what the checkpoint certifies
about them — the removed count and the head hash, never the
removed records themselves), how much history is still in the
log and how old it is, and what `blackbox compact --keep-days`
and `--keep-n` would remove next, as dry runs — asking for the
prompt removes nothing. It takes optional `keep_days` (default
30) and `keep_n` (default 1000) arguments for the projections.
An eighth prompt, `summarize_aggregates`, hands the client a
prompt for the session at a glance: the same totals `blackbox
aggregates` prints — step count and duration, per-type and
per-tool counts, per-model tokens and cost, error steps, the
slowest step, and how many records an earlier compaction
removed or sampling dropped — with the transcript included so
the client can see the totals in context. The totals count
steps only, so compacted-away records and sampled-out steps
are reported separately, and asking for the prompt removes
nothing.
Point a client at that command as a stdio server. Closing the server
(EOF / Ctrl-C) writes the summary record, like closing a Recorder.

**Sampling.** A fourth tool, `blackbox_sample`, asks the connected
client for an LLM completion (MCP `sampling/createMessage`): the
server sends the messages and gets the sampled text back, without
ever holding a model API key itself. The prompt is redacted before
it is sent, and the sampled completion is recorded in the log as a
`model_call` step, like any other model call. If the client declines
or does not support sampling, the tool reports an error instead of
failing the server. The `Sampler` class in `sampling.py` is the same
request as a clean Python API — `sample()` returns the text, model
and stop reason, `sample_or_none()` returns None on a decline
instead of raising — over any transport with a `request()` method;
`StdioTransport` is the stdio one, and `scripts/demo_sampling.py`
shows the whole flow against a fake client, no real model needed.

**Sampling.** `Recorder(path, sample_rate=0.5, sample_seed=7)` keeps
each step with probability 0.5 — probabilistic head-based sampling
for long runs where recording every step is too much. The choice is
made by a seeded RNG, so the same seed keeps the same steps. A
dropped step is never written and never enters the hash chain, so
the chain covers exactly what is in the log and `verify` still
passes; the summary records `sample_rate` and `sampled_out`, so a
sampled log is honest about what it skipped. Tokens and cost in the
summary count only kept steps. The default rate is 1.0: everything
kept, no sampling fields in the summary.

**Tail-based sampling.** `blackbox tail-sample data/run.jsonl --out
spans.json` exports the session as OTLP spans, decided after the
full trace is known: a trace with an error step, a span slower than
5 s (`--slow-ms`) or a cost over $1 (`--max-cost-usd`) keeps every
span; a boring trace keeps each span with probability `--keep-rate`
(default 0.5, `--seed` makes the choice reproducible). The decision
is recorded in the export summary: `kept_all` or `sampled`, which
triggers fired, the rate, and how many spans were kept and dropped.
Kept spans keep their original ids, so they still point at the
records they came from. With `--causal`, a sampled span pulls in
its causal ancestors from the trace reconstruction, so the sampled
trace stays connected — a tool_result keeps its tool_call, a
tool_call keeps the model_call that decided it — and the spans'
parents come from that same causal tree instead of the hash chain,
so every non-root kept span's parent is a span in the export. The
ancestors pulled in are counted separately in the decision
(`causal_ancestors_kept`), so the export stays honest about which
spans the RNG chose and which were added to keep the tree
connected. Without `--causal` a kept span's chain parent may name a
dropped record, as before.

**OpenTelemetry export.** `python -m blackbox otel data/run.jsonl`
writes the session as OTLP/JSON spans, one per step, in a single
trace — no OpenTelemetry SDK needed, the exporter only builds dicts.
The trace id comes from the first record's hash, each span id from
its own record's hash, and each span's parent from the previous
record's hash, so the span tree is the hash chain. Span attributes
carry the seq, type, hashes, tokens, model/tool names and the full
payload, and each span runs from its step's timestamp to the next
step's. The summary record is not a span. `spans_to_records` in
`otel.py` rebuilds the steps from the spans, so an exported trace
can be checked back against the log it came from. With `--causal`
(`blackbox otel --causal`, `blackbox export --causal`, also for
`--live`), each span's parent comes from the causal trace
reconstruction instead of the hash chain, so the exported trace
reconstructs the same parent-child tree as `blackbox trace` —
tool_results hang off their tool_calls, tool_calls off the
model_call that decided them — and each span carries a
`blackbox.causal_parent` attribute marking the parent as
reconstructed. The default export is unchanged. The limits are the
trace reconstruction's: the links are inferred from step order, not
recorded, so a result for a tool called twice in a row pairs with
the earlier call by FIFO order, and an orphaned result falls back
to the previous step. Note the background `LiveExporter` (spans
sent as they are recorded) still uses chain parents: a causal
parent needs the full trace, which is only known after the run.

**Live OTLP export.** `python -m blackbox export data/run.jsonl
--live` POSTs the same spans to an OTLP/HTTP collector instead of
writing a file (default `http://localhost:4318/v1/traces`, override
with `--otlp-endpoint`). Connection errors and 429 / 5xx responses
are retried with exponential backoff (`--max-retries`, default 3);
other 4xx responses are not retried, and a failed export exits 1
with the error on stderr instead of pretending it worked. For
recording-time export, pass `otlp_endpoint` to the Recorder (a URL,
or `True` for the default endpoint) and each kept step is sent by a
background thread as it is recorded — `LiveExporter` in
`live_export.py` batches up to 100 spans or 5 seconds, and
`close()` flushes the queue before returning. A live span ends 1 ms
after it starts (there is no next step yet to look ahead to), and a
failed batch is counted as dropped, never retried forever: a
collector being down must not block the agent being recorded. The
summary record is never exported.

**Query.** `python -m blackbox query data/run.jsonl --model gpt-5
--min-cost 0.01` prints only the steps matching every filter given:
`--type` (repeatable), `--model`, `--tool`, `--min-tokens`,
`--min-cost` (USD, from the published list prices — a call with a
plain int token count or an unknown model has no cost and never
matches), `--errors-only` (an error field, a status of "error", or a
tool result mentioning an error), and `--since` / `--until` ISO
timestamps. The summary record is never a match unless `--type
summary` is given. `--json` prints each match as one raw JSON record
per line, `--count` prints only the number of matches, and the
default transcript output ends with a footer: matches, token total,
priced cost total, and how many matched model calls were unpriced.
`query_records` / `query_log` in `query.py` are the same filter as a
Python API, and both work on `.jsonl` logs and `.db` stores.

**Causal trace reconstruction.** `python -m blackbox trace
data/run.jsonl` prints the session as a parent-child tree instead
of a flat list: a tool_result hangs off the tool_call it answers
(matched first-in first-out per tool, so parallel calls to the same
tool pair with their results in order), a tool_call hangs off the
model_call that decided to call it, and a model_call hangs off the
step before it — usually the tool_result it is reacting to. A rule
that finds nothing falls back to the previous step, so an orphaned
result from a resumed or truncated log still lands somewhere instead
of being dropped. `--json` prints the nested tree as JSON.
`link_parents` / `build_trace` in `trace.py` are the same
reconstruction as a Python API, and both work on `.jsonl` logs and
`.db` stores. The links are a reconstruction, not a claim about
intent: the log records no parent pointers, so a result for a tool
called twice in a row pairs with the earlier call by FIFO order.

**Retention / compaction.** `python -m blackbox compact data/run.jsonl
--keep-days 30` (or `--keep-n 1000`) drops the records older than the
cutoff and writes a checkpoint record in their place: the hash of the
last removed record, how many were removed, and the cutoff. The
checkpoint is the new genesis of the hash chain — the kept records
are renumbered from seq 1 and re-chained from it — so `verify` still
passes on a compacted log, and tampering with a kept record, or with
the checkpoint itself, is still caught at the exact seq. A trailing
summary is dropped (a fresh one is written on the next close, as on
resume) and its event/token totals are copied into the checkpoint.
Compacting twice accumulates the count, and `--dry-run` reports what
would be removed without writing. The checkpoint certifies that N
records existed up to a given head hash and were removed — it does
not let anyone re-verify the removed records themselves; they are
gone. Compaction works on `.jsonl` logs and `.db` stores.

**Session aggregates.** `python -m blackbox aggregates data/run.jsonl`
prints the session at a glance: step count, start/end and duration,
per-type and per-tool counts, per-model calls, tokens and cost (cost
`None` for a model with no known price or no input/output split,
never a guessed number), total tokens and priced cost, error steps,
the slowest step (time until the next step), and how many records an
earlier compaction removed. `--json` prints the same totals as JSON.
`session_aggregates` in `aggregates.py` is the same computation as a
pure function over a list of records, read-only — it removes or
changes nothing — so a later MCP prompt can hand the client the
session summary instead of the whole transcript. It works on
`.jsonl` logs and `.db` stores.

**Cost totals.** `python -m blackbox costs data/run.jsonl` breaks
the session's cost down per model and per tool: per model reuses the
shared `model_usage` from `replay.py`, so it agrees with the cost
summary and the diff; per tool charges a model call to the tool it
decided to invoke, using the same causal rule as the trace
reconstruction — a tool_call hangs off the most recent model_call.
A model call invoking several tools is counted in full under each
of them, so per-tool totals can exceed the session total; a tool
with no invoking model call is counted with no cost. `--json`
prints the same breakdown as JSON. `cost_totals` in `aggregates.py`
is the same computation as a pure function over a list of records,
read-only.

## What it is not

- Not a hosted platform. There is nothing to sign up for and nowhere
  your logs go unless you put them there.
- Not an eval framework. It records what happened; judging whether it
  was good is a different tool. (Diffing two runs is a start, not an
  eval.)
- Not a model. The demo is a scripted fake run with no model calls and
  no API keys.

## Measured

- 324 tests, all passing (`make test`).
- Demo run: 8 steps + summary, 9 records, 510 tokens
  (`make demo`, then `verify` and `replay`).

## Roadmap

- MCP server wrapper: stdio server with record/verify/replay tools
  done (v0.4.0); resources and prompts done (v0.5.0); second prompt
  (`summarize_costs`) done (v0.6.0); third prompt
  (`summarize_anomalies`) done (v0.7.0); sampling done (v0.8.0).
- Span/trace export to OpenTelemetry: OTLP/JSON exporter in `otel.py`
  (trace/span ids from the hash chain, span attributes for model/tool
  calls, `spans_to_records` round-trip) and an `otel` CLI command,
  done (v0.9.0).
- Span sampling: probabilistic head-based sampling in the Recorder
  (`sample_rate`, seeded RNG, dropped steps counted in the summary,
  hash chain covers only what is kept), done (v0.10.0).
- Live OTLP export: `OtlpHttpExporter` POSTs OTLP/JSON spans to a
  collector with retry/backoff, `LiveExporter` batches spans on a
  background thread, `Recorder(otlp_endpoint=...)` exports live as
  it records, and `blackbox export --live` posts a whole log,
  done (v0.11.0).
- OTLP metrics signals: per-model call counts, input/output token
  totals, latency (time until the next step) and cost in USD where
  the model has a known price, plus per-tool call counts, as OTLP
  Sum metrics (`records_to_metrics`, `export_otlp_metrics`, and a
  `blackbox metrics` CLI command), done (v0.12.0).
- Tail-based sampling: keep all error/anomalous traces, sample the
  rest, so a busy session still exports a useful subset, done
  (v0.13.0); `tail_sample_records` keeps every span of a trace with
  an error, a slow span (>5 s) or high cost (>$1) and samples the
  rest at a configurable rate with a seed, the decision (kept_all /
  sampled, triggers that fired, rate, kept/dropped counts) is
  recorded in the export summary, and `blackbox tail-sample` writes
  the sampled OTLP export.
- Query CLI: `blackbox query` filters a session's steps by type,
  model, tool, minimum tokens, minimum USD cost, errors only, and
  time range (filters combine with AND, summary excluded unless
  asked for), with `--json` and `--count` output modes and a footer
  reporting matches, token total and priced cost, done (v0.14.0).
- Causal trace reconstruction: `blackbox trace` rebuilds the
  parent-child tree of a session — tool_results hang off their
  tool_calls (FIFO per tool), tool_calls off the model_call that
  decided them, model_calls off the previous step — as an indented
  tree or nested JSON (`link_parents` / `build_trace` in `trace.py`),
  done (v0.15.0).
- Retention / compaction: `blackbox compact` drops records older
  than a cutoff (`--keep-days`) or beyond the newest N (`--keep-n`)
  and anchors them with a checkpoint record (head hash, removed
  count, cutoff) that becomes the new genesis of the hash chain, so
  `verify` still passes and tampering after compaction is still
  caught; `--dry-run` reports without writing, done (v0.16.0).
- Causal parents in the OTLP export: `blackbox otel --causal` /
  `blackbox export --causal` sets each span's parentSpanId from the
  causal trace reconstruction (`link_parents`) instead of the hash
  chain, so the exported trace is the same parent-child tree
  `blackbox trace` prints, with a `blackbox.causal_parent`
  attribute on each span marking the parent as reconstructed; the
  default export is unchanged, done (v0.17.0).
- Causal tail sampling: `blackbox tail-sample --causal` closes a
  sampled trace under its causal ancestors — a kept span pulls in
  the tool_call / model_call above it, counted separately as
  `causal_ancestors_kept` in the decision — and takes span parents
  from the causal tree of the full log, so every non-root kept
  span's parent is a span in the export; the plain (non-causal)
  export is unchanged, done (v0.18.0).
- Parent gaps in tail sampling: in a plain (non-causal) sampled
  export a kept span whose chain parent was dropped keeps its
  parentSpanId naming the real dropped parent and is marked with a
  `blackbox.parent_sampled_out` attribute, and the decision counts
  them as `parent_gaps` (0 for kept-all traces and for `--causal`
  sampling, where closure guarantees every kept span's parent is
  in the export); a full export has no gaps and is unchanged, done
  (v0.19.0).
- Fourth MCP prompt: `summarize_sampling` hands the client a prompt
  for the session's tail-sampling decisions — kept vs sampled-out
  counts, which triggers fired, parent gaps, and a per-model /
  per-type breakdown of the sampled-out spans — computed with the
  tail-sample defaults (keep rate 0.5, seed 0 so the decision is
  reproducible); the prompt says the sampled-out spans are still
  in the log, done (v0.20.0).
- Fifth MCP prompt: `summarize_trace` hands the client a prompt
  for the session's reconstructed causal trace — step count,
  span-type breakdown, slowest steps, error steps, parent gaps
  (a kept span whose causal parent would be sampled out by the
  tail-sample defaults), and per-tool / per-model cost and token
  totals, with the trace tree printed so the shape is visible —
  done (v0.21.0).
- Sixth MCP prompt: `summarize_diff` hands the client a prompt
  for what changed between this session and another run —
  changed, added and removed steps (position by position, like
  `blackbox diff`), per-model token and cost deltas, and new vs
  fixed errors — with both transcripts included; it takes the
  other run's log path as its `other_path` argument, and
  `diff_records` in `replay.py` is the same comparison as a
  pure function with no printing, done (v0.22.0).
- Seventh MCP prompt: `summarize_retention` hands the client a
  prompt for the session's retention state — what earlier
  compactions removed (certified by the checkpoint: removed
  count, head hash, cutoff, kept count, and the dropped run's
  event/token totals), how much history is still in the log and
  how old it is, and what `blackbox compact --keep-days` and
  `--keep-n` would remove next, computed as dry runs of
  `compact_log` so asking for the prompt removes nothing; it
  takes optional `keep_days` (default 30) and `keep_n`
  (default 1000) arguments, done (v0.23.0).
- Session aggregates: `session_aggregates` in `aggregates.py`
  computes per-session totals from the records — step count,
  duration, per-type and per-tool counts, per-model calls, tokens
  and cost, error steps, the slowest step, and how many records an
  earlier compaction removed — read-only, with a `blackbox
  aggregates` CLI command (report or `--json`), done (v0.24.0).
- Eighth MCP prompt: `summarize_aggregates` hands the client a
  prompt for the session at a glance — the same totals
  `session_aggregates` computes (step count and duration,
  per-type and per-tool counts, per-model calls, tokens and
  cost, error steps, the slowest step, compacted-away and
  sampled-out counts) — via a shared `format_aggregates`
  report, read-only, with the transcript included; no
  arguments, done (v0.25.0).
- Aggregates resource: `blackbox://session/aggregates` exposes
  the same totals as JSON — step count and duration, per-type
  and per-tool counts, per-model calls, tokens and cost, error
  steps, the slowest step, compacted-away and sampled-out
  counts — so any client can read the session at a glance
  without a prompt round-trip, reusing `session_aggregates`
  from `aggregates.py`, done (v0.26.0).
- Trace resource: `blackbox://session/trace` exposes the
  reconstructed causal trace as JSON — step count, span-type
  breakdown, the slowest steps, error steps, parent gaps, and
  the parent-child tree — so any client can read the trace
  shape without a prompt round-trip, reusing `build_trace`
  and `link_parents` from `trace.py` through a shared
  `session_trace` (step latencies and error steps use the
  same helpers as the `summarize_trace` prompt, so the
  resource and the prompt always agree), done (v0.27.0).
- Diff resource: `blackbox://session/diff` exposes what
  changed between this session and another run as JSON —
  changed, added and removed steps (position by position,
  like `blackbox diff`), per-model token and cost deltas, and
  new vs fixed errors — so any client can read the diff
  without a prompt round-trip, reusing `diff_records` and a
  shared `session_diff` from `replay.py` (per-model usage
  and pricing use the same rules as the `summarize_diff`
  prompt, so the resource and the prompt always agree).
  A resource read takes no arguments, so the other run's
  log path rides in the URI as a query parameter —
  `blackbox://session/diff?other_path=<path>` — and reading
  it without one, or with an unreadable log, is an error,
  done (v0.28.0).
- Cost totals: `cost_totals` in `aggregates.py` breaks the session's
  cost down per model and per tool as JSON-able data — per model
  reuses the shared `model_usage` from `replay.py` and `cost_for`,
  so it agrees with the cost summary, the diff prompt and the diff
  resource (a plain int token count is not priced, an unknown model
  costs None, never a guessed number); per tool charges a model
  call to the tool it decided to invoke, using the same causal rule
  as `link_parents` in `trace.py`, in full under each tool it
  invoked, with how many attributed calls were unpriced reported
  alongside — with a `blackbox costs` CLI command (report or
  `--json`), done (v0.29.0).
- MCP resource template for the diff: `blackbox://session/diff/
  {other_path}`, listed by `resources/templates/list`, so a
  client can read the diff against any named run by expanding
  the template instead of building a query string — the other
  run's log path is the URI path after the prefix (an absolute
  path keeps its leading slash, a path with special characters
  is percent-encoded and decoded on read). It reuses the same
  `session_diff` as the `blackbox://session/diff?other_path=...`
  resource and the `summarize_diff` prompt, so the template, the
  prompt and the resource always agree (cost None for an
  unpriced call, never a guessed number); reading the template
  without a path, or with an unreadable log, is an error, done
  (v0.30.0).
  Next: resource templates for the other session resources, so a
  client can discover the whole session surface from
  `resources/templates/list`.
- MCP resource template for a trace subtree:
  `blackbox://session/trace/{seq}`, listed by
  `resources/templates/list`, so a client can read one branch of
  the session's causal trace — the subtree rooted at step seq —
  without the whole tree. It reuses the same `trace_subtree`
  over `build_trace` as the `blackbox://session/trace`
  resource, so the template and the resource always agree; a
  step number that is not an integer, or a step the session
  does not have, is an error, done (v0.31.0).
- MCP resource templates for the session aggregates:
  `blackbox://session/aggregates/model/{model}` and
  `blackbox://session/aggregates/tool/{tool}`, listed by
  `resources/templates/list`, so a client can read one model's
  or one tool's totals — calls, input/output tokens and USD
  cost — without the whole breakdown. They reuse the same
  `cost_totals` as the `blackbox costs` report, so the
  template and the report always agree; a model or tool the
  session does not have, or an empty name, is an error, done
  (v0.32.0).
- MCP resource template for one step's raw record:
  `blackbox://session/record/{seq}`, listed by
  `resources/templates/list`, so a client can read one step's
  record — seq, timestamp, type, payload, tokens and the hash
  chain — without the whole log. It reuses the same
  `load_records` as the `blackbox://session.json` resource,
  so the template and the resource always agree; a step number
  that is not an integer, or a step the session does not have,
  is an error, done (v0.33.0).
- MCP resource template for one step's transcript line:
  `blackbox://session/transcript/{seq}`, listed by
  `resources/templates/list`, so a client can read one step's
  formatted line — the same line the `blackbox://session`
  resource carries for that step, including the prompt line of
  a model_call — without the whole transcript. It reuses the
  same `format_record` from `replay.py` as the transcript
  resource, so the template and the resource always agree; a
  step number that is not an integer, or a step the session
  does not have, is an error, done (v0.34.0).
  Next: a resource template for one step's OTLP span
  (`blackbox://session/span/{seq}`), so a client can read one
  step as a span without exporting the whole trace.

## Layout

```
blackbox/
  recorder.py     append-only hash-chained log, fsync per record,
                  head-based span sampling (sample_rate, seeded RNG)
  sqlite_store.py SQLite store: same records, same chain, WAL + synchronous=FULL
  costs.py        per-model cost tracking: list prices, cost math
  mcp_server.py   MCP server on stdio: record / verify / replay tools,
                  session resources, resource templates for the
                  diff, a trace subtree, aggregates slices, one
                  step's raw record and one step's transcript line, analyze_session,
                  summarize_costs, summarize_anomalies,
                  summarize_sampling, summarize_trace,
                  summarize_diff and summarize_retention prompts,
                  blackbox_sample tool (MCP sampling)
  sampling.py     MCP sampling client: Sampler, StdioTransport,
                  SamplingError / NotSupported / Declined
  tail_sampling.py  tail-based sampling: keep whole error/slow/
                  expensive traces, sample the rest, decision in
                  the export summary, `blackbox tail-sample` CLI;
                  --causal closes a sampled trace under its causal
                  ancestors and takes span parents from the causal
                  tree of the full log; plain sampling marks kept
                  spans whose parent was dropped with
                  blackbox.parent_sampled_out and counts them as
                  parent_gaps in the decision
  query.py        query CLI: filter a session's steps by type,
                  model, tool, min tokens, min cost, errors only,
                  time range; --json and --count output modes
  trace.py        causal trace reconstruction: parent-child tree
                  (tool_result -> tool_call, tool_call -> model_call),
                  indented tree or nested JSON; `blackbox trace` CLI;
                  `session_trace` for the blackbox://session/trace
                  MCP resource, `trace_subtree` for the
                  blackbox://session/trace/{seq} template
  compact.py      retention / compaction: drop old records, anchor
                  them with a hash-chain checkpoint record (head
                  hash, removed count, cutoff); `blackbox compact` CLI
  aggregates.py   session aggregates: per-session totals (steps,
                  duration, per-model tokens/cost, per-tool calls,
                  errors, slowest step) and cost totals (per-model
                  and per-tool cost/token breakdowns) as pure
                  functions, with `blackbox aggregates` and
                  `blackbox costs` CLI commands
  otel.py         OpenTelemetry export: records to OTLP/JSON spans,
                  trace/span ids from the hash chain, round-trip, and
                  OTLP Sum metrics (per-model calls, tokens, latency,
                  cost; per-tool calls); --causal parents from the
                  causal trace reconstruction
  live_export.py  live OTLP export: POST spans to a collector over
                  HTTP with retry/backoff, background batching
                  thread, Recorder integration, `export --live` CLI
  redact.py       redaction before write
  verify.py       replay the chain, find the first broken record
  replay.py       transcript printer + run diff (`diff_records`,
                  `session_diff` as JSON, per-model token usage)
  __main__.py     CLI: verify | replay | diff | mcp | otel | export
                  | metrics | tail-sample | query | trace | compact
                  | aggregates | costs
scripts/
  demo.py         scripted fake agent run, 8 steps, no real model calls
  demo_sampling.py  MCP sampling against a fake client, no real model
tests/            324 tests, all passing
```
