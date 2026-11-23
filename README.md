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
The session is also readable as two resources:
`blackbox://session` (the transcript) and `blackbox://session.json`
(the raw records, hash chain included) — both read from the same log
the tools write to, so a client always sees the current session.
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
repeated call may be a legitimate retry.
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
records they came from.

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
can be checked back against the log it came from.

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

## What it is not

- Not a hosted platform. There is nothing to sign up for and nowhere
  your logs go unless you put them there.
- Not an eval framework. It records what happened; judging whether it
  was good is a different tool. (Diffing two runs is a start, not an
  eval.)
- Not a model. The demo is a scripted fake run with no model calls and
  no API keys.

## Measured

- 161 tests, all passing (`make test`).
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

## Layout

```
blackbox/
  recorder.py     append-only hash-chained log, fsync per record,
                  head-based span sampling (sample_rate, seeded RNG)
  sqlite_store.py SQLite store: same records, same chain, WAL + synchronous=FULL
  costs.py        per-model cost tracking: list prices, cost math
  mcp_server.py   MCP server on stdio: record / verify / replay tools,
                  session resources, analyze_session,
                  summarize_costs and summarize_anomalies prompts,
                  blackbox_sample tool (MCP sampling)
  sampling.py     MCP sampling client: Sampler, StdioTransport,
                  SamplingError / NotSupported / Declined
  tail_sampling.py  tail-based sampling: keep whole error/slow/
                  expensive traces, sample the rest, decision in
                  the export summary, `blackbox tail-sample` CLI
  otel.py         OpenTelemetry export: records to OTLP/JSON spans,
                  trace/span ids from the hash chain, round-trip, and
                  OTLP Sum metrics (per-model calls, tokens, latency,
                  cost; per-tool calls)
  live_export.py  live OTLP export: POST spans to a collector over
                  HTTP with retry/backoff, background batching
                  thread, Recorder integration, `export --live` CLI
  redact.py       redaction before write
  verify.py       replay the chain, find the first broken record
  replay.py       transcript printer + run diff
  __main__.py     CLI: verify | replay | diff | mcp | otel | export
scripts/
  demo.py         scripted fake agent run, 8 steps, no real model calls
  demo_sampling.py  MCP sampling against a fake client, no real model
tests/            161 tests, all passing
```
