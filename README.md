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
includes the transcript and the chain-verification result.
Point a client at that command as a stdio server. Closing the server
(EOF / Ctrl-C) writes the summary record, like closing a Recorder.

## What it is not

- Not a hosted platform. There is nothing to sign up for and nowhere
  your logs go unless you put them there.
- Not an eval framework. It records what happened; judging whether it
  was good is a different tool. (Diffing two runs is a start, not an
  eval.)
- Not a model. The demo is a scripted fake run with no model calls and
  no API keys.

## Measured

- 71 tests, all passing (`make test`).
- Demo run: 8 steps + summary, 9 records, 510 tokens
  (`make demo`, then `verify` and `replay`).

## Roadmap

- MCP server wrapper: stdio server with record/verify/replay tools
  done (v0.4.0); resources and prompts done (v0.5.0).
  Next: sampling, or a second prompt.
- Span/trace export to OpenTelemetry.

## Layout

```
blackbox/
  recorder.py     append-only hash-chained log, fsync per record
  sqlite_store.py SQLite store: same records, same chain, WAL + synchronous=FULL
  costs.py        per-model cost tracking: list prices, cost math
  mcp_server.py   MCP server on stdio: record / verify / replay tools,
                  session resources, analyze_session prompt
  redact.py       redaction before write
  verify.py       replay the chain, find the first broken record
  replay.py       transcript printer + run diff
  __main__.py     CLI: verify | replay | diff | mcp
scripts/
  demo.py         scripted fake agent run, 8 steps, no real model calls
tests/            71 tests, all passing
```
