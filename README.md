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
no database, no dependencies — pure Python 3 standard library.

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

**Durability.** Every record is flushed and fsynced as it is written.
A flight recorder that loses the last seconds of a crashed run is not
much use.

## What it is not

- Not a hosted platform. There is nothing to sign up for and nowhere
  your logs go unless you put them there.
- Not an eval framework. It records what happened; judging whether it
  was good is a different tool. (Diffing two runs is a start, not an
  eval.)
- Not a model. The demo is a scripted fake run with no model calls and
  no API keys.

## Measured

- 23 tests, all passing (`make test`).
- Demo run: 8 steps + summary, 9 records, 510 tokens
  (`make demo`, then `verify` and `replay`).

## Roadmap

- SQLite backend for querying runs instead of grepping JSONL.
- MCP server wrapper, so any MCP client can record through blackbox.
- Span/trace export to OpenTelemetry.
- Cost tracking per model.

## Layout

```
blackbox/
  recorder.py   append-only hash-chained log, fsync per record
  redact.py     redaction before write
  verify.py     replay the chain, find the first broken record
  replay.py     transcript printer + run diff
  __main__.py   CLI: verify | replay | diff
scripts/
  demo.py       scripted fake agent run, 8 steps, no real model calls
tests/          23 tests, all passing
```
