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

# Record one step from the shell (no Python script, no MCP server)
python -m blackbox record data/run.jsonl --type tool_call \
    --payload '{"tool": "search", "args": {"query": "mcp"}}'

# Record a whole run from a pipeline, one JSON step per line
printf '%s\n' \
    '{"type": "model_call", "payload": {"model": "gpt-5"}, "tokens": {"input": 100, "output": 20}}' \
    '{"type": "tool_call", "payload": {"tool": "search"}}' \
    | python -m blackbox record data/run.jsonl --stdin

# Or record a whole run from a file, same step JSON
python -m blackbox record data/run.jsonl --file steps.jsonl

# Check the log: OK, or the exact seq of the first broken record
python -m blackbox verify data/demo.jsonl
# OK: 9 records, hash chain intact

# Read the run back as a transcript
python -m blackbox replay data/demo.jsonl

# Read one step's raw record, hash chain included
python -m blackbox show data/demo.jsonl 1

# ... or a window of records around it
python -m blackbox show data/demo.jsonl 1 --before 1 --after 1

# Follow a live run as it happens
python -m blackbox watch data/run.jsonl

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

Record one step from the shell:

```bash
python -m blackbox record data/run.jsonl --type model_call \
    --payload '{"model": "gpt-5", "prompt": "..."}' \
    --tokens '{"input": 1200, "output": 340}' \
    --ts 2026-11-24T01:00:00Z
```

It appends exactly the step the Recorder writes — same redaction,
hash chain and fsync, so the step agrees with `blackbox span`, the
MCP templates and the full export — and then closes the log with a
fresh summary, so the log is always left valid. `--ts` is optional:
without it the step is stamped with the current time, and a
timestamp that does not parse, or carries no timezone, is an error
instead of a guessed time. `--tokens` takes a plain int or an
input/output split as JSON; `--task` / `--model` name the session,
defaulting to the log's existing values. Bad input exits non-zero
and leaves the log untouched.

Record steps in bulk from a pipeline:

```bash
printf '%s\n' \
    '{"type": "model_call", "payload": {"model": "gpt-5"}, "tokens": {"input": 100, "output": 20}}' \
    '{"type": "tool_call", "payload": {"tool": "search"}}' \
    | python -m blackbox record data/run.jsonl --stdin
# recorded 2 steps (seqs 0-1)
```

`--stdin` reads one JSON object per line (blank lines are skipped),
or a single JSON array of step objects — each step carries its own
`type`, and optionally `payload` (a JSON object, default `{}`),
`tokens` (a plain int or an input/output split) and `ts` (ISO 8601
with a timezone, default now). Every step goes through the same
Recorder path as `blackbox record` — redacted, hash-chained and
fsynced — and the log is closed once with a fresh summary covering
all of them, so a shell pipeline can log a whole run without one
process per step. All steps are validated before the log is opened,
so a bad line, a reserved type (`summary` / `checkpoint`), bad
tokens or a bad timestamp exits non-zero and leaves the log
untouched; `--stdin` cannot be combined with `--type`, `--payload`,
`--tokens` or `--ts`. It works on `.jsonl` logs and `.db` stores,
and `--task` / `--model` still name the session.

`--file steps.jsonl` reads the same step JSON as `--stdin` from a
file instead of a pipe — one JSON object per line (blank lines
skipped) or a single JSON array — for runs too big to pipe. The
same all-or-nothing validation applies: a bad line, a reserved
type, bad tokens or a bad timestamp exits non-zero and leaves the
log untouched, and `--file` cannot be combined with `--stdin` or
the single-step flags. A missing or unreadable file is an error,
and `--task` / `--model` still name the session.

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
exposes eleven tools: `blackbox_record` (append one step — same
redaction, hash chain and fsync as the library), `blackbox_verify`
(check the chain), `blackbox_replay` (read the transcript back),
`blackbox_query` (filter the session's steps — by type, model,
tool, minimum tokens, minimum USD cost, errors only, and time range
— the same filter as `blackbox query`, returning the matching raw
records as JSON with the match count, token total and priced cost
total; a bad bound, a reversed window, `--last` combined with
`--since` / `--until`, or a wrongly typed argument is an
INVALID_PARAMS error, the tool-call analogue of the CLI's exit
code 2), `blackbox_cost` (cost totals per model and per tool —
the same breakdown `blackbox costs` prints, through the shared
`cost_totals` from `aggregates.py`, so the tool and the command
always agree; with no arguments the whole breakdown, with
`model` or `tool` just that model's or tool's slice, the same
entry the aggregates/model and aggregates/tool templates serve;
a model or tool the session does not have, an empty name, or a
wrongly typed argument is an INVALID_PARAMS error, and an
unpriced call costs `None`, never a guessed number), and
`blackbox_aggregates` (session totals — step count and duration,
per-type and per-tool counts, per-model tokens and cost, error
steps, the slowest step, and how many records an earlier
compaction removed or sampling dropped, the same totals
`blackbox aggregates` prints, through the shared
`session_aggregates` over `_window_records` from
`aggregates.py`, so the tool, the command and the
`blackbox://session/aggregates` resource always agree; with no
arguments the whole session, with `since` / `until` / `last`
only the steps in that window, the same window as
`blackbox aggregates`; a bad bound, a reversed window, `--last`
combined with `--since` / `--until`, or a wrongly typed
argument is an INVALID_PARAMS error, the tool-call analogue of
the CLI's exit code 2), and `blackbox_trace` (one step's full
record as JSON — the record for step `seq`, the same record the
`blackbox://session/record/{seq}` template serves, so the tool
and the template always agree — or with `before` / `after` a
bounded window of consecutive steps around it, in log order,
clamped at the session's ends and at most 100 steps; with
`since` / `until` / `last` only the steps in that window, the
same window as `blackbox trace` and the trace resource, so a
step the window dropped is not in the session; a missing `seq`,
a `seq` that is not an integer, a step the session does not
have — including the summary record, which is bookkeeping, not
a step — a negative `before` / `after`, a window over 100
steps, a bad bound, a reversed window, `--last` combined with
`--since` / `--until`, or a wrongly typed argument is an
INVALID_PARAMS error, the tool-call analogue of the CLI's exit
code 2; read-only, it removes or changes nothing), and
`blackbox_diff` (what changed between this session and another
run — changed, added and removed steps position by position,
per-model token and cost deltas, and new vs fixed errors, the
same diff `blackbox diff` prints and the
`blackbox://session/diff` resource serves, through the shared
`session_diff` from `replay.py`, so the tool, the command, the
resource and the template always agree; with `since` / `until` /
`last` only the steps in that window are compared, the same
window as `blackbox diff`, applied to both runs, so two windows
of one session can be compared by naming the same log as the
other run; a missing or empty `otherPath`, a run that cannot be
read, a bad bound, a reversed window, `--last` combined with
`--since` / `--until`, or a wrongly typed argument is an
INVALID_PARAMS error, the tool-call analogue of the CLI's exit
code 2; read-only, it removes or changes nothing), and
`blackbox_span` (one step as a single OTLP span — the same span
the `blackbox://session/span/{seq}` template serves, through the
shared `session_span` from `otel.py`, so the tool and the
template always agree — or with `before` / `after` a bounded
window of consecutive spans around it, in log order, clamped at
the session's ends and at most 100 spans; with `since` / `until` /
`last` only the steps in that window, the same window as
`blackbox trace` and the trace resource, so a step the window
dropped is not in the session; a missing `seq`, a `seq` that is
not an integer, a step the session does not have — including the
summary record, which is bookkeeping, not a step — a negative
`before` / `after`, a window over 100 spans, a bad bound, a
reversed window, `--last` combined with `--since` / `--until`, a
wrongly typed argument, or a step with no usable timestamp is an
INVALID_PARAMS error, the tool-call analogue of the CLI's exit
code 2; read-only, it removes or changes nothing), and
`blackbox_transcript` (one step's formatted transcript line —
the same line the `blackbox://session/transcript/{seq}` template
serves, through the shared `session_transcript_step` from
`replay.py`, so the tool and the template always agree — or with
`before` / `after` a bounded window of consecutive lines around
it, in log order, clamped at the session's ends and at most 100
lines; with `since` / `until` / `last` only the steps in that
window, the same window as `blackbox replay` and the transcript
resource, so a step the window dropped is not in the session; a
missing `seq`, a `seq` that is not an integer, a step the session
does not have — including the summary record, which is
bookkeeping, not a step — a negative `before` / `after`, a window
over 100 steps, a bad bound, a reversed window, `--last` combined
with `--since` / `--until`, or a wrongly typed argument is an
INVALID_PARAMS error, the tool-call analogue of the CLI's exit
code 2; read-only, it removes or changes nothing).
The session is also readable as five resources:
`blackbox://session` (the transcript — the same `--since` /
`--until` / `--last` window as `blackbox replay` rides in the
query string too, e.g.
`blackbox://session?since=2026-11-24T01:00:00Z
&until=2026-11-24T02:00:00Z`, and narrows the session before
the transcript is built; a bad bound, a reversed window, or
`--last` combined with `--since` / `--until` is an error),
`blackbox://session.json`
(the raw records, hash chain included), and the same `--since` /
`--until` / `--last` window as `blackbox replay` rides in the
query string too, e.g.
`blackbox://session.json?since=2026-11-24T01:00:00Z
&until=2026-11-24T02:00:00Z`, and narrows the records before
they are returned; a bad bound, a reversed window, or `--last`
combined with `--since` / `--until` is an error — the summary
and checkpoint records are kept untouched, so the bookkeeping
still describes the whole log, not just the window,
`blackbox://session/aggregates` (the session totals as JSON —
step count and duration, per-type and per-tool counts, per-model
tokens and cost, error steps, the slowest step, and how many
records an earlier compaction removed or sampling dropped, the
same totals `blackbox aggregates` prints — the same `--since` /
`--until` / `--last` window as `blackbox aggregates` rides in
the query string too, e.g.
`blackbox://session/aggregates?since=2026-11-24T01:00:00Z
&until=2026-11-24T02:00:00Z`, and narrows the session before
the totals are computed; a bad bound, a reversed window, or
`--last` combined with `--since` / `--until` is an error), and
`blackbox://session/trace` (the reconstructed causal trace as
JSON — step count, span-type breakdown, the slowest steps, error
steps, parent gaps, and the parent-child tree, the same trace
`blackbox trace` prints) — the same `--since` / `--until` /
`--last` window as `blackbox trace` rides in the query string
too, e.g. `blackbox://session/trace?since=2026-11-24T01:00:00Z
&until=2026-11-24T02:00:00Z`, and narrows the session before
the trace is built; a bad bound, a reversed window, or `--last`
combined with `--since` / `--until` is an error — all read from
the same log the tools write to, so a client always sees the
current session. A fifth resource, `blackbox://session/diff`, exposes
what changed between this session and another run as JSON —
changed, added and removed steps (position by position, like
`blackbox diff`), per-model token and cost deltas, and new vs
fixed errors, the same diff the `summarize_diff` prompt
summarises; read it as
`blackbox://session/diff?other_path=<path to the other run's
log>` — a resource read takes no arguments, so the other run's
path rides in the URI as a query parameter, and reading the
resource without it is an error. The same `--since` / `--until`
/ `--last` window as `blackbox diff` rides in the query string
too — `?other_path=<path>&since=2026-11-24T01:00:00Z&until=
2026-11-24T02:00:00Z` — and narrows both runs before they are
compared; a bad bound, a reversed window, or `--last` combined
with `--since` / `--until` is an error. The same diff is also a resource
template, `blackbox://session/diff/{other_path}`, listed by
`resources/templates/list`: a client expands the template with the
other run's log path — `blackbox://session/diff/data/run-b.jsonl`
— so the other run is named in the URI itself instead of a query
string, and the window rides in the query string after the path
(`.../run-b.jsonl?since=...`); the template, the `summarize_diff` prompt and the
resource all read the same diff, so they always agree.
A second resource template, `blackbox://session/trace/{seq}`,
exposes one branch of the session's causal trace: the subtree
rooted at step seq — the same node the
`blackbox://session/trace` resource carries for that step, with
its children in log order — so a client can read one branch
without the whole tree, e.g. `blackbox://session/trace/2`.
The same `--since` / `--until` / `--last` window as
`blackbox trace` rides in the query string after the step number
too, e.g. `blackbox://session/trace/2?since=2026-11-24T01:00:00Z`,
and narrows the session before the subtree is built; a bad
bound, a reversed window, or `--last` combined with `--since` /
`--until` is an error.
Reading it with a step number that is not an integer, or a step
the session does not have, is an error. Five more resource
templates name one slice each:
`blackbox://session/aggregates/model/{model}` and
`blackbox://session/aggregates/tool/{tool}` (one model's or one
tool's totals), `blackbox://session/record/{seq}` (one step's raw
record), `blackbox://session/transcript/{seq}` (one step's
formatted line), and `blackbox://session/span/{seq}` (one step as
a single OTLP span — the step kind as the span name, the step
number as the span id, start/end times from the record's
timestamps, and a USD cost attribute on a priced model_call,
never a guessed number). Each reuses the same computation as the
resource or report it mirrors, so the template and the original
always agree; a step number that is not an integer, or a step the
session does not have, is an error.
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
nothing. A ninth prompt, `summarize_errors`, hands the client
a prompt for the session's recent error steps: the error
steps from `blackbox query --errors-only`, grouped by type,
tool and model with counts, newest first, bounded to a
`limit` argument (default 10 groups), with the transcript
included so the client can see the failures in context. The
prompt says the groups are leads, not verdicts — a repeated
failure may be a legitimate retry — and that a bounded summary
only covers the newest groups, so it never claims a session
was clean when older failures were left out.
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
dropped record, as before. `--since` / `--until` (and `--last`)
narrow the sampled export to a window, the same window as
`blackbox otel --since` / `--until`, applied before the tail
decision, so the triggers are judged on the windowed steps only —
an error outside the window does not keep the trace whole.

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
With `--since` / `--until` (or `--last 10m`) the export narrows to
a window — the same closed window as `blackbox query --since` /
`--until`, so a bad bound, a reversed window, or `--last` combined
with `--since` / `--until` is an error, exit code 2 — and the last
kept step runs to the summary timestamp, as in the unwindowed
export. The same flags work on `blackbox export`, file export and
`--live`.

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
tool result mentioning an error), `--since` / `--until` ISO
timestamps, and `--last 10m` (a relative window — a duration like
30s, 2h, 1d or 1h30m — keeping only steps from the last that long,
as a lower bound of now minus the duration, the same window as
`blackbox watch --last`, so a finished session can be narrowed to
the recent past the same way a live tail can; --last cannot be
combined with --since or --until, because it sets the lower bound
itself, and a value that is not a duration is an error, exit code
2). The summary record is never a match unless `--type
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

**Live tail.** `python -m blackbox watch data/run.jsonl` follows a
live log and prints each new step as it lands, in the same format as
the `blackbox replay` transcript, so a running agent's flight
recorder can be followed in real time. By default watching starts at
the end of the log and prints only new steps; `--from-start`
replays the existing steps first, then follows. It works on `.jsonl`
logs and `.db` stores, waits for a log that does not exist yet
instead of erroring, and stops cleanly on Ctrl-C (exit 0). A log
that shrinks while it is watched — a Recorder resuming drops the
trailing summary, a rotation replaces the file — is re-read from the
start, and steps already printed are skipped by record hash (a
resumed step reuses the dropped summary's seq, so the seq alone
cannot tell a new step from an old one), so nothing is printed twice
and nothing new is missed. A malformed appended line is reported on
stderr and skipped, never a crash — and a line that fails to parse
is treated as a possible desync first (a resume that rewrites and
re-appends between two polls leaves the file larger than the old
offset, so the byte offset goes stale and the next read starts
mid-line), so the log is re-read and the missed steps still land.
`watch_log` in `watch.py` is the same tail as a Python API, with
`stop_after` / `max_records` test hooks for stopping without a
signal. `--type` (repeatable) keeps only steps of that type and
`--errors-only` only steps that look like errors — the same error
rule as `blackbox query`, so the two commands always agree — and
`--model` keeps only model_call steps for one model and `--tool`
keeps only tool_call / tool_result steps for one tool, both
matched case-insensitively on the payload, the same rules as
`blackbox query`, so a live tail can be narrowed to one model or
one tool. `--min-tokens` keeps only steps with at least that many
tokens and `--min-cost` only model_call steps costing at least
that many USD, by the query's cost rule — a call with a plain int
token count or an unknown model has no cost and never matches
--min-cost — so a live tail can be narrowed to heavy or expensive
steps like `blackbox query` can. `--since` / `--until` keep only
steps whose timestamp falls in the closed window [since, until],
the same window as `blackbox query`, parsed the way `blackbox
record --ts` parses a timestamp — ISO 8601 with a timezone,
normalised to UTC, and a value that does not parse or carries no
timezone is an error, exit code 2 — so a live tail can be narrowed
to a time window like `blackbox query` can. `--last 10m` keeps
only steps from the last 10 minutes (30s, 2h, 1d, 1h30m — a
duration, not a timestamp), as a lower bound of now minus that
duration, fixed when watching starts, so a live tail can be
narrowed to the recent past without computing an absolute
timestamp; a step that lands later is stamped later than the bound
and always passes, so --last only narrows the --from-start replay
of what is already in the log. --last cannot be combined with
--since or --until, because it sets the lower bound itself, and a
value that is not a duration is an error, exit code 2. The summary is kept only when
`--type summary` is given, and never under `--errors-only`,
`--model`, `--tool`, `--min-tokens`, `--min-cost`, `--since`,
`--until` or `--last`. Filters combine with AND, a filtered step is still
marked seen (so a re-read after a shrink does not re-check it),
and `max_records` counts printed steps, not skipped ones.

## What it is not

- Not a hosted platform. There is nothing to sign up for and nowhere
  your logs go unless you put them there.
- Not an eval framework. It records what happened; judging whether it
  was good is a different tool. (Diffing two runs is a start, not an
  eval.)
- Not a model. The demo is a scripted fake run with no model calls and
  no API keys.

## Measured

- 641 tests, all passing (`make test`).
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
- MCP resource template for one step's OTLP span:
  `blackbox://session/span/{seq}`, listed by
  `resources/templates/list`, so a client can read one step as
  a single OTLP span without exporting the whole trace. It
  reuses the same `record_to_span` from `otel.py` as the full
  `blackbox otel` export, so the span agrees with the export,
  the transcript and the raw record — with the step number as
  the span id and the step kind as the span name, start/end
  times from the record's timestamps (the next step's, or the
  summary's for the last step; 1 ms if there is no later
  timestamp), and a `blackbox.cost_usd` attribute on a
  model_call with an input/output split and a known price; an
  unpriced call carries none, never a guessed number, and a
  step with no usable timestamp is an error instead of a span
  with invented times. The summary record is not a step, so it
  has no span, done (v0.35.0).
- `blackbox span` CLI command: prints one step as an OTLP span —
  the same span the `blackbox://session/span/{seq}` template
  serves (`session_span` from `otel.py`, so the command, the
  template and the full `blackbox otel` export always agree) —
  as a readable report (name, span/trace/parent ids, start/end
  times, duration, one `key: value` line per attribute) or as
  JSON with `--json`, so the span is readable without an MCP
  client; an unknown step, the summary record, or a step with
  no usable timestamp is an error, done (v0.36.0).
- `blackbox record` CLI command: appends one step to a log from
  the shell — type, payload JSON and token counts as flags, an
  optional `--ts` (taken as given, normalised to UTC, or the
  current time when omitted; an unparseable or timezone-less
  timestamp is an error, never a guessed time) — reusing the
  Recorder, so the step is redacted, hash-chained and fsynced
  exactly like a step recorded from code or the MCP server, and
  agrees with `blackbox span`, the MCP templates and the full
  export; it closes the log with a fresh summary (keeping the
  log's existing task/model unless `--task` / `--model`
  override them), works on `.jsonl` logs and `.db` stores, and
  bad input exits non-zero without touching the log, done
  (v0.37.0).
  Next: record steps in bulk from stdin — pipe JSONL steps into
  `blackbox record --stdin` so a shell pipeline can log a whole
  run without one process per step.
- `blackbox record --stdin`: reads steps from stdin — one JSON
  object per line (blank lines skipped) or a single JSON array of
  step objects, each step carrying its own type, payload, tokens
  and ts — and appends them through the same Recorder path as
  `blackbox record` (redacted, hash-chained, fsynced, one fresh
  summary for the whole batch), so a shell pipeline can log a
  whole run without one process per step; every step is validated
  before the log is opened, so a bad line, a reserved type, bad
  tokens or a bad timestamp exits non-zero and leaves the log
  untouched, and it works on `.jsonl` logs and `.db` stores,
  done (v0.38.0).
- `blackbox record --file`: reads steps from a file — the same
  step JSON as `--stdin` (one JSON object per line, blank lines
  skipped, or a single JSON array of step objects, each carrying
  its own type, payload, tokens and ts) — and appends them through
  the same Recorder path (redacted, hash-chained, fsynced, one
  fresh summary for the whole batch), for runs too big to pipe;
  every step is validated before the log is opened, so a bad line,
  a reserved type, bad tokens or a bad timestamp exits non-zero
  and leaves the log untouched; `--file` cannot be combined with
  `--stdin` or the single-step flags, and a missing or unreadable
  file is an error; it works on `.jsonl` logs and `.db` stores,
  done (v0.39.0).
- `blackbox watch`: tails a live log and prints each new step as it
  lands, in the same format as the `blackbox replay` transcript —
  `--from-start` replays the existing steps first, the default
  starts at the end and prints only new steps — so a running
  agent's flight recorder can be followed in real time. It works on
  `.jsonl` logs (byte-offset tail) and `.db` stores (polling for
  rows past the last printed seq), waits for a log that does not
  exist yet, and stops cleanly on Ctrl-C (exit 0). A log that
  shrinks while watched — a Recorder resuming drops the trailing
  summary, a rotation replaces the file — is re-read from the start
  with already-printed steps skipped by record hash (a resumed step
  reuses the dropped summary's seq, so the seq alone cannot tell a
  new step from an old one); a line that fails to parse is treated
  as a possible desync first and re-read, so a rewrite between two
  polls still lands the missed steps, and a genuinely malformed
  appended line is reported on stderr and skipped, never a crash,
  done (v0.40.0).
- Watch filter flags: `blackbox watch --type tool_call`
  (repeatable) keeps only steps of that type, and `--errors-only`
  only steps that look like errors — the same error rule as
  `blackbox query`, so the two commands always agree. The summary
  record is kept only when `--type summary` is given explicitly,
  and never under `--errors-only`; filters combine with AND, a
  filtered step is still marked seen so a re-read after a shrink
  does not re-check it, and `max_records` counts printed steps,
  not skipped ones. Works on `.jsonl` logs and `.db` stores, done
  (v0.41.0).
- Watch model/tool filters: `blackbox watch --model gpt-5` keeps
  only model_call steps for that model, and `blackbox watch
  --tool search` keeps only tool_call / tool_result steps for
  that tool — both matched case-insensitively on the payload, the
  same rules as `blackbox query`, so the two commands always
  agree. The summary is kept only when `--type summary` is given,
  and never under `--model` or `--tool`; filters combine with AND,
  a filtered step is still marked seen, and `max_records` counts
  printed steps, not skipped ones. Works on `.jsonl` logs and
  `.db` stores, done (v0.42.0).
  Next: `--min-tokens` / `--min-cost` filters for watch, so a live
  tail can be narrowed to heavy or expensive steps like
  `blackbox query` can.
- Watch token/cost filters: `blackbox watch --min-tokens 1000`
  keeps only steps with at least that many tokens, and
  `blackbox watch --min-cost 0.01` only model_call steps costing
  at least that many USD — by the query's cost rule, so a call
  with a plain int token count or an unknown model has no cost and
  never matches. The summary is kept only when `--type summary` is
  given, and never under `--min-tokens` or `--min-cost`, the same
  rule as `blackbox query`; filters combine with AND, a filtered
  step is still marked seen, and `max_records` counts printed
  steps, not skipped ones. Works on `.jsonl` logs and `.db` stores,
  done (v0.43.0).
  Next: `--since` / `--until` time-range filters for watch, so a
  live tail can be narrowed to a time window like `blackbox query`
  can.
- Watch time-range filters: `blackbox watch --since
  2026-11-24T01:00:00Z --until 2026-11-24T02:00:00Z` keeps only
  steps whose timestamp falls in the closed window [since, until]
  — the same window as `blackbox query`, so the two commands always
  agree — parsed the way `blackbox record --ts` parses a timestamp:
  ISO 8601 with a timezone, normalised to UTC, and a value that
  does not parse or carries no timezone is an error, exit code 2,
  as is --since after --until. Bounds compare as datetimes, so a
  step stamped with fractional seconds compares correctly against
  a whole-second bound. The summary is kept only when `--type
  summary` is given, and never under `--since` or `--until`, the
  same rule as `blackbox query`; filters combine with AND, a
  filtered step is still marked seen, and `max_records` counts
  printed steps, not skipped ones. Works on `.jsonl` logs and
  `.db` stores, done (v0.44.0).
- Watch relative window: `blackbox watch --last 10m` keeps only
  steps from the last 10 minutes — a duration (30s, 2h, 1d,
  1h30m), not a timestamp — as a lower bound of now minus that
  duration, fixed when watching starts, so a live tail can be
  narrowed to the recent past without computing an absolute
  timestamp; a step that lands later is stamped later than the
  bound and always passes, so --last only narrows the --from-start
  replay of what is already in the log. --last cannot be combined
  with --since or --until, because it sets the lower bound itself,
  and a value that is not a duration is an error, exit code 2. The
  summary is kept only when `--type summary` is given, and never
  under --last, the same rule as `blackbox query`; filters combine
  with AND, a filtered step is still marked seen, and
  `max_records` counts printed steps, not skipped ones. Works on
  `.jsonl` logs and `.db` stores, done (v0.45.0).
- Query relative window: `blackbox query --last 10m` keeps only
  steps from the last 10 minutes — a duration (30s, 2h, 1d,
  1h30m), not a timestamp — as a lower bound of now minus that
  duration, so a finished session can be narrowed to the recent
  past the same way a live tail can. --last cannot be combined
  with --since or --until, because it sets the lower bound itself,
  and a value that is not a duration is an error, exit code 2. The
  lower bound compares as datetimes, so a record stamped with
  fractional seconds compares correctly against a whole-second
  bound; a record whose timestamp does not parse, or carries no
  timezone, never matches --last. The summary is kept only when
  `--type summary` is given, the same rule as the other query
  filters; filters combine with AND, and `query_records` /
  `query_log` in `query.py` take the same filter as a Python API
  (a duration string or a timedelta, with `now` fixable for
  tests). `_parse_duration` / `_parse_record_ts` moved from
  `watch.py` into `query.py` — the filter helpers live with the
  filter — and `watch.py` imports them, so `blackbox watch
  --last` is unchanged. Works on `.jsonl` logs and `.db` stores,
  done (v0.46.0).
  Next: a `--last` relative window for `blackbox aggregates` /
  `blackbox costs`, so a finished session's totals can be narrowed
  to the recent past the same way its steps can.
- Aggregates/costs relative window: `blackbox aggregates --last
  10m` and `blackbox costs --last 10m` narrow a finished session's
  totals to the last 10 minutes — a duration (30s, 2h, 1d, 1h30m),
  not a timestamp — as a lower bound of now minus that duration,
  the same relative window as `blackbox query --last`, so a
  session's totals can be narrowed to the recent past the same way
  its steps can. The window reuses the shared duration parser and
  filter in `query.py`, so a value that is not a duration is an
  error, exit code 2, and a step whose timestamp does not parse, or
  carries no timezone, never matches. The summary and checkpoint
  records are kept untouched, so `sampled_out` and
  `compacted_count` still describe the whole log, not just the
  window. `aggregates_log` / `cost_totals_log` in `aggregates.py`
  take the same filter as a Python API (a duration string or a
  timedelta, with `now` fixable for tests). Works on `.jsonl` logs
  and `.db` stores, done (v0.47.0).
- Aggregates/costs time-range filters: `blackbox aggregates
  --since 2026-11-22T10:01:00Z --until 2026-11-22T10:04:30Z` and
  the same flags on `blackbox costs` narrow a finished session's
  totals to an absolute window — ISO 8601 timestamps with a
  timezone, the same closed window as `blackbox query --since` /
  `--until`, compared as datetimes so a record stamped with
  fractional seconds compares correctly against a whole-second
  bound. The window reuses the shared bound parser (`_parse_bound`
  moved from `watch.py` into `query.py`, where the other filter
  helpers live) and `_parse_record_ts`, so a value that does not
  parse or carries no timezone is an error, exit code 2, --since
  after --until is an error, exit code 2, --last cannot be combined
  with --since or --until, exit code 2, and a step whose timestamp
  does not parse, or carries no timezone, never matches. The
  summary and checkpoint records are kept untouched, so
  `sampled_out` and `compacted_count` still describe the whole log,
  not just the window. `aggregates_log` / `cost_totals_log` in
  `aggregates.py` take the same filters as a Python API. Works on
  `.jsonl` logs and `.db` stores, done (v0.48.0).
- Metrics time-range filters: `blackbox metrics --since
  2026-11-22T10:01:00Z --until 2026-11-22T10:04:30Z` (and
  `--last 10m`) narrow the OTLP metrics export to an absolute or
  relative window — the same closed window as `blackbox query
  --since` / `--until`, through the shared `_window_records` from
  `aggregates.py`, so a value that does not parse or carries no
  timezone is an error, exit code 2, --since after --until is an
  error, exit code 2, --last cannot be combined with --since or
  --until, exit code 2, and a step whose timestamp does not parse,
  or carries no timezone, never matches. The summary and
  checkpoint records are kept untouched, and latency is measured
  between kept steps, with the last kept step running to the
  summary timestamp, as in the unwindowed export.
  `export_otlp_metrics` / `export_otlp_metrics_file` in `otel.py`
  take the same filters as a Python API (a duration string or a
  timedelta for --last, with `now` fixable for tests). Works on
  `.jsonl` logs and `.db` stores, done (v0.49.0). Along the way,
  `records_to_metrics` stopped treating a checkpoint as a step —
  on a compacted log its 09:00 timestamp used to end the last
  model call's latency with a negative gap — and a step whose
  timestamp does not parse, or carries no timezone, now
  contributes no latency instead of crashing the export.
- OTel span time-range filters: `blackbox otel --since
  2026-11-22T10:01:00Z --until 2026-11-22T10:04:30Z` (and
  `--last 10m`, and the same flags on `blackbox export`, file
  export and `--live`) narrow the OTLP span export to an absolute
  or relative window — the same closed window as `blackbox query
  --since` / `--until`, through the shared `_window_records` from
  `aggregates.py`, so a value that does not parse or carries no
  timezone is an error, exit code 2, --since after --until is an
  error, exit code 2, --last cannot be combined with --since or
  --until, exit code 2, and a step whose timestamp does not parse,
  or carries no timezone, never matches. The summary and
  checkpoint records are kept untouched, and the last kept step
  runs to the summary timestamp, as in the unwindowed export.
  `export_otlp` / `export_otlp_file` in `otel.py` and
  `export_log` in `live_export.py` take the same filters as a
  Python API (a duration string or a timedelta for --last, with
  `now` fixable for tests). Works on `.jsonl` logs and `.db`
  stores, done (v0.50.0). Along the way, `records_to_spans`
  stopped treating a checkpoint as a step — on a compacted log
  its 09:00 timestamp used to end the last step's span with a
  negative gap — takes the last step's end time from the last
  summary record instead of assuming it is the last record, and
  a step whose timestamp does not parse, or carries no timezone,
  gets no span and contributes no latency instead of crashing
  the export.
  Next: `--since` / `--until` time-range filters for `blackbox
  tail-sample`, so a sampled span export can be narrowed to a
  window the same way the full export can.
- Tail-sample time-range filters: `blackbox tail-sample --since
  2026-11-22T10:01:00Z --until 2026-11-22T10:01:03Z` (and
  `--last 10m`) narrow the tail-sampled OTLP export to an absolute
  or relative window — the same closed window as `blackbox query
  --since` / `--until`, through the shared `_window_records` from
  `aggregates.py`, so a value that does not parse or carries no
  timezone is an error, exit code 2, --since after --until is an
  error, exit code 2, --last cannot be combined with --since or
  --until, exit code 2, and a step whose timestamp does not parse,
  or carries no timezone, never matches. The window applies
  before the tail decision, so the error / slow-span / high-cost
  triggers are judged on the windowed steps only — an error
  outside the window does not keep the trace whole — and the
  last kept step runs to the summary timestamp, as in the
  unwindowed export. `export_tail_sampled_otlp` /
  `export_tail_sampled_otlp_file` in `tail_sampling.py` take the
  same filters as a Python API (a duration string or a timedelta
  for --last, with `now` fixable for tests). Works on `.jsonl`
  logs and `.db` stores, done (v0.51.0). Along the way,
  `tail_sample_records` stopped treating a checkpoint as a step
  — it was counted in `total_spans` and its 09:00 timestamp could
  end a step's span with a negative gap — takes the last step's
  end time from the last summary record instead of assuming it
  is the last record, and a step whose timestamp does not parse,
  or carries no timezone, contributes no duration instead of
  crashing the export.
  Next: `--since` / `--until` time-range filters for `blackbox
  trace`, so a causal trace can be narrowed to a window the same
  way the exports can.
- Trace time-range filters: `blackbox trace --since
  2026-11-22T10:01:00Z --until 2026-11-22T10:01:03Z` (and
  `--last 10m`) narrow the causal trace to an absolute or
  relative window — the same closed window as `blackbox query
  --since` / `--until`, through the shared `_window_records` from
  `aggregates.py`, so a value that does not parse or carries no
  timezone is an error, exit code 2, --since after --until is an
  error, exit code 2, --last cannot be combined with --since or
  --until, exit code 2, and a step whose timestamp does not parse,
  or carries no timezone, never matches. The window applies
  before the links are rebuilt, so a kept step whose parent fell
  outside the window becomes a root instead of pointing at a
  step that is not in the tree. `trace_log` in `trace.py` takes
  the same filters as a Python API (a duration string or a
  timedelta for --last, with `now` fixable for tests). Works on
  `.jsonl` logs and `.db` stores, done (v0.52.0).
- Replay time-range filters: `blackbox replay --since
  2026-11-22T10:02:00Z --until 2026-11-22T10:04:00Z` (and
  `--last 10m`) narrow the transcript to an absolute or relative
  window — the same closed window as `blackbox query --since` /
  `--until`, through the shared `_window_records` from
  `aggregates.py` (deferred import in `replay.py`, which
  `aggregates.py` already imports), so a value that does not parse
  or carries no timezone is an error, exit code 2, --since after
  --until is an error, exit code 2, --last cannot be combined with
  --since or --until, exit code 2, and a step whose timestamp does
  not parse, or carries no timezone, never matches. The summary
  and checkpoint records are kept untouched, so the transcript
  still ends with the whole log's totals, not just the window's.
  `replay` in `replay.py` takes the same filters as a Python API
  (a duration string or a timedelta for --last, with `now` fixable
  for tests). Works on `.jsonl` logs and `.db` stores, done
  (v0.53.0).
- Diff time-range filters: `blackbox diff --since
  2026-11-22T10:02:00Z --until 2026-11-22T10:04:00Z` (and
  `--last 10m`) narrow the comparison to an absolute or relative
  window — the same closed window as `blackbox query --since` /
  `--until`, through the shared `_window_records` from
  `aggregates.py`, so a value that does not parse or carries no
  timezone is an error, exit code 2, --since after --until is an
  error, exit code 2, --last cannot be combined with --since or
  --until, exit code 2, and a step whose timestamp does not parse,
  or carries no timezone, never matches. The window is applied to
  both runs before they are compared position by position, so a
  step outside the window cannot make the runs look different.
  The summary and checkpoint records are kept untouched, so the
  comparison still runs over the whole log's bookkeeping, not
  just the window's. `diff` in `replay.py` takes the same filters
  as a Python API (a duration string or a timedelta for --last,
  with `now` fixable for tests). Works on `.jsonl` logs and `.db`
  stores, done (v0.54.0).
- Diff resource time-range filters: `blackbox://session/diff`
  takes the same `--since` / `--until` / `--last` window as
  `blackbox diff`, as query parameters — e.g.
  `blackbox://session/diff?other_path=<path>&since=
  2026-11-24T01:00:00Z&until=2026-11-24T02:00:00Z`, and the
  `blackbox://session/diff/{other_path}` template takes it the
  same way, after the path — through the shared `_window_records`
  from `aggregates.py`, so a value that does not parse or carries
  no timezone is an error, --since after --until is an error, and
  --last cannot be combined with --since or --until is an error
  (an INVALID_PARAMS error on a resource read, exit code 2 on the
  CLI), and a step whose timestamp does not parse, or carries no
  timezone, never matches. The window is applied to both runs before they
  are compared position by position, so a step outside the window
  cannot make the runs look different. The summary and checkpoint
  records are kept untouched, so the comparison still runs over
  the whole log's bookkeeping, not just the window's. Works on
  `.jsonl` logs and `.db` stores, done (v0.55.0).
- Trace resource time-range filters: `blackbox://session/trace`
  takes the same `--since` / `--until` / `--last` window as
  `blackbox trace`, as query parameters — e.g.
  `blackbox://session/trace?since=2026-11-24T01:00:00Z&until=
  2026-11-24T02:00:00Z` — through the shared `_window_records`
  from `aggregates.py`, so a value that does not parse or carries
  no timezone is an error, --since after --until is an error, and
  --last cannot be combined with --since or --until is an error
  (an INVALID_PARAMS error on a resource read, exit code 2 on the
  CLI), and a step whose timestamp does not parse, or carries no
  timezone, never matches. The window is applied to the session
  before the trace is built, so a step outside the window cannot
  change the tree. The summary and checkpoint records are kept
  untouched, so the trace still runs over the whole log's
  bookkeeping, not just the window's. Works on `.jsonl` logs and
  `.db` stores, done (v0.56.0).
- Trace template time-range filters:
  `blackbox://session/trace/{seq}` takes the same `--since` /
  `--until` / `--last` window as `blackbox trace`, as query
  parameters after the step number — e.g.
  `blackbox://session/trace/2?since=2026-11-24T01:00:00Z&until=
  2026-11-24T02:00:00Z` — through the shared `_window_records`
  from `aggregates.py`, so a value that does not parse or carries
  no timezone is an error, --since after --until is an error, and
  --last cannot be combined with --since or --until is an error
  (an INVALID_PARAMS error on a resource read, exit code 2 on the
  CLI), and a step whose timestamp does not parse, or carries no
  timezone, never matches. The window is applied to the session
  before the subtree is built, so a step outside the window
  cannot change the branch, and a step the window dropped is not
  in the session. The summary and checkpoint records are kept
  untouched, so the trace still runs over the whole log's
  bookkeeping, not just the window's. Works on `.jsonl` logs and
  `.db` stores, done (v0.57.0).
- Aggregates resource time-range filters:
  `blackbox://session/aggregates` takes the same `--since` /
  `--until` / `--last` window as `blackbox aggregates`, as query
  parameters — e.g.
  `blackbox://session/aggregates?since=2026-11-24T01:00:00Z&until=
  2026-11-24T02:00:00Z` — through the shared `_window_records`
  from `aggregates.py`, so a value that does not parse or carries
  no timezone is an error, --since after --until is an error, and
  --last cannot be combined with --since or --until is an error
  (an INVALID_PARAMS error on a resource read, exit code 2 on the
  CLI), and a step whose timestamp does not parse, or carries no
  timezone, never matches. The window is applied to the session
  before the totals are computed, so a step outside the window
  cannot change them. The summary and checkpoint records are kept
  untouched, so `sampled_out` and `compacted_count` still
  describe the whole log, not just the window. Works on `.jsonl`
  logs and `.db` stores, done (v0.58.0).
- Transcript resource time-range filters:
  `blackbox://session` takes the same `--since` / `--until` /
  `--last` window as `blackbox replay`, as query parameters —
  e.g. `blackbox://session?since=2026-11-24T01:00:00Z&until=
  2026-11-24T02:00:00Z` — through the shared `_window_records`
  from `aggregates.py`, so a value that does not parse or carries
  no timezone is an error, --since after --until is an error, and
  --last cannot be combined with --since or --until is an error
  (an INVALID_PARAMS error on a resource read, exit code 2 on the
  CLI), and a step whose timestamp does not parse, or carries no
  timezone, never matches. The window is applied to the session
  before the transcript is built, so a step outside the window
  cannot change it. The summary and checkpoint records are kept
  untouched, so the transcript still ends with the whole log's
  totals, not just the window's. Works on `.jsonl` logs and `.db`
  stores, done (v0.59.0).
- Raw records resource time-range filters:
  `blackbox://session.json` takes the same `--since` / `--until`
  / `--last` window as `blackbox replay`, as query parameters —
  e.g. `blackbox://session.json?since=2026-11-24T01:00:00Z&until=
  2026-11-24T02:00:00Z` — through the shared `_window_records`
  from `aggregates.py`, so a value that does not parse or carries
  no timezone is an error, --since after --until is an error, and
  --last cannot be combined with --since or --until is an error
  (an INVALID_PARAMS error on a resource read, exit code 2 on the
  CLI), and a step whose timestamp does not parse, or carries no
  timezone, never matches. The window is applied to the records
  before they are returned, so a step outside the window cannot
  change them. The summary and checkpoint records are kept
  untouched, so the bookkeeping still describes the whole log,
  not just the window. Works on `.jsonl` logs and `.db` stores,
  done (v0.60.0).
  With this, every session resource that returns a set of steps
  takes the window — `blackbox://session`, `blackbox://
  session.json`, `blackbox://session/aggregates`, `blackbox://
  session/trace`, `blackbox://session/diff` and the diff and
  trace templates. The single-slice templates (`record/{seq}`,
  `transcript/{seq}`, `span/{seq}`, `aggregates/model/{model}`,
  `aggregates/tool/{tool}`) return one step or one slice, so
  there is no window to apply there.
- MCP query tool: `blackbox_query` filters the session's steps
  from an MCP client — by type, model, tool, minimum tokens,
  minimum USD cost, errors only, and `--since` / `--until` /
  `--last` — the same filter as `blackbox query`, through the
  shared `query_records` from `query.py`, so the tool and the
  command always agree: filters combine with AND, the summary is
  excluded unless "summary" is in the type list, a model_call
  with a plain int token count or an unknown model has no cost
  and never matches `minCost`. It returns the matching raw
  records as JSON with the match count, token total, priced cost
  total and how many matched model calls were unpriced — the same
  footer the CLI prints. A bad bound, a reversed window,
  `--last` combined with `--since` / `--until`, or a wrongly
  typed argument is an INVALID_PARAMS error, the tool-call
  analogue of the CLI's exit code 2, done (v0.61.0).
- Ninth MCP prompt: `summarize_errors` hands the client a prompt
  for the session's recent error steps — the error steps from
  `blackbox query --errors-only`, through the shared
  `query_records` from `query.py`, so the prompt and the command
  always agree (an explicit error field, a status of "error", or
  a tool result mentioning an error) — grouped by type, tool and
  model with counts, groups newest first, step numbers inside a
  group newest first too, and bounded to a `limit` argument
  (default 10 groups, a positive integer, anything else an
  INVALID_PARAMS error), with the groups beyond the bound
  reported as "N more error steps in M more groups" so a bounded
  summary never claims a session was clean when older failures
  were left out; the prompt says the groups are leads, not
  verdicts, and includes the transcript so the client can see
  the failures in context, done (v0.62.0).
- MCP cost tool: `blackbox_cost` returns the session's cost
  totals as JSON — the whole per-model / per-tool breakdown
  `blackbox costs` prints, or just one model's or one tool's
  slice with a `model` / `tool` argument, the same entry the
  aggregates/model and aggregates/tool templates serve —
  through the shared `cost_totals` from `aggregates.py`, so
  the tool, the command and the templates always agree (an
  unpriced call costs None, never a guessed number); a model
  or tool the session does not have, an empty name, or a
  wrongly typed argument is an INVALID_PARAMS error, done
  (v0.63.0).
- MCP aggregates tool: `blackbox_aggregates` returns the session
  totals as JSON — the same dict `blackbox aggregates --json`
  prints, through the shared `session_aggregates` over
  `_window_records` from `aggregates.py`, so the tool, the
  command and the `blackbox://session/aggregates` resource always
  agree (step count and duration, per-type and per-tool counts,
  per-model tokens and cost, error steps, the slowest step, and
  how many records an earlier compaction removed or sampling
  dropped; an unpriced call costs None, never a guessed number);
  with no arguments the whole session, with `since` / `until` /
  `last` only the steps in that window, the same window as
  `blackbox aggregates` and the resource. A bad bound, a reversed
  window, `--last` combined with `--since` / `--until`, an empty
  bound, or a wrongly typed argument is an INVALID_PARAMS error.
  Read-only: it removes or changes nothing, done (v0.64.0).
- MCP trace tool: `blackbox_trace` returns one step's full
  record as JSON — the record for step `seq`, the same record
  the `blackbox://session/record/{seq}` template serves, read
  from the same `load_records`, so the tool and the template
  always agree — or with `before` / `after` a bounded window
  of consecutive steps around it, in log order, clamped at the
  session's ends and at most 100 steps; with `since` / `until`
  / `last` only the steps in that window, through the shared
  `_window_records` from `aggregates.py`, the same window as
  `blackbox trace` and the trace resource, so a step the window
  dropped is not in the session. A missing `seq`, a `seq` that
  is not an integer, a step the session does not have
  (including the summary record, which is bookkeeping, not a
  step), a negative `before` / `after`, a window over 100
  steps, a bad bound, a reversed window, `--last` combined
  with `--since` / `--until`, or a wrongly typed argument is
  an INVALID_PARAMS error. Read-only: it removes or changes
  nothing, done (v0.65.0).
- MCP diff tool: `blackbox_diff` returns what changed between
  this session and another run as JSON — the same dict
  `session_diff` from `replay.py` computes for `blackbox diff`,
  the `blackbox://session/diff` resource and the
  `blackbox://session/diff/{other_path}` template, through a
  shared `_diff_data` helper, so the tool, the command, the
  resource and the template always agree (changed, added and
  removed steps position by position, the token totals of both
  runs, per-model token and cost deltas, and new vs fixed
  errors; an unpriced call costs None, never a guessed number);
  with `since` / `until` / `last` only the steps in that window
  are compared, the same window as `blackbox diff`, applied to
  both runs, so two windows of one session can be compared by
  naming the same log as the other run. A missing or empty
  `otherPath`, a run that cannot be read, a bad bound, a
  reversed window, `--last` combined with `--since` / `--until`,
  or a wrongly typed argument is an INVALID_PARAMS error.
  Read-only: it removes or changes nothing, done (v0.66.0).
  Next: an MCP span tool, so a client can read one step as a
  single OTLP span as a tool call the same way the
  `blackbox://session/span/{seq}` template serves it.
- MCP span tool: `blackbox_span` returns one step as a single
  OTLP span as JSON — the same span `session_span` from `otel.py`
  builds for the `blackbox://session/span/{seq}` template, so the
  tool and the template always agree (the step kind as the span
  name, the step number as the span id, start/end times from the
  record's timestamps, and a USD cost attribute on a priced
  model_call, never a guessed number) — or with `before` /
  `after` a bounded window of consecutive spans around it, in log
  order, clamped at the session's ends and at most 100 spans;
  with `since` / `until` / `last` only the steps in that window,
  through the shared `_window_records` from `aggregates.py`, the
  same window as `blackbox trace` and the trace resource, so a
  step the window dropped is not in the session. A missing `seq`,
  a `seq` that is not an integer, a step the session does not
  have — including the summary record, which is bookkeeping, not
  a step — a negative `before` / `after`, a window over 100
  spans, a bad bound, a reversed window, `--last` combined with
  `--since` / `--until`, a wrongly typed argument, or a step with
  no usable timestamp is an INVALID_PARAMS error. Read-only: it
  removes or changes nothing, done (v0.67.0).
- MCP transcript tool: `blackbox_transcript` returns one step's
  formatted transcript line as JSON — the same line
  `session_transcript_step` from `replay.py` builds for the
  `blackbox://session/transcript/{seq}` template, so the tool
  and the template always agree (the line `blackbox replay`
  prints for that step, including the prompt line of a
  model_call) — or with `before` / `after` a bounded window of
  consecutive lines around it, in log order, clamped at the
  session's ends and at most 100 lines; with `since` / `until` /
  `last` only the steps in that window, through the shared
  `_window_records` from `aggregates.py`, the same window as
  `blackbox replay` and the transcript resource, so a step the
  window dropped is not in the session. A missing `seq`, a `seq`
  that is not an integer, a step the session does not have —
  including the summary record, which is bookkeeping, not a
  step — a negative `before` / `after`, a window over 100
  steps, a bad bound, a reversed window, `--last` combined with
  `--since` / `--until`, or a wrongly typed argument is an
  INVALID_PARAMS error. Read-only: it removes or changes
  nothing, done (v0.68.0).
- `blackbox transcript` CLI command: prints one step's
  transcript line — or with `--before` / `--after` a bounded
  window of consecutive lines around it, in log order, clamped
  at the session's ends and at most 100 lines; with `--since` /
  `--until` / `--last` only the steps in that window — as
  transcript lines, or as JSON with `--json`, reusing the shared
  `transcript_window` from `replay.py` (the same `_window_records`
  and `session_transcript_step` the `blackbox_transcript` MCP
  tool now calls too), so the command, the tool and the
  `blackbox://session/transcript/{seq}` template always agree;
  an unknown step, the summary record, a negative `--before` /
  `--after`, a window over 100 lines, a bad bound, a reversed
  window, or `--last` combined with `--since` / `--until` is an
  error, exit code 2, done (v0.69.0).
- `blackbox show` CLI command: prints one step's raw record —
  the same record the `blackbox://session/record/{seq}` template
  serves, through the shared `session_record` from `replay.py`
  (the template now calls it too), so the command and the
  template always agree, byte for byte with `--json` — as a
  readable report (the transcript line plus the hash chain) or
  as JSON with `--json`, so the record is readable without an
  MCP client; the summary record can be shown too, since it is
  a record in the log even though it is bookkeeping, not a step;
  an unknown step is an error, exit code 2. Works on `.jsonl`
  logs and `.db` stores, done (v0.70.0).
- Window flags for `blackbox show`: prints a bounded window of
  raw records around a step — `--before` / `--after` consecutive
  records around it, in log order, clamped at the session's ends
  and at most 100 records; `--since` / `--until` / `--last` narrows
  the session first, so a step the window dropped is not in the
  session — as readable reports (the transcript line plus the hash
  chain, one block per record) or as JSON with `--json` (the window
  as a dict; with no window flags, the single raw record, byte for
  byte the record the `blackbox://session/record/{seq}` template
  serves), reusing the shared `show_window` from `replay.py`
  (the same `_window_records` the `blackbox_transcript` MCP tool
  and `blackbox transcript` use), so the command and the template
  always agree; the summary record can be shown in a window too,
  since it is a record in the log even though it is bookkeeping,
  not a step; an unknown step, a negative `--before` / `--after`,
  a window over 100 records, a bad bound, a reversed window, or
  `--last` combined with `--since` / `--until` is an error, exit
  code 2, done (v0.71.0).
- Window flags for `blackbox span`: prints a bounded window of
  OTLP spans around a step — `--before` / `--after` consecutive
  spans around it, in log order, clamped at the session's ends
  and at most 100 spans; `--since` / `--until` / `--last` narrows
  the session first, so a step the window dropped is not in the
  session — as readable reports (one `format_span` report per
  span) or as JSON with `--json` (the window as a dict; with no
  window flags, the single span, byte for byte the span the
  `blackbox://session/span/{seq}` template serves), reusing the
  shared `span_window` from `otel.py` (the `blackbox_span` MCP
  tool now calls it too), so the command, the tool and the
  template always agree; the summary record has no span, so it
  cannot be spanned, even in a window; an unknown step, a
  negative `--before` / `--after`, a window over 100 spans, a
  bad bound, a reversed window, `--last` combined with `--since`
  / `--until`, or a step with no usable timestamp is an error,
  exit code 2, done (v0.72.0).
  Next: a `--causal` flag for `blackbox span`, so a window of
  spans takes its parents from the causal trace reconstruction
  instead of the hash chain, the same tree `blackbox trace`
  prints.

## Layout

```
blackbox/
  recorder.py     append-only hash-chained log, fsync per record,
                  head-based span sampling (sample_rate, seeded RNG)
  sqlite_store.py SQLite store: same records, same chain, WAL + synchronous=FULL
  costs.py        per-model cost tracking: list prices, cost math
  mcp_server.py   MCP server on stdio: record / verify / replay /
                  query / cost / aggregates / trace / diff / span /
                  transcript tools, session resources, resource
                  templates for the
                  diff, a trace subtree, aggregates slices, one
                  step's raw record, one step's transcript line and
                  one step's OTLP span, analyze_session,
                  summarize_costs, summarize_anomalies,
                  summarize_sampling, summarize_trace,
                  summarize_diff, summarize_retention,
                  summarize_aggregates and summarize_errors prompts,
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
                  parent_gaps in the decision; --since / --until /
                  --last narrow the sampled export to a window
                  before the tail decision
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
                  causal trace reconstruction; `session_span` renders
                  one step as a single span for the
                  blackbox://session/span/{seq} template and the
                  `blackbox span` CLI command (`format_span` prints
                  it as a readable report); `span_window` builds a
                  bounded window of spans around one step, shared by
                  the CLI and the blackbox_span MCP tool
  live_export.py  live OTLP export: POST spans to a collector over
                  HTTP with retry/backoff, background batching
                  thread, Recorder integration, `export --live` CLI
  redact.py       redaction before write
  verify.py       replay the chain, find the first broken record
  replay.py       transcript printer + run diff (`diff_records`,
                  `session_diff` as JSON, per-model token usage;
                  `session_transcript_step` renders one step's line
                  for the transcript template and tool;
                  `session_record` finds one step's raw record for
                  the record template and the `blackbox show`
                  command)
  __main__.py     CLI: verify | replay | diff | mcp | record | otel
                  | export | metrics | tail-sample | query | trace
                  | compact | aggregates | costs | span | watch | show
  watch.py        live tail: follow a log and print new steps as
                  they land, in the replay transcript format —
                  .jsonl by byte offset, .db by polling past the
                  last printed seq; a shrunk or replaced log is
                  re-read from the start with already-printed steps
                  skipped by record hash, a line that fails to
                  parse is treated as a possible desync and
                  re-read, and a genuinely malformed line is
                  reported and skipped; --type / --errors-only /
                  --model / --tool / --min-tokens / --min-cost
                  filters (the query's match and cost rules) narrow
                  the tail, and Ctrl-C stops cleanly
scripts/
  demo.py         scripted fake agent run, 8 steps, no real model calls
  demo_sampling.py  MCP sampling against a fake client, no real model
tests/            599 tests, all passing
```
