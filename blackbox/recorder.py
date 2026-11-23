"""The recorder: append-only, hash-chained, fsynced.

Each record is one JSON line (or one SQLite row):

    {"seq": 0, "ts": "...", "type": "tool_call",
     "payload": {...}, "tokens": 120,
     "prev_hash": "0000...", "hash": "ab12..."}

The hash covers everything except itself: the canonical JSON of the
record (sorted keys, compact separators) with the previous record's
hash included. The first record's prev_hash is 64 zeros. Break any
record and every hash after it stops matching, so tampering is caught
at the first changed record.

The backend is chosen by the path: a .db/.sqlite/.sqlite3 path writes
to a SQLite store, anything else to a .jsonl log. Both backends store
the same records and the same hash chain, so verify, replay and diff
work identically on either.

Payloads are redacted before they are hashed or written, so a secret
never reaches the log. Every record is flushed and fsynced (JSONL) or
committed with synchronous=FULL (SQLite): a flight recorder that
loses the last seconds of a crashed run is not much use.

Sampling: pass sample_rate < 1.0 and each step is kept with that
probability (a seeded RNG makes the choice reproducible). Dropped
steps are never written and never enter the hash chain — the chain
covers exactly what is in the log — but the summary records the rate
and how many steps were dropped, so a sampled log is honest about
what it is missing.
"""

import hashlib
import json
import os
import random
from datetime import datetime, timezone

from .costs import cost_for
from .redact import redact
from .sqlite_store import SQLiteStore, is_sqlite_path

GENESIS_PREV_HASH = "0" * 64


def _now_iso():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _hash_record(record):
    """Hash a record dict that has no 'hash' key yet."""
    canonical = json.dumps(record, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _tokens_total(tokens):
    if tokens is None:
        return 0
    if isinstance(tokens, dict):
        return sum(v for v in tokens.values() if isinstance(v, int))
    if isinstance(tokens, int):
        return tokens
    return 0


class Recorder:
    """Append agent steps to a hash-chained log.

    If the log already exists, recording resumes after its last record,
    so a crashed run can be continued without breaking the chain.

    sample_rate (0, 1]: probability of keeping each step. 1.0 keeps
    everything (the default). Below 1.0, dropped steps are counted in
    _sampled_out and reported in the summary, but never written and
    never hashed — the chain covers exactly what is in the log.
    sample_seed makes the keep/drop choice reproducible; without it
    the choice is random per Recorder.

    otlp_endpoint: post each kept step live to an OTLP/HTTP collector
    (default collector endpoint if given as True, or a URL string).
    The exporter runs on a background thread, so a slow or down
    collector never blocks recording; a failed batch is counted as
    dropped, not retried forever. The summary record is not exported.
    """

    def __init__(self, path, task=None, model=None, sample_rate=1.0,
                 sample_seed=None, otlp_endpoint=None,
                 otlp_batch_size=100, otlp_flush_interval=5.0):
        if not 0 < sample_rate <= 1.0:
            raise ValueError("sample_rate must be in (0, 1]")
        self.path = path
        self.task = task
        self.model = model
        self.sample_rate = sample_rate
        self._rng = random.Random(sample_seed)
        self._sampled_out = 0
        self._trace_id = None
        if otlp_endpoint is True:
            from .live_export import DEFAULT_OTLP_ENDPOINT
            otlp_endpoint = DEFAULT_OTLP_ENDPOINT
        self._live = None
        if otlp_endpoint:
            # Imported here, not at module level: live_export imports
            # otel, which imports replay, which imports this module.
            from .live_export import LiveExporter
            self._live = LiveExporter(
                otlp_endpoint, batch_size=otlp_batch_size,
                flush_interval=otlp_flush_interval)
            self._live.start()
        self._seq = 0
        self._prev_hash = GENESIS_PREV_HASH
        self._count = 0
        self._total_tokens = 0
        # Per-model usage: model -> {"input_tokens", "output_tokens",
        # "tokens"}. "tokens" holds unsplit int counts, which cannot be
        # priced honestly; input/output come from tokens dicts.
        self._cost = {}
        self._started = None
        self._closed = False
        self._sqlite = is_sqlite_path(path)
        parent = os.path.dirname(os.path.abspath(path))
        os.makedirs(parent, exist_ok=True)
        if self._sqlite:
            self._store = SQLiteStore(path)
            self._file = None
        else:
            self._store = None
            self._file = None
        if self._has_data():
            self._resume()
        if not self._sqlite:
            self._file = open(path, "a", encoding="utf-8")

    def _has_data(self):
        if self._sqlite:
            return self._store.has_rows()
        return os.path.exists(self.path) and os.path.getsize(self.path) > 0

    def _load_existing(self):
        if self._sqlite:
            return self._store.read_all()
        with open(self.path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    def _drop_last(self):
        if self._sqlite:
            self._store.delete_last()
            return
        with open(self.path, "r", encoding="utf-8") as f:
            lines = [line for line in f if line.strip()]
        with open(self.path, "w", encoding="utf-8") as f:
            f.writelines(lines[:-1])

    def _accumulate_cost(self, model, tokens):
        """Fold one model_call's tokens into the per-model usage."""
        if not model:
            return
        entry = self._cost.setdefault(
            model, {"input_tokens": 0, "output_tokens": 0, "tokens": 0})
        if isinstance(tokens, dict):
            if isinstance(tokens.get("input"), int):
                entry["input_tokens"] += tokens["input"]
            if isinstance(tokens.get("output"), int):
                entry["output_tokens"] += tokens["output"]
        elif isinstance(tokens, int):
            entry["tokens"] += tokens

    def _resume(self):
        records = self._load_existing()
        # A trailing summary describes the run up to the last close.
        # Drop it: the resumed run writes a fresh summary when it closes,
        # so the log keeps exactly one summary, always last.
        if records and records[-1].get("type") == "summary":
            self._drop_last()
            records = records[:-1]
        for record in records:
            self._count += 1
            self._total_tokens += _tokens_total(record.get("tokens"))
            if self._started is None:
                self._started = record.get("ts")
            if record.get("type") == "model_call":
                model = (record.get("payload") or {}).get("model") or self.model
                self._accumulate_cost(model, record.get("tokens"))
        if records:
            last = records[-1]
            self._seq = last["seq"] + 1
            self._prev_hash = last["hash"]
            self._trace_id = records[0]["hash"][:32]

    def _write_record(self, record):
        if self._sqlite:
            self._store.append(record)
            return
        line = json.dumps(record, sort_keys=True)
        self._file.write(line + "\n")
        self._file.flush()
        os.fsync(self._file.fileno())

    def record(self, event_type, payload, tokens=None):
        """Append one step. Payload is redacted before hashing/writing.

        With sampling on, a step may be dropped instead: it is counted
        in _sampled_out, never written, and record() returns None.
        A kept step returns its seq, as before.
        """
        if self._closed:
            raise RuntimeError("Recorder is closed")
        if self._rng.random() >= self.sample_rate:
            self._sampled_out += 1
            return None
        record = {
            "seq": self._seq,
            "ts": _now_iso(),
            "type": event_type,
            "payload": redact(payload),
            "tokens": tokens,
            "prev_hash": self._prev_hash,
        }
        record["hash"] = _hash_record(record)
        self._write_record(record)
        if self._started is None:
            self._started = record["ts"]
        self._prev_hash = record["hash"]
        self._seq += 1
        self._count += 1
        self._total_tokens += _tokens_total(tokens)
        if self._live is not None:
            if self._trace_id is None:
                self._trace_id = record["hash"][:32]
            self._live.enqueue_record(record, self._trace_id)
        if event_type == "model_call":
            model = record["payload"].get("model") or self.model
            self._accumulate_cost(model, tokens)
        return record["seq"]

    def close(self):
        """Write the final summary record and close the log."""
        if self._closed:
            return
        cost_by_model = {}
        total_cost = 0.0
        for model, entry in sorted(self._cost.items()):
            # No input/output split means the usage cannot be priced
            # honestly: a plain int token count has no split to apply
            # rates to. cost_for() itself returns None for unknown
            # models, so either way unpriced usage stays None, never 0.
            cost = None
            if entry["input_tokens"] or entry["output_tokens"]:
                cost = cost_for(model, entry["input_tokens"],
                                entry["output_tokens"])
            cost_by_model[model] = dict(entry, cost_usd=cost)
            if cost is not None:
                total_cost += cost
        summary = {
            "events": self._count,
            "started": self._started,
            "ended": _now_iso(),
            "total_tokens": self._total_tokens,
            "cost_usd": round(total_cost, 6),
            "cost_by_model": cost_by_model,
            "task": self.task,
            "model": self.model,
        }
        if self.sample_rate < 1.0:
            # A sampled log says so, and how much it skipped: the
            # summary describes the log as it is, not the run as it was.
            summary["sample_rate"] = self.sample_rate
            summary["sampled_out"] = self._sampled_out
        record = {
            "seq": self._seq,
            "ts": summary["ended"],
            "type": "summary",
            "payload": redact(summary),
            "tokens": None,
            "prev_hash": self._prev_hash,
        }
        record["hash"] = _hash_record(record)
        self._write_record(record)
        if self._sqlite:
            self._store.close()
        else:
            self._file.close()
        if self._live is not None:
            self._live.stop()
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False
