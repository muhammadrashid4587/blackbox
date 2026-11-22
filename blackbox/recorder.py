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
"""

import hashlib
import json
import os
from datetime import datetime, timezone

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
    """

    def __init__(self, path, task=None, model=None):
        self.path = path
        self.task = task
        self.model = model
        self._seq = 0
        self._prev_hash = GENESIS_PREV_HASH
        self._count = 0
        self._total_tokens = 0
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
        if records:
            last = records[-1]
            self._seq = last["seq"] + 1
            self._prev_hash = last["hash"]

    def _write_record(self, record):
        if self._sqlite:
            self._store.append(record)
            return
        line = json.dumps(record, sort_keys=True)
        self._file.write(line + "\n")
        self._file.flush()
        os.fsync(self._file.fileno())

    def record(self, event_type, payload, tokens=None):
        """Append one step. Payload is redacted before hashing/writing."""
        if self._closed:
            raise RuntimeError("Recorder is closed")
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
        return record["seq"]

    def close(self):
        """Write the final summary record and close the log."""
        if self._closed:
            return
        summary = {
            "events": self._count,
            "started": self._started,
            "ended": _now_iso(),
            "total_tokens": self._total_tokens,
            "task": self.task,
            "model": self.model,
        }
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
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False
