"""Retention / compaction: drop old records, anchor them with a checkpoint.

A log grows forever; compaction keeps it bounded. `compact_log` drops
the records older than a cutoff (by timestamp, `--keep-days`) or
beyond the newest N (`--keep-n`), and writes a checkpoint record in
their place:

    {"seq": 0, "ts": "<now>", "type": "checkpoint",
     "payload": {"head_hash": "<hash of the last removed record>",
                 "compacted_count": 42, "cutoff_ts": "<ISO cutoff or null>",
                 "kept_count": 3, ...},
     "prev_hash": "0000...", "hash": "<sha256 of the checkpoint>"}

The checkpoint is the new genesis of the hash chain: the kept records
are renumbered from seq 1 with their prev_hash/hash recomputed so the
chain runs checkpoint -> kept records. `verify` still passes on a
compacted log, because it replays the chain from the checkpoint
forward — and tampering with a kept record, or with the checkpoint
itself, is still caught at the exact seq, because every hash after
the change stops matching.

What the checkpoint proves, and what it does not: it certifies that N
records existed up to head_hash H and were removed at the cutoff — so
a compacted log cannot silently pretend the old records never
existed, and the count of removed records is on the record. It does
not let anyone re-verify the removed records themselves; they are
gone. Compaction is a deliberate, destructive retention decision,
not a free lunch.

Rules:
- A trailing summary is dropped, not compacted: a compacted log gets
  a fresh summary when the Recorder next closes, the same as a
  resumed log. Its event/token totals are copied into the checkpoint
  payload so the information is not silently lost.
- Compacting a log that already starts with a checkpoint replaces
  that checkpoint: the new compacted_count adds the newly removed
  records to the old count, and head_hash moves to the newest removed
  record, so the chain of custody survives repeated compactions.
- Nothing to remove means no rewrite: the log is left byte-identical.
- JSONL writes go through a temp file + os.replace, so a crash
  mid-compaction leaves the old log intact. SQLite replaces all rows
  in one transaction.
"""

import json
import os
from datetime import datetime, timedelta, timezone

from .recorder import GENESIS_PREV_HASH, _hash_record, _now_iso
from .replay import load_records
from .sqlite_store import SQLiteStore, is_sqlite_path


def _parse_ts(ts, seq):
    """Parse an ISO timestamp to an aware datetime, or raise ValueError."""
    if not isinstance(ts, str):
        raise ValueError("record seq %d has no timestamp: %r" % (seq, ts))
    try:
        parsed = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        raise ValueError("record seq %d has unparseable timestamp: %r"
                         % (seq, ts)) from None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def compact_log(path, keep_days=None, keep_n=None, dry_run=False):
    """Compact a log, keeping recent records and checkpointing the rest.

    keep_days: keep records at or after now minus this many days.
    keep_n: keep the newest this many records. At least one must be
    given; when both are given, a record must satisfy both to be kept.

    Returns a dict: {"changed", "compacted", "kept", "head_hash",
    "checkpoint_hash", "cutoff_ts", "path"}. With dry_run, nothing is
    written and checkpoint_hash is None. When nothing would be
    removed, changed is False and the log is untouched.
    """
    if keep_days is None and keep_n is None:
        raise ValueError("compact needs --keep-days or --keep-n")
    if keep_days is not None and keep_days < 0:
        raise ValueError("keep_days must be >= 0")
    if keep_n is not None and keep_n < 0:
        raise ValueError("keep_n must be >= 0")

    records = load_records(path)
    if not records:
        return {"changed": False, "compacted": 0, "kept": 0,
                "head_hash": None, "checkpoint_hash": None,
                "cutoff_ts": None, "path": path}

    # A trailing summary describes the run up to the last close; it is
    # dropped, like on resume, and its totals are carried into the
    # checkpoint instead of being silently lost.
    summary = None
    if records[-1].get("type") == "summary":
        summary = records[-1].get("payload") or {}
        records = records[:-1]

    # An existing checkpoint is superseded by the new one: its count
    # carries forward, its own record is not counted as compacted.
    old_compacted = 0
    if records and records[0].get("type") == "checkpoint":
        old_compacted = (records[0].get("payload") or {}).get(
            "compacted_count", 0) or 0
        records = records[1:]

    cutoff = None
    if keep_days is not None:
        cutoff = datetime.now(timezone.utc) - timedelta(days=keep_days)

    # A record is kept only if it is new enough (when a cutoff applies)
    # and within the newest keep_n (when a count applies).
    newest_start = len(records) - keep_n if keep_n is not None else 0
    kept = []
    kept_positions = set()
    for position, record in enumerate(records):
        if cutoff is not None and _parse_ts(record["ts"],
                                            record["seq"]) < cutoff:
            continue
        if position < newest_start:
            continue
        kept.append(record)
        kept_positions.add(position)

    removed = [r for position, r in enumerate(records)
               if position not in kept_positions]
    if not removed:
        return {"changed": False, "compacted": 0, "kept": len(kept),
                "head_hash": None, "checkpoint_hash": None,
                "cutoff_ts": cutoff.isoformat().replace("+00:00", "Z")
                if cutoff is not None else None,
                "path": path}

    head_hash = removed[-1]["hash"]
    compacted_count = old_compacted + len(removed)
    payload = {
        "head_hash": head_hash,
        "compacted_count": compacted_count,
        "cutoff_ts": cutoff.isoformat().replace("+00:00", "Z")
        if cutoff is not None else None,
        "keep_n": keep_n,
        "kept_count": len(kept),
    }
    if summary is not None:
        if summary.get("events") is not None:
            payload["original_events"] = summary["events"]
        if summary.get("total_tokens") is not None:
            payload["original_total_tokens"] = summary["total_tokens"]

    checkpoint = {
        "seq": 0,
        "ts": _now_iso(),
        "type": "checkpoint",
        "payload": payload,
        "tokens": None,
        "prev_hash": GENESIS_PREV_HASH,
    }
    checkpoint["hash"] = _hash_record(checkpoint)

    # Kept records are renumbered from seq 1 and re-chained from the
    # checkpoint: the checkpoint is the new genesis.
    new_records = [checkpoint]
    prev_hash = checkpoint["hash"]
    for position, record in enumerate(kept, start=1):
        new_record = {
            "seq": position,
            "ts": record["ts"],
            "type": record["type"],
            "payload": record["payload"],
            "tokens": record.get("tokens"),
            "prev_hash": prev_hash,
        }
        new_record["hash"] = _hash_record(new_record)
        new_records.append(new_record)
        prev_hash = new_record["hash"]

    result = {"changed": True, "compacted": len(removed),
              "kept": len(kept), "head_hash": head_hash,
              "checkpoint_hash": checkpoint["hash"],
              "cutoff_ts": payload["cutoff_ts"], "path": path}
    if dry_run:
        result["checkpoint_hash"] = None
        return result

    if is_sqlite_path(path):
        store = SQLiteStore(path)
        try:
            store.replace_all(new_records)
        finally:
            store.close()
    else:
        _write_jsonl_atomic(path, new_records)
    return result


def _write_jsonl_atomic(path, records):
    """Rewrite a .jsonl log via temp file + replace, fsynced."""
    tmp_path = path + ".compact-tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)
