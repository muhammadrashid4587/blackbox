"""Verify a log: replay the hash chain from scratch.

Works on both backends: a .db/.sqlite/.sqlite3 path verifies the
SQLite store, anything else a .jsonl log.

Returns a dict:

    {"ok": True, "records": 9, "first_bad_seq": None, "error": None}

or, on the first problem found:

    {"ok": False, "records": 9, "first_bad_seq": 4,
     "error": "hash mismatch at seq 4"}

What is checked, in order, for every record:
1. (JSONL only) The line parses as JSON. A torn final line (a
   crashed write) fails here.
2. seq is exactly the record's position in the log.
3. prev_hash equals the previous record's hash (64 zeros for seq 0).
4. The stored hash equals the hash recomputed from the record's content.
5. If the last record is a summary, its event count matches the number
   of records before it. Deleting records from the middle or the end
   breaks either the chain or this count.

A log with no summary still verifies: it just means the run never
closed cleanly (crash, Ctrl-C), which is not tampering.

A compacted log starts with a checkpoint record (type "checkpoint"),
the new genesis of the chain after `compact` removed old records.
It verifies the same way — the chain is replayed from the checkpoint
forward — and the result reports how many records the checkpoint
compacted away.
"""

import json

from .recorder import GENESIS_PREV_HASH, _hash_record
from .sqlite_store import is_sqlite_path, load_sqlite_records


def _check_chain(records):
    """Replay seq/prev_hash/hash linkage over ordered records.

    Returns (first_bad_seq, error), or (None, None) when the chain
    is intact.
    """
    prev_hash = GENESIS_PREV_HASH
    for position, record in enumerate(records):
        if record.get("seq") != position:
            return position, ("seq mismatch at position %d: expected %d, "
                              "got %r" % (position, position,
                                          record.get("seq")))
        if record.get("prev_hash") != prev_hash:
            return position, "prev_hash mismatch at seq %d" % position
        stored_hash = record.get("hash")
        recomputed = _hash_record({k: v for k, v in record.items()
                                   if k != "hash"})
        if stored_hash != recomputed:
            return position, ("hash mismatch at seq %d: record was modified"
                              % position)
        prev_hash = stored_hash
    return None, None


def _check_summary(records):
    """The summary's event count must match the records before it."""
    if records and records[-1].get("type") == "summary":
        expected = records[-1]["payload"].get("events")
        actual = sum(1 for r in records[:-1] if r["type"] != "summary")
        if expected != actual:
            return records[-1]["seq"], ("summary says %r events but log "
                                        "holds %d: records are missing"
                                        % (expected, actual))
    return None, None


def _fail(records_so_far, first_bad_seq, error):
    return {"ok": False, "records": records_so_far,
            "first_bad_seq": first_bad_seq, "error": error}


def _finish(records):
    bad_seq, error = _check_chain(records)
    if bad_seq is not None:
        # records counted up to and including the bad one, as when the
        # chain was checked line by line.
        return _fail(bad_seq + 1, bad_seq, error)
    bad_seq, error = _check_summary(records)
    if bad_seq is not None:
        return _fail(len(records), bad_seq, error)
    result = {"ok": True, "records": len(records),
              "first_bad_seq": None, "error": None}
    # A compacted log starts with a checkpoint record, the new genesis
    # of the chain: report how many records it compacted away.
    if records and records[0].get("type") == "checkpoint":
        result["compacted"] = (records[0].get("payload") or {}).get(
            "compacted_count", 0) or 0
        result["checkpoint_hash"] = records[0].get("hash")
    else:
        result["compacted"] = 0
        result["checkpoint_hash"] = None
    return result


def verify(path):
    if is_sqlite_path(path):
        return _verify_sqlite(path)
    return _verify_jsonl(path)


def _verify_sqlite(path):
    try:
        records = load_sqlite_records(path)
    except FileNotFoundError:
        return {"ok": False, "records": 0, "first_bad_seq": None,
                "error": "file not found: %s" % path}
    except ValueError as exc:
        return {"ok": False, "records": 0, "first_bad_seq": None,
                "error": str(exc)}
    return _finish(records)


def _verify_jsonl(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            lines = f.read().splitlines()
    except FileNotFoundError:
        return {"ok": False, "records": 0, "first_bad_seq": None,
                "error": "file not found: %s" % path}

    parsed = []
    for position, line in enumerate(lines):
        if not line.strip():
            return _fail(len(parsed), position,
                         "blank line at position %d" % position)
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            return _fail(len(parsed), position,
                         "unparseable line at position %d (torn write?)"
                         % position)
        parsed.append(record)

    return _finish(parsed)


def verify_cli(path):
    result = verify(path)
    if result["ok"]:
        message = "OK: %d records, hash chain intact" % result["records"]
        if result.get("compacted"):
            message += (" (compacted log: %d records removed, "
                        "checkpoint %s)"
                        % (result["compacted"],
                           result["checkpoint_hash"][:12]))
        print(message)
        return 0
    print("FAIL: %s" % result["error"])
    return 1
