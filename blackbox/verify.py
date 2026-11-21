"""Verify a log: replay the hash chain from scratch.

Returns a dict:

    {"ok": True, "records": 9, "first_bad_seq": None, "error": None}

or, on the first problem found:

    {"ok": False, "records": 9, "first_bad_seq": 4,
     "error": "hash mismatch at seq 4"}

What is checked, in order, for every line:
1. The line parses as JSON. A torn final line (a crashed write) fails here.
2. seq is exactly the line's position in the file.
3. prev_hash equals the previous record's hash (64 zeros for seq 0).
4. The stored hash equals the hash recomputed from the record's content.
5. If the last record is a summary, its event count matches the number
   of records before it. Deleting records from the middle or the end
   breaks either the chain or this count.

A log with no summary still verifies: it just means the run never
closed cleanly (crash, Ctrl-C), which is not tampering.
"""

import hashlib
import json

from .recorder import GENESIS_PREV_HASH, _hash_record


def verify(path):
    records = 0
    prev_hash = GENESIS_PREV_HASH
    parsed = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            lines = f.read().splitlines()
    except FileNotFoundError:
        return {"ok": False, "records": 0, "first_bad_seq": None,
                "error": "file not found: %s" % path}

    for position, line in enumerate(lines):
        if not line.strip():
            return {"ok": False, "records": records, "first_bad_seq": position,
                    "error": "blank line at position %d" % position}
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            return {"ok": False, "records": records, "first_bad_seq": position,
                    "error": "unparseable line at position %d (torn write?)" % position}
        records += 1
        if record.get("seq") != position:
            return {"ok": False, "records": records, "first_bad_seq": position,
                    "error": "seq mismatch at position %d: expected %d, got %r"
                            % (position, position, record.get("seq"))}
        if record.get("prev_hash") != prev_hash:
            return {"ok": False, "records": records, "first_bad_seq": position,
                    "error": "prev_hash mismatch at seq %d" % position}
        stored_hash = record.get("hash")
        recomputed = _hash_record({k: v for k, v in record.items() if k != "hash"})
        if stored_hash != recomputed:
            return {"ok": False, "records": records, "first_bad_seq": position,
                    "error": "hash mismatch at seq %d: record was modified" % position}
        prev_hash = stored_hash
        parsed.append(record)

    if parsed and parsed[-1].get("type") == "summary":
        expected = parsed[-1]["payload"].get("events")
        actual = sum(1 for r in parsed[:-1] if r["type"] != "summary")
        if expected != actual:
            return {"ok": False, "records": records,
                    "first_bad_seq": parsed[-1]["seq"],
                    "error": "summary says %r events but log holds %d: "
                             "records are missing" % (expected, actual)}

    return {"ok": True, "records": records, "first_bad_seq": None, "error": None}


def verify_cli(path):
    result = verify(path)
    if result["ok"]:
        print("OK: %d records, hash chain intact" % result["records"])
        return 0
    print("FAIL: %s" % result["error"])
    return 1
