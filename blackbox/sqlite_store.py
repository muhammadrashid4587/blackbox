"""SQLite store for blackbox records.

The SQLite backend stores exactly the same record dicts as the JSONL
log - seq, ts, type, payload, tokens, prev_hash, hash - so the hash
chain is backend-independent: a record's hash is computed over the
same canonical JSON either way, and verify replays the same chain.

Schema:

    records(seq INTEGER PRIMARY KEY, ts TEXT, type TEXT,
            payload TEXT, tokens TEXT, prev_hash TEXT, hash TEXT)

payload and tokens are stored as JSON text (tokens is NULL when the
record has no token count), so a row round-trips to the exact record
dict that was hashed.

Durability: journal_mode=WAL with synchronous=FULL, one commit per
record - the SQLite equivalent of the JSONL backend's fsync per
record. close() checkpoints the WAL, so the .db file alone holds the
whole run.
"""

import json
import os
import sqlite3

SQLITE_SUFFIXES = (".db", ".sqlite", ".sqlite3")

SCHEMA = """
CREATE TABLE IF NOT EXISTS records (
    seq INTEGER PRIMARY KEY,
    ts TEXT NOT NULL,
    type TEXT NOT NULL,
    payload TEXT NOT NULL,
    tokens TEXT,
    prev_hash TEXT NOT NULL,
    hash TEXT NOT NULL
)
"""

_COLUMNS = "seq, ts, type, payload, tokens, prev_hash, hash"


def is_sqlite_path(path):
    """True when the path names a SQLite store (.db/.sqlite/.sqlite3)."""
    return os.path.splitext(path)[1].lower() in SQLITE_SUFFIXES


def _row_to_record(row):
    seq, ts, type_, payload, tokens, prev_hash, hash_ = row
    return {
        "seq": seq,
        "ts": ts,
        "type": type_,
        "payload": json.loads(payload),
        "tokens": None if tokens is None else json.loads(tokens),
        "prev_hash": prev_hash,
        "hash": hash_,
    }


def _select_all(conn):
    rows = conn.execute(
        "SELECT %s FROM records ORDER BY seq" % _COLUMNS).fetchall()
    return [_row_to_record(row) for row in rows]


def load_sqlite_records(path):
    """Read every record from a SQLite store, in seq order.

    Raises FileNotFoundError for a missing file and ValueError when the
    file exists but is not a blackbox database.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    conn = sqlite3.connect(path)
    try:
        return _select_all(conn)
    except sqlite3.DatabaseError as exc:
        raise ValueError("not a blackbox database: %s" % path) from exc
    finally:
        conn.close()


class SQLiteStore:
    """Append-only record store backed by one SQLite file."""

    def __init__(self, path):
        parent = os.path.dirname(os.path.abspath(path))
        os.makedirs(parent, exist_ok=True)
        self.path = path
        self.conn = sqlite3.connect(path)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=FULL")
        self.conn.execute(SCHEMA)
        self.conn.commit()

    def append(self, record):
        """Insert one record and commit: durable before returning."""
        self.conn.execute(
            "INSERT INTO records (%s) VALUES (?, ?, ?, ?, ?, ?, ?)"
            % _COLUMNS,
            (record["seq"], record["ts"], record["type"],
             json.dumps(record["payload"], sort_keys=True),
             None if record["tokens"] is None else json.dumps(
                 record["tokens"], sort_keys=True),
             record["prev_hash"], record["hash"]))
        self.conn.commit()

    def read_all(self):
        return _select_all(self.conn)

    def has_rows(self):
        return self.conn.execute(
            "SELECT COUNT(*) FROM records").fetchone()[0] > 0

    def delete_last(self):
        """Drop the newest record (the trailing summary, on resume)."""
        self.conn.execute(
            "DELETE FROM records WHERE seq = "
            "(SELECT MAX(seq) FROM records)")
        self.conn.commit()

    def close(self):
        """Checkpoint the WAL into the .db file, then close."""
        self.conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        self.conn.commit()
        self.conn.close()
