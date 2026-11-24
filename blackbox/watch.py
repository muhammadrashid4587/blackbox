"""blackbox watch: tail a live log and print new steps as they land.

    python -m blackbox watch data/run.jsonl
    python -m blackbox watch data/run.jsonl --from-start

A running agent's flight recorder can be followed in real time: each
new step is printed as soon as it lands, in the same format as the
`blackbox replay` transcript. By default watching starts at the end
of the log and prints only new steps; --from-start replays the
existing steps first, then follows. --type keeps only steps of that
type (repeatable), --errors-only only steps that look like errors —
the same error rule as `blackbox query` — --model keeps only
model_call steps for one model and --tool keeps only tool_call /
tool_result steps for one tool, both the same match rules as
`blackbox query`, so a busy run can be followed for just the steps
you care about. The summary record is printed by default, but under
a filter it follows the same rule as `blackbox query`: it is kept
only when --type summary is given explicitly, and never under
--errors-only, --model or --tool.

Both backends work: a .jsonl log is tailed by byte offset, a .db
store by polling for rows past the last printed seq. A log that
shrinks while it is watched — a Recorder resuming drops the trailing
summary, a rotation replaces the file — is re-read from the start,
and steps already printed are skipped (each record's hash is unique
in the chain, and a resumed step reuses the dropped summary's seq,
so seq alone cannot tell a new step from an old one), so nothing is
printed twice and nothing new is missed. The shrink is not always
visible — a resume that rewrites and re-appends between two polls
leaves the file larger than the old offset, so the byte offset goes
stale and the next read starts mid-line. A line that fails to parse
is therefore treated as a possible desync, not just a bad line: the
log is re-read from the start, which recovers the missed steps, and
a genuinely malformed line is reported once and skipped, never a
crash. Ctrl-C stops cleanly with exit code 0.

Polling, not inotify: the tool is pure stdlib and must work wherever
the log lives, including a mount where file-change notifications do
not fire.
"""

import hashlib
import json
import os
import sqlite3
import sys
import time
from urllib.parse import quote

from .query import _is_error
from .replay import format_record
from .sqlite_store import _row_to_record, is_sqlite_path

DEFAULT_POLL_INTERVAL = 0.2
_READ_CHUNK = 65536


def _record_identity(record, line):
    """What identifies a printed record: its hash-chain hash, or the
    raw line when the record carries none. Hashes are unique per
    record, so an already-printed step is never printed twice, even
    after the log is re-read from the start."""
    return record.get("hash") or hashlib.sha256(line).hexdigest()


def _is_record(record):
    """True when a parsed JSON value has what format_record needs:
    a seq, a timestamp, a type, a payload object and tokens that are
    absent, an int or an input/output split."""
    if not isinstance(record, dict):
        return False
    if not isinstance(record.get("seq"), int):
        return False
    if not isinstance(record.get("ts"), str):
        return False
    if not isinstance(record.get("type"), str):
        return False
    if not isinstance(record.get("payload", {}), dict):
        return False
    tokens = record.get("tokens")
    if tokens is not None and not isinstance(tokens, (int, dict)):
        return False
    return True


def _matches_filter(record, types, errors_only, model=None, tool=None):
    """Whether a record passes the watch filters: --type keeps only
    steps of a listed type (the summary is kept only when "summary"
    is listed, the same rule as `blackbox query`), --errors-only
    keeps only steps that look like errors by the query's rule,
    --model keeps only model_call steps for one model and --tool
    keeps only tool_call / tool_result steps for one tool — both
    matched case-insensitively on the payload, the same rules as
    `blackbox query`, so the two commands always agree. The summary
    is never a model_call or a tool_call, so it is dropped under
    --model or --tool. Filters combine with AND, like
    `blackbox query`."""
    if types is not None and record["type"] not in types:
        return False
    payload = record.get("payload") or {}
    if model is not None:
        if record["type"] != "model_call":
            return False
        if str(payload.get("model", "")).lower() != model.lower():
            return False
    if tool is not None:
        if record["type"] not in ("tool_call", "tool_result"):
            return False
        if str(payload.get("tool", "")).lower() != tool.lower():
            return False
    if errors_only and not _is_error(record):
        return False
    return True


def _print_record(record, out):
    out.write(format_record(record) + "\n")
    out.flush()


class _JsonlTail:
    """Tail a .jsonl log by byte offset."""

    def __init__(self, path):
        self.path = path
        self.f = None
        self.offset = 0
        self.inode = None
        self.buffer = b""
        self.lineno = 0
        self.reported_bad = set()

    def close(self):
        if self.f is not None:
            self.f.close()
            self.f = None

    def _open(self):
        self.f = open(self.path, "rb")
        self.inode = os.fstat(self.f.fileno()).st_ino
        self.offset = 0
        self.buffer = b""
        self.lineno = 0

    def scan_existing(self):
        """Lines and record hashes in the file now, without printing.

        Used when watching starts at the end: the hashes seed the
        printed set, so a later re-read (the file shrank) skips the
        steps that were already there, and the line count lets a
        malformed appended line be reported by its line number.
        """
        count = 0
        hashes = set()
        with open(self.path, "rb") as f:
            for line in f:
                count += line.count(b"\n")
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except (json.JSONDecodeError, ValueError):
                    continue
                if isinstance(record, dict) and record.get("hash"):
                    hashes.add(record["hash"])
        return count, hashes

    def seek_end(self):
        """Start following from the end: nothing already in the log
        is printed. Returns the hashes of the existing records, so
        a later re-read skips them."""
        self.lineno, hashes = self.scan_existing()
        self.f.seek(0, os.SEEK_END)
        self.offset = os.path.getsize(self.path)
        return hashes

    def poll(self, printed, out, err, types=None, errors_only=False,
             model=None, tool=None):
        """One check for new steps. Returns how many were printed."""
        try:
            st = os.stat(self.path)
        except FileNotFoundError:
            # The file vanished mid-watch (a rotation in progress);
            # the next poll re-opens it when it reappears.
            return 0
        if (self.f is None or st.st_ino != self.inode
                or st.st_size < self.offset):
            # Rotated, replaced, or truncated: a Recorder resuming
            # drops the trailing summary, so the file shrinks by one
            # line. Re-open and re-read from the start; records
            # already printed are skipped by hash, so nothing is
            # printed twice and the new steps still land.
            if self.f is not None:
                self.f.close()
                if st.st_size < self.offset:
                    err.write("watch: %s shrank — re-reading from "
                              "the start\n" % self.path)
                else:
                    err.write("watch: %s was replaced — re-reading "
                              "from the start\n" % self.path)
                err.flush()
            self._open()
        data = self.f.read()
        if not data:
            return 0
        self.offset += len(data)
        self.buffer += data
        count, needs_reread = self._drain(printed, out, err,
                                          reread_on_bad=True,
                                          types=types,
                                          errors_only=errors_only,
                                          model=model, tool=tool)
        if needs_reread:
            count += self._reread(printed, out, err, types=types,
                                  errors_only=errors_only,
                                  model=model, tool=tool)
        return count

    def _drain(self, printed, out, err, reread_on_bad,
               types=None, errors_only=False, model=None, tool=None):
        """Print every complete line in the buffer.

        Returns (count, needs_reread): needs_reread when a line
        failed to parse — the byte offset may have gone stale
        because the log was rewritten between polls, so the caller
        re-reads from the start instead of trusting the offset.
        A record that fails the filters is still marked printed,
        so a later re-read skips it instead of re-checking it —
        but it is not counted, so max_records counts printed
        steps, not skipped ones.
        """
        count = 0
        while b"\n" in self.buffer:
            line, self.buffer = self.buffer.split(b"\n", 1)
            self.lineno += 1
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                if reread_on_bad:
                    return count, True
                self._report_bad(line, "not valid JSON", err)
                continue
            if not _is_record(record):
                if reread_on_bad:
                    return count, True
                self._report_bad(line, "not a blackbox record", err)
                continue
            identity = _record_identity(record, line)
            if identity in printed:
                continue
            printed.add(identity)
            if not _matches_filter(record, types, errors_only,
                                   model=model, tool=tool):
                continue
            _print_record(record, out)
            count += 1
        return count, False

    def _report_bad(self, line, reason, err):
        """Report a malformed line once: the same line is seen again
        on the re-read that a desync triggers, and must not be
        reported twice."""
        key = hashlib.sha256(line).hexdigest()
        if key in self.reported_bad:
            return
        self.reported_bad.add(key)
        err.write("watch: line %d: %s — skipping\n"
                  % (self.lineno, reason))
        err.flush()

    def _reread(self, printed, out, err, types=None, errors_only=False,
                model=None, tool=None):
        """Re-read the whole log from the start: the offset went
        stale. Records already printed are skipped by hash, so the
        missed steps land and nothing is printed twice; a malformed
        line is reported once and skipped, never another re-read."""
        self.close()
        self._open()
        data = self.f.read()
        self.offset = len(data) if data else 0
        self.buffer = data if data else b""
        if not data:
            return 0
        # The buffer holds the whole file, so a trailing partial line
        # waits for its newline instead of being reported.
        count, _ = self._drain(printed, out, err, reread_on_bad=False,
                               types=types, errors_only=errors_only,
                               model=model, tool=tool)
        return count


class _DbTail:
    """Tail a .db store by polling for rows past the last printed seq."""

    def __init__(self, path):
        self.path = path
        self.conn = None
        self.last_seq = -1
        self._last_error = None

    def close(self):
        if self.conn is not None:
            self.conn.close()
            self.conn = None

    def _open(self):
        # Read-only, autocommit: every query sees the latest
        # committed rows, and watching never writes to the store.
        self.conn = sqlite3.connect(
            "file:%s?mode=ro" % quote(self.path), uri=True)
        self.conn.isolation_level = None

    def existing_hashes(self):
        """Hashes of every row in the store now, and the highest seq.

        Used when watching starts at the end: the hashes seed the
        printed set, so a later re-read skips the steps that were
        already there. A missing records table means the store is
        empty (a Recorder has not opened it yet), not an error.
        """
        try:
            rows = self.conn.execute(
                "SELECT seq, hash FROM records ORDER BY seq").fetchall()
        except sqlite3.DatabaseError:
            return set(), -1
        if not rows:
            return set(), -1
        return {row[1] for row in rows}, rows[-1][0]

    def poll(self, printed, out, err, types=None, errors_only=False,
             model=None, tool=None):
        """One check for new steps. Returns how many were printed.

        The last printed seq advances past every valid record,
        printed or filtered out, so a filtered step is not
        re-fetched on every poll.
        """
        try:
            max_seq = self.conn.execute(
                "SELECT MAX(seq) FROM records").fetchone()[0]
        except sqlite3.DatabaseError as exc:
            message = str(exc)
            if "no such table" in message:
                # The store has no records table yet — a Recorder
                # has not opened it. Not an error; say so once.
                if self._last_error != "no-table":
                    err.write("watch: %s has no records yet — "
                              "waiting\n" % self.path)
                    err.flush()
                    self._last_error = "no-table"
                return 0
            if self._last_error != message:
                err.write("watch: cannot read %s: %s\n"
                          % (self.path, message))
                err.flush()
                self._last_error = message
            return 0
        self._last_error = None
        if max_seq is None:
            # The store is empty (every row was removed): start over.
            self.last_seq = -1
            return 0
        if max_seq < self.last_seq:
            # Either a resume (the trailing summary row was dropped)
            # or a rewrite (compaction renumbered the log). The row
            # at the new max seq tells them apart: a resumed log's
            # last step is a record already printed; a rewritten
            # log's records all carry new hashes.
            row = self.conn.execute(
                "SELECT hash FROM records WHERE seq = ?",
                (max_seq,)).fetchone()
            if row is not None and row[0] in printed:
                self.last_seq = max_seq
            else:
                err.write("watch: %s was rewritten — re-reading from "
                          "the start\n" % self.path)
                err.flush()
                self.last_seq = -1
        try:
            rows = self.conn.execute(
                "SELECT seq, ts, type, payload, tokens, prev_hash, "
                "hash FROM records WHERE seq >= ? ORDER BY seq",
                (self.last_seq,)).fetchall()
        except sqlite3.DatabaseError as exc:
            err.write("watch: cannot read %s: %s\n" % (self.path, exc))
            err.flush()
            return 0
        printed_now = 0
        for row in rows:
            try:
                record = _row_to_record(row)
            except (ValueError, TypeError, IndexError):
                err.write("watch: seq %s: unreadable row — skipping\n"
                          % (row[0],))
                err.flush()
                continue
            if not _is_record(record):
                err.write("watch: seq %s: not a blackbox record — "
                          "skipping\n" % (row[0],))
                err.flush()
                continue
            identity = _record_identity(record, b"")
            if identity in printed:
                continue
            printed.add(identity)
            self.last_seq = max(self.last_seq, record["seq"])
            if not _matches_filter(record, types, errors_only,
                                   model=model, tool=tool):
                continue
            _print_record(record, out)
            printed_now += 1
        return printed_now


def watch_log(path, from_start=False,
              poll_interval=DEFAULT_POLL_INTERVAL,
              output=None, error_output=None, stop_after=None,
              max_records=None, types=None, errors_only=False,
              model=None, tool=None):
    """Follow a log, printing each new step's transcript line as it
    lands, in the same format `blackbox replay` prints.

    By default watching starts at the end of the log and prints only
    new steps; `from_start` replays the existing steps first, then
    follows. Works on .jsonl logs and .db stores. A malformed
    appended line is reported on stderr and skipped, never a crash.
    A log that shrinks or is replaced while watched is re-read from
    the start, with already-printed steps skipped by record hash.

    `types` keeps only steps of a listed type (the summary is kept
    only when "summary" is listed, the same rule as `blackbox
    query`); `errors_only` keeps only steps that look like errors
    by the query's rule; `model` keeps only model_call steps for
    one model and `tool` keeps only tool_call / tool_result steps
    for one tool, both matched case-insensitively on the payload,
    the same rules as `blackbox query`. Filters combine with AND,
    and a filtered step is still marked seen, so it is not
    re-checked on a re-read. `max_records` counts printed steps,
    not skipped ones.

    `stop_after` (seconds) and `max_records` stop the loop and
    return: test hooks, and the only way to stop besides Ctrl-C.
    Returns the number of records printed. Raises KeyboardInterrupt
    on Ctrl-C — the caller turns that into a clean exit.
    """
    if poll_interval <= 0:
        raise ValueError("poll_interval must be positive")
    out = output if output is not None else sys.stdout
    err = error_output if error_output is not None else sys.stderr
    deadline = None if stop_after is None else time.monotonic() + stop_after

    def _time_left():
        return deadline is None or time.monotonic() < deadline

    # A log that does not exist yet — a run about to start — is
    # waited for, not an error.
    waited = False
    while not os.path.exists(path):
        if not waited:
            err.write("watch: waiting for %s to appear...\n" % path)
            err.flush()
            waited = True
        if not _time_left():
            return 0
        time.sleep(poll_interval)

    printed = set()
    if is_sqlite_path(path):
        tail = _DbTail(path)
        tail._open()
        if from_start:
            tail.last_seq = -1
        else:
            printed, tail.last_seq = tail.existing_hashes()
    else:
        tail = _JsonlTail(path)
        tail._open()
        if from_start:
            tail.lineno = 0
        else:
            printed = tail.seek_end()
    count = 0
    try:
        while _time_left():
            count += tail.poll(printed, out, err, types=types,
                               errors_only=errors_only,
                               model=model, tool=tool)
            if max_records is not None and count >= max_records:
                return count
            if not _time_left():
                return count
            time.sleep(poll_interval)
        return count
    finally:
        tail.close()
