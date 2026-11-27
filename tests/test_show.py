"""`blackbox show`: one step's raw record — the same record the
blackbox://session/record/{seq} template serves, readable without
an MCP client. The report form carries the transcript line plus the
hash chain, and --json prints the record itself; an unknown step is
an error, never an invented record. The summary record is a record
in the log, so it can be shown too — unlike a span or a transcript
line, which it has none of."""

import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout

from blackbox import Recorder
from blackbox.__main__ import main
from blackbox.mcp_server import MCPServer
from blackbox.replay import (format_record, load_records, session_record,
                             show_window)


def _request(msg_id, method, params=None):
    msg = {"jsonrpc": "2.0", "id": msg_id, "method": method}
    if params is not None:
        msg["params"] = params
    return msg


class ShowCliTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "run.jsonl")
        with Recorder(self.path, task="t", model="gpt-5") as rec:
            rec.record("model_call", {"model": "gpt-5", "prompt": "hi"},
                       tokens={"input": 1200, "output": 340})
            rec.record("tool_call", {"tool": "search",
                                     "args": {"query": "mcp"}})
            rec.record("tool_result", {"tool": "search",
                                       "result": "3 results"})
        self.records = load_records(self.path)

    def tearDown(self):
        self.tmp.cleanup()

    def _run(self, *argv, path=None):
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = main(["show", path or self.path]
                        + [str(a) for a in argv])
        return code, out.getvalue(), err.getvalue()

    def test_report_is_the_replay_line_plus_the_hash_chain(self):
        code, out, err = self._run(1)
        self.assertEqual(code, 0)
        self.assertEqual(err, "")
        record = session_record(self.records, 1)
        lines = out.splitlines()
        self.assertEqual(lines[0], format_record(record))
        self.assertEqual(lines[1], "    hash: %s" % record["hash"])
        self.assertEqual(lines[2], "    prev_hash: %s"
                         % record["prev_hash"])
        self.assertIn("tool_call", out)
        self.assertIn("search(query=mcp)", out)

    def test_json_is_the_raw_record(self):
        code, out, err = self._run(1, "--json")
        self.assertEqual(code, 0)
        self.assertEqual(err, "")
        self.assertEqual(json.loads(out),
                         session_record(self.records, 1))
        data = json.loads(out)
        self.assertEqual(data["seq"], 1)
        self.assertEqual(data["type"], "tool_call")
        self.assertEqual(data["payload"],
                         {"tool": "search", "args": {"query": "mcp"}})
        self.assertIn("hash", data)
        self.assertIn("prev_hash", data)

    def test_json_agrees_with_the_record_template(self):
        # The same record the blackbox://session/record/{seq}
        # template serves, byte for byte: the CLI and the template
        # always agree.
        server = MCPServer(self.path, task="t")
        try:
            server.handle_message(_request(
                1, "initialize", {"protocolVersion": "2024-11-05"}))
            response = server.handle_message(_request(
                2, "resources/read",
                {"uri": "blackbox://session/record/1"}))
        finally:
            server.close()
        template_text = response["result"]["contents"][0]["text"]
        code, out, err = self._run(1, "--json")
        self.assertEqual(code, 0)
        self.assertEqual(out, template_text + "\n")

    def test_summary_record_can_be_shown(self):
        # The summary is bookkeeping, not a step, so it has no span
        # and no transcript line — but it is a record in the log,
        # and the template serves it, so show does too.
        with Recorder(self.path, task="t", model="gpt-5"):
            pass  # closing writes the summary record (seq 3)
        code, out, err = self._run(3, "--json")
        self.assertEqual(code, 0)
        self.assertEqual(err, "")
        data = json.loads(out)
        self.assertEqual(data["seq"], 3)
        self.assertEqual(data["type"], "summary")
        code, out, err = self._run(3)
        self.assertEqual(code, 0)
        self.assertIn("summary", out)
        self.assertIn("hash:", out)

    def test_unknown_seq_is_an_error(self):
        code, out, err = self._run(99)
        self.assertEqual(code, 2)
        self.assertEqual(out, "")
        self.assertIn("no step 99", err)
        code, out, err = self._run(99, "--json")
        self.assertEqual(code, 2)
        self.assertEqual(out, "")
        self.assertIn("no step 99", err)

    def test_window_around_seq_is_in_log_order(self):
        code, out, err = self._run(1, "--before", "1", "--after", "1")
        self.assertEqual(code, 0)
        self.assertEqual(err, "")
        # One three-line block per record, in log order, with a
        # blank line between blocks (a model_call's transcript
        # line is two lines, so a block can be four lines).
        blocks = out.split("\n\n")
        self.assertEqual(len(blocks), 3)
        for seq, block in zip((0, 1, 2), blocks):
            record = session_record(self.records, seq)
            expected = ("%s\n    hash: %s\n    prev_hash: %s"
                        % (format_record(record), record["hash"],
                           record["prev_hash"]))
            self.assertEqual(block.rstrip("\n"), expected)

    def test_window_clamps_at_the_session_ends(self):
        code, out, err = self._run(0, "--before", "5")
        self.assertEqual(code, 0)
        self.assertIn("[0]", out)
        self.assertNotIn("[1]", out)
        code, out, err = self._run(2, "--after", "5")
        self.assertEqual(code, 0)
        self.assertIn("[2]", out)
        self.assertNotIn("[1]", out)

    def test_window_json_matches_show_window(self):
        code, out, err = self._run(1, "--before", "1", "--after", "1",
                                   "--json")
        self.assertEqual(code, 0)
        self.assertEqual(err, "")
        self.assertEqual(json.loads(out),
                         show_window(self.records, 1,
                                     before=1, after=1))
        data = json.loads(out)
        self.assertEqual(data["seq"], 1)
        self.assertEqual((data["before"], data["after"]), (1, 1))
        self.assertEqual(data["count"], 3)
        self.assertEqual(data["total"], 4)  # 3 steps + the summary
        self.assertEqual([r["seq"] for r in data["records"]],
                         [0, 1, 2])

    def test_window_can_include_the_summary_record(self):
        # The summary is a record in the log, so a window around
        # it carries the step before it too.
        with Recorder(self.path, task="t", model="gpt-5"):
            pass  # closing writes the summary record (seq 3)
        code, out, err = self._run(3, "--before", "1", "--json")
        self.assertEqual(code, 0)
        data = json.loads(out)
        self.assertEqual([r["seq"] for r in data["records"]], [2, 3])
        self.assertEqual(data["total"], 4)

    def test_window_narrows_the_session(self):
        # Every step was recorded just now, so a window starting in
        # 2020 keeps the whole session and one ending in 2020 keeps
        # none of it — the step is not in the session.
        code, out, err = self._run(0, "--since", "2020-01-01T00:00:00Z")
        self.assertEqual(code, 0)
        self.assertIn("[0]", out)
        code, out, err = self._run(0, "--until", "2020-01-01T00:00:00Z")
        self.assertEqual(code, 2)
        self.assertEqual(out, "")
        self.assertIn("no step 0", err)
        code, out, err = self._run(2, "--last", "1h")
        self.assertEqual(code, 0)
        self.assertIn("[2]", out)

    def test_window_errors_are_exit_code_2(self):
        for argv in (("--before", "-1"), ("--after", "-1"),
                     ("--before", "100"),
                     ("--since", "not-a-timestamp"),
                     ("--last", "not-a-duration"),
                     ("--since", "2026-11-24T02:00:00Z",
                      "--until", "2026-11-24T01:00:00Z"),
                     ("--last", "1h",
                      "--since", "2020-01-01T00:00:00Z")):
            code, out, err = self._run(1, *argv)
            self.assertEqual(code, 2, argv)
            self.assertEqual(out, "")

    def test_show_window_rejects_a_window_over_100_records(self):
        with self.assertRaises(ValueError):
            show_window(self.records, 1, before=50, after=50)

    def test_non_integer_seq_is_an_error(self):
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            with self.assertRaises(SystemExit) as caught:
                main(["show", self.path, "abc"])
        self.assertEqual(caught.exception.code, 2)

    def test_missing_file_is_an_error(self):
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = main(["show",
                         os.path.join(self.tmp.name, "nope.jsonl"), "0"])
        self.assertEqual(code, 1)
        self.assertEqual(out.getvalue(), "")
        self.assertIn("file not found", err.getvalue())

    def test_works_on_a_sqlite_store(self):
        db_path = os.path.join(self.tmp.name, "run.db")
        with Recorder(db_path, task="t", model="gpt-5") as rec:
            rec.record("tool_call", {"tool": "search",
                                     "args": {"query": "mcp"}})
        code, out, err = self._run(0, "--json", path=db_path)
        self.assertEqual(code, 0)
        self.assertEqual(err, "")
        data = json.loads(out)
        self.assertEqual(data["seq"], 0)
        self.assertEqual(data["type"], "tool_call")
        code, out, err = self._run(0, path=db_path)
        self.assertEqual(code, 0)
        self.assertIn("tool_call", out)
        self.assertIn("hash:", out)


if __name__ == "__main__":
    unittest.main()
