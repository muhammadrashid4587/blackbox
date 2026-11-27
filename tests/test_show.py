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
from blackbox.replay import format_record, load_records, session_record


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
