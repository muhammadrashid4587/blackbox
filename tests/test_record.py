"""`blackbox record`: append one step to a log from the shell — the
same record the Recorder writes (redacted, hash-chained, fsynced),
so a step can be recorded without the MCP server or a Python script.
The step agrees with `blackbox span`, the MCP templates and the full
export; a supplied timestamp is taken as given (never invented) and
bad input exits non-zero without touching the log."""

import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone

from blackbox import Recorder
from blackbox.__main__ import main
from blackbox.otel import records_to_spans, session_span
from blackbox.recorder import _normalize_ts
from blackbox.replay import load_records
from blackbox.verify import verify


class RecordCliTest(unittest.TestCase):
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

    def tearDown(self):
        self.tmp.cleanup()

    def _run(self, *argv):
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = main(["record", self.path] + [str(a) for a in argv])
        return code, out.getvalue(), err.getvalue()

    def test_record_appends_step_to_new_log(self):
        path = os.path.join(self.tmp.name, "fresh.jsonl")
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = main(["record", path, "--type", "tool_call",
                         "--payload", '{"tool": "search", "args": '
                         '{"query": "mcp"}}'])
        self.assertEqual(code, 0)
        self.assertEqual(err.getvalue(), "")
        self.assertEqual(out.getvalue(), "recorded step 0 (tool_call)\n")
        records = load_records(path)
        self.assertEqual(len(records), 2)  # the step + a summary
        step = records[0]
        self.assertEqual(step["seq"], 0)
        self.assertEqual(step["type"], "tool_call")
        self.assertEqual(step["payload"],
                         {"tool": "search", "args": {"query": "mcp"}})
        self.assertEqual(step["prev_hash"], "0" * 64)
        self.assertEqual(records[1]["type"], "summary")
        self.assertEqual(records[1]["payload"]["events"], 1)
        self.assertTrue(verify(path)["ok"])

    def test_record_appends_to_existing_log_and_keeps_chain(self):
        code, out, err = self._run(
            "--type", "tool_result",
            "--payload", '{"tool": "search", "result": "3 results"}')
        self.assertEqual(code, 0)
        self.assertEqual(out, "recorded step 3 (tool_result)\n")
        self.assertEqual(err, "")
        records = load_records(self.path)
        self.assertEqual(len(records), 5)  # 4 steps + summary
        self.assertEqual(records[3]["seq"], 3)
        self.assertEqual(records[3]["type"], "tool_result")
        self.assertEqual(records[3]["prev_hash"], records[2]["hash"])
        self.assertEqual(records[4]["type"], "summary")
        self.assertEqual(records[4]["payload"]["events"], 4)
        self.assertTrue(verify(self.path)["ok"])

    def test_record_preserves_existing_task_and_model(self):
        code, out, err = self._run(
            "--type", "tool_call",
            "--payload", '{"tool": "search"}')
        self.assertEqual(code, 0)
        summary = load_records(self.path)[-1]["payload"]
        self.assertEqual(summary["task"], "t")
        self.assertEqual(summary["model"], "gpt-5")

    def test_record_flags_override_existing_task_and_model(self):
        code, out, err = self._run(
            "--type", "tool_call",
            "--payload", '{"tool": "search"}',
            "--task", "other", "--model", "gpt-5-mini")
        self.assertEqual(code, 0)
        summary = load_records(self.path)[-1]["payload"]
        self.assertEqual(summary["task"], "other")
        self.assertEqual(summary["model"], "gpt-5-mini")

    def test_record_explicit_ts_is_used_verbatim(self):
        code, out, err = self._run(
            "--type", "tool_call",
            "--payload", '{"tool": "search"}',
            "--ts", "2026-11-24T01:00:00Z")
        self.assertEqual(code, 0)
        records = load_records(self.path)
        self.assertEqual(records[3]["ts"], "2026-11-24T01:00:00Z")

    def test_record_explicit_ts_with_offset_normalises_to_utc(self):
        code, out, err = self._run(
            "--type", "tool_call",
            "--payload", '{"tool": "search"}',
            "--ts", "2026-11-24T03:00:00+02:00")
        self.assertEqual(code, 0)
        records = load_records(self.path)
        self.assertEqual(records[3]["ts"], "2026-11-24T01:00:00Z")

    def test_record_default_ts_is_now(self):
        before = datetime.now(timezone.utc)
        code, out, err = self._run(
            "--type", "tool_call",
            "--payload", '{"tool": "search"}')
        self.assertEqual(code, 0)
        ts = load_records(self.path)[3]["ts"]
        stamp = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        self.assertLessEqual(abs((stamp - before).total_seconds()), 60)

    def test_record_tokens_int(self):
        code, out, err = self._run(
            "--type", "tool_call",
            "--payload", '{"tool": "search"}', "--tokens", "120")
        self.assertEqual(code, 0)
        records = load_records(self.path)
        self.assertEqual(records[3]["tokens"], 120)
        self.assertEqual(records[4]["payload"]["total_tokens"],
                         120 + 1540 + 0)  # 1200 + 340 from setUp

    def test_record_tokens_split_prices_model_call(self):
        code, out, err = self._run(
            "--type", "model_call",
            "--payload", '{"model": "gpt-5", "prompt": "hi"}',
            "--tokens", '{"input": 1200, "output": 340}')
        self.assertEqual(code, 0)
        records = load_records(self.path)
        self.assertEqual(records[3]["tokens"],
                         {"input": 1200, "output": 340})
        summary = records[4]["payload"]
        # gpt-5: 2400 in + 680 out at $1.25/$10 per 1M = $0.0098.
        self.assertEqual(summary["cost_by_model"]["gpt-5"]["cost_usd"],
                         0.0098)
        self.assertEqual(summary["cost_usd"], 0.0098)

    def test_record_redacts_secrets(self):
        code, out, err = self._run(
            "--type", "tool_call",
            "--payload", '{"tool": "search", "api_key": "sk-live-123"}')
        self.assertEqual(code, 0)
        records = load_records(self.path)
        self.assertEqual(records[3]["payload"]["api_key"], "[REDACTED]")
        with open(self.path, encoding="utf-8") as f:
            self.assertNotIn("sk-live-123", f.read())

    def test_record_agrees_with_span_and_full_export(self):
        code, out, err = self._run(
            "--type", "model_call",
            "--payload", '{"model": "gpt-5", "prompt": "hi"}',
            "--tokens", '{"input": 1200, "output": 340}',
            "--ts", "2026-11-24T01:00:00Z")
        self.assertEqual(code, 0)
        records = load_records(self.path)
        span = session_span(records, 3)
        exported = records_to_spans(records)[3]
        self.assertEqual(span["traceId"], exported["traceId"])
        self.assertEqual(span["startTimeUnixNano"],
                         exported["startTimeUnixNano"])
        self.assertEqual(span["endTimeUnixNano"],
                         exported["endTimeUnixNano"])
        self.assertEqual(span["parentSpanId"], exported["parentSpanId"])
        # session_span adds blackbox.cost_usd on a priced model_call
        # (by design — see otel.py); the rest of the attributes agree.
        span_attrs = [a for a in span["attributes"]
                      if a["key"] != "blackbox.cost_usd"]
        self.assertEqual(span_attrs, exported["attributes"])
        cost_attrs = [a for a in span["attributes"]
                      if a["key"] == "blackbox.cost_usd"]
        self.assertEqual(len(cost_attrs), 1)
        self.assertEqual(cost_attrs[0]["value"]["doubleValue"], 0.0049)
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = main(["span", self.path, "3", "--json"])
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(out.getvalue()), span)

    def test_record_sqlite_backend(self):
        path = os.path.join(self.tmp.name, "run.db")
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = main(["record", path, "--type", "tool_call",
                         "--payload", '{"tool": "search"}',
                         "--task", "t", "--model", "gpt-5"])
        self.assertEqual(code, 0)
        self.assertEqual(out.getvalue(), "recorded step 0 (tool_call)\n")
        records = load_records(path)
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]["type"], "tool_call")
        self.assertTrue(verify(path)["ok"])
        # A second step resumes the .db log without breaking the chain.
        with redirect_stdout(out), redirect_stderr(err):
            code = main(["record", path, "--type", "tool_result",
                         "--payload", '{"tool": "search", '
                         '"result": "ok"}'])
        self.assertEqual(code, 0)
        self.assertTrue(verify(path)["ok"])
        self.assertEqual(load_records(path)[-1]["payload"]["events"], 2)

    def test_record_twice_in_a_row(self):
        for seq in (3, 4):
            code, out, err = self._run(
                "--type", "tool_call",
                "--payload", '{"tool": "search"}')
            self.assertEqual(code, 0)
            self.assertEqual(out, "recorded step %d (tool_call)\n" % seq)
        self.assertTrue(verify(self.path)["ok"])
        summary = load_records(self.path)[-1]["payload"]
        self.assertEqual(summary["events"], 5)
        self.assertEqual(summary["task"], "t")

    def test_bad_payload_json_is_an_error_and_leaves_log_untouched(self):
        before = load_records(self.path)
        code, out, err = self._run("--type", "tool_call",
                                   "--payload", "{not json")
        self.assertEqual(code, 2)
        self.assertEqual(out, "")
        self.assertIn("payload", err)
        self.assertEqual(load_records(self.path), before)

    def test_payload_must_be_an_object(self):
        code, out, err = self._run("--type", "tool_call",
                                   "--payload", '["search"]')
        self.assertEqual(code, 2)
        self.assertEqual(out, "")
        self.assertIn("payload", err)

    def test_bad_tokens_is_an_error(self):
        for bad in ("abc", "-5", '{"input": "many"}', "true", "[1, 2]"):
            code, out, err = self._run(
                "--type", "tool_call",
                "--payload", '{"tool": "search"}', "--tokens", bad)
            self.assertEqual(code, 2, bad)
            self.assertEqual(out, "", bad)
            self.assertIn("tokens", err, bad)

    def test_bad_ts_is_an_error(self):
        for bad in ("not-a-time", "2026-11-24T01:00:00", "yesterday"):
            code, out, err = self._run(
                "--type", "tool_call",
                "--payload", '{"tool": "search"}', "--ts", bad)
            self.assertEqual(code, 2, bad)
            self.assertEqual(out, "", bad)
            self.assertIn("ts", err, bad)

    def test_reserved_type_is_an_error(self):
        for bad in ("summary", "checkpoint", ""):
            code, out, err = self._run(
                "--type", bad, "--payload", '{"tool": "search"}')
            self.assertEqual(code, 2, repr(bad))
            self.assertEqual(out, "")
            self.assertIn("type", err)

    def test_missing_type_is_an_error(self):
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            with self.assertRaises(SystemExit) as caught:
                main(["record", self.path, "--payload", "{}"])
        self.assertEqual(caught.exception.code, 2)

    def test_bad_input_into_new_log_creates_nothing(self):
        path = os.path.join(self.tmp.name, "never.jsonl")
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = main(["record", path, "--type", "tool_call",
                         "--payload", "{not json"])
        self.assertEqual(code, 2)
        self.assertFalse(os.path.exists(path))


class RecorderTsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "run.jsonl")

    def tearDown(self):
        self.tmp.cleanup()

    def test_normalize_ts(self):
        self.assertEqual(_normalize_ts("2026-11-24T01:00:00Z"),
                         "2026-11-24T01:00:00Z")
        self.assertEqual(_normalize_ts("2026-11-24T03:00:00+02:00"),
                         "2026-11-24T01:00:00Z")
        self.assertEqual(_normalize_ts("2026-11-24T01:00:00.500Z"),
                         "2026-11-24T01:00:00.500000Z")

    def test_normalize_ts_rejects_bad_input(self):
        for bad in ("not-a-time", "2026-11-24T01:00:00", "", None, 123):
            with self.assertRaises(ValueError, msg=repr(bad)):
                _normalize_ts(bad)

    def test_record_with_ts(self):
        with Recorder(self.path, task="t") as rec:
            seq = rec.record("tool_call", {"tool": "search"},
                             ts="2026-11-24T01:00:00Z")
        self.assertEqual(seq, 0)
        records = load_records(self.path)
        self.assertEqual(records[0]["ts"], "2026-11-24T01:00:00Z")

    def test_record_with_bad_ts_raises(self):
        with Recorder(self.path, task="t") as rec:
            with self.assertRaises(ValueError):
                rec.record("tool_call", {"tool": "search"},
                           ts="2026-11-24T01:00:00")


if __name__ == "__main__":
    unittest.main()
