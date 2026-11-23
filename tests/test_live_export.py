"""Live OTLP export: spans actually arrive at a local collector.

A tiny HTTP server stands in for a collector: it records every
POST body it receives and can be told to fail the first N requests
with 500, so retry/backoff is tested against a real socket, not a
mock.
"""

import json
import os
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from blackbox import LiveExporter, OtlpHttpExporter, Recorder
from blackbox.__main__ import main as cli_main
from blackbox.live_export import DEFAULT_OTLP_ENDPOINT


class _CaptureHandler(BaseHTTPRequestHandler):
    server_state = None  # set per test: {"bodies": [], "fail_first": 0}

    def do_POST(self):
        state = self.server_state
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        state["bodies"].append(body)
        state["requests"] += 1
        if state["requests"] <= state["fail_first"]:
            self.send_response(500)
            self.end_headers()
            self.wfile.write(b"boom")
            return
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b"{}")

    def log_message(self, *args):
        pass


class LiveExportTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "run.jsonl")
        with Recorder(self.path, task="t", model="m") as rec:
            rec.record("model_call", {"model": "m", "prompt": "hi"},
                       tokens={"input": 10, "output": 5})
            rec.record("tool_call", {"tool": "search", "args": {"q": "x"}})
            rec.record("tool_result", {"tool": "search", "result": "ok"})
        self.state = {"bodies": [], "requests": 0, "fail_first": 0}
        handler = type("H", (_CaptureHandler,), {"server_state": self.state})
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self.thread = threading.Thread(target=self.server.serve_forever,
                                       daemon=True)
        self.thread.start()
        self.endpoint = "http://127.0.0.1:%d/v1/traces" % self.server.server_address[1]

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.tmp.cleanup()

    def _exporter(self, **kw):
        kw.setdefault("backoff_base", 0.01)
        return OtlpHttpExporter(self.endpoint, **kw)

    def _spans_received(self):
        spans = []
        for body in self.state["bodies"]:
            data = json.loads(body)
            spans.extend(data["resourceSpans"][0]["scopeSpans"][0]["spans"])
        return spans

    def test_default_endpoint_is_the_standard_otlp_http_one(self):
        self.assertEqual(DEFAULT_OTLP_ENDPOINT,
                         "http://localhost:4318/v1/traces")

    def test_export_log_posts_spans_to_collector(self):
        exporter = self._exporter()
        self.assertTrue(exporter.export_log(self.path))
        self.assertEqual(self.state["requests"], 1)
        spans = self._spans_received()
        self.assertEqual(len(spans), 3)  # 3 steps, summary not a span
        self.assertEqual([s["name"] for s in spans],
                         ["model_call m", "tool_call search",
                          "tool_result search"])
        self.assertEqual(exporter.exported_spans, 3)
        self.assertEqual(exporter.dropped_spans, 0)
        trace_id = spans[0]["traceId"]
        for span in spans:
            self.assertEqual(span["traceId"], trace_id)

    def test_retry_after_transient_500_then_succeeds(self):
        self.state["fail_first"] = 2
        exporter = self._exporter(max_retries=3)
        self.assertTrue(exporter.export_log(self.path))
        self.assertEqual(self.state["requests"], 3)
        # The server captured every attempt's body, including the two
        # failed ones; only the last (successful) one counts.
        last = json.loads(self.state["bodies"][-1])
        spans = last["resourceSpans"][0]["scopeSpans"][0]["spans"]
        self.assertEqual(len(spans), 3)
        self.assertEqual(exporter.failed_posts, 0)

    def test_gives_up_after_max_retries_and_counts_dropped(self):
        self.state["fail_first"] = 10
        exporter = self._exporter(max_retries=2)
        self.assertFalse(exporter.export_log(self.path))
        self.assertEqual(self.state["requests"], 3)  # 1 + 2 retries
        self.assertEqual(exporter.failed_posts, 1)
        self.assertEqual(exporter.dropped_spans, 3)
        self.assertEqual(exporter.last_error, "HTTP 500")

    def test_unreachable_collector_returns_false_not_raise(self):
        dead = OtlpHttpExporter("http://127.0.0.1:1/v1/traces",
                                max_retries=1, backoff_base=0.01, timeout=2)
        self.assertFalse(dead.export_log(self.path))
        self.assertEqual(dead.dropped_spans, 3)
        self.assertIsNotNone(dead.last_error)


class LiveExporterThreadTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "run.jsonl")
        with Recorder(self.path, task="t", model="m") as rec:
            rec.record("model_call", {"model": "m", "prompt": "hi"},
                       tokens={"input": 10, "output": 5})
            rec.record("tool_call", {"tool": "search", "args": {"q": "x"}})
        self.state = {"bodies": [], "requests": 0, "fail_first": 0}
        handler = type("H", (_CaptureHandler,), {"server_state": self.state})
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self.thread = threading.Thread(target=self.server.serve_forever,
                                       daemon=True)
        self.thread.start()
        self.endpoint = "http://127.0.0.1:%d/v1/traces" % self.server.server_address[1]

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.tmp.cleanup()

    def _spans_received(self):
        spans = []
        for body in self.state["bodies"]:
            data = json.loads(body)
            spans.extend(data["resourceSpans"][0]["scopeSpans"][0]["spans"])
        return spans

    def test_thread_batches_and_flushes_on_stop(self):
        from blackbox.otel import records_to_spans
        from blackbox.replay import load_records
        spans = records_to_spans(load_records(self.path))
        with LiveExporter(self.endpoint, batch_size=100,
                          flush_interval=30, backoff_base=0.01) as live:
            live.enqueue_spans(spans)
        self.assertEqual(self.state["requests"], 1)
        received = self._spans_received()
        self.assertEqual(len(received), 2)
        self.assertEqual(live.exported_spans, 2)

    def test_thread_posts_when_batch_size_reached(self):
        from blackbox.otel import records_to_spans
        from blackbox.replay import load_records
        spans = records_to_spans(load_records(self.path))
        live = LiveExporter(self.endpoint, batch_size=1,
                            flush_interval=30, backoff_base=0.01)
        live.start()
        live.enqueue_spans(spans)
        live.stop()
        self.assertEqual(self.state["requests"], 2)
        self.assertEqual(len(self._spans_received()), 2)

    def test_recorder_exports_live_as_it_records(self):
        live_path = os.path.join(self.tmp.name, "live.jsonl")
        with Recorder(live_path, task="t", model="m",
                      otlp_endpoint=self.endpoint,
                      otlp_flush_interval=30) as rec:
            rec.record("model_call", {"model": "m", "prompt": "hi"},
                       tokens={"input": 10, "output": 5})
            rec.record("tool_call", {"tool": "search", "args": {"q": "x"}})
        # close() flushes the exporter thread before returning.
        self.assertEqual(self.state["requests"], 1)
        received = self._spans_received()
        self.assertEqual(len(received), 2)
        self.assertEqual([s["name"] for s in received],
                         ["model_call m", "tool_call search"])
        self.assertEqual(received[0]["traceId"], received[1]["traceId"])

    def test_recorder_keeps_recording_when_collector_is_down(self):
        live_path = os.path.join(self.tmp.name, "down.jsonl")
        with Recorder(live_path, task="t", model="m",
                      otlp_endpoint="http://127.0.0.1:1/v1/traces",
                      otlp_flush_interval=0.1) as rec:
            rec.record("tool_call", {"tool": "search", "args": {"q": "x"}})
        # Recording itself must not raise or lose the step.
        with open(live_path) as f:
            records = [json.loads(line) for line in f]
        self.assertEqual(len(records), 2)  # step + summary
        self.assertEqual(records[0]["type"], "tool_call")


class ExportCliTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "run.jsonl")
        with Recorder(self.path, task="t", model="m") as rec:
            rec.record("tool_call", {"tool": "search", "args": {"q": "x"}})
        self.state = {"bodies": [], "requests": 0, "fail_first": 0}
        handler = type("H", (_CaptureHandler,), {"server_state": self.state})
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self.thread = threading.Thread(target=self.server.serve_forever,
                                       daemon=True)
        self.thread.start()
        self.endpoint = "http://127.0.0.1:%d/v1/traces" % self.server.server_address[1]

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.tmp.cleanup()

    def test_export_live_command_posts_and_reports(self):
        rc = cli_main(["export", self.path, "--live",
                       "--otlp-endpoint", self.endpoint,
                       "--max-retries", "1"])
        self.assertEqual(rc, 0)
        self.assertEqual(self.state["requests"], 1)
        body = json.loads(self.state["bodies"][0])
        spans = body["resourceSpans"][0]["scopeSpans"][0]["spans"]
        self.assertEqual(len(spans), 1)
        self.assertEqual(spans[0]["name"], "tool_call search")

    def test_export_live_command_fails_cleanly_when_collector_down(self):
        rc = cli_main(["export", self.path, "--live",
                       "--otlp-endpoint", "http://127.0.0.1:1/v1/traces",
                       "--max-retries", "0"])
        self.assertEqual(rc, 1)
        self.assertEqual(self.state["requests"], 0)


if __name__ == "__main__":
    unittest.main()
