"""Live OTLP export: POST spans to a collector over HTTP.

v0.9.0 exported OTLP/JSON spans to a file; this module sends them
live instead, to any collector that accepts OTLP/HTTP JSON (the
default endpoint is the standard one, http://localhost:4318/v1/traces).

Two ways to use it:

- One-shot: ``OtlpHttpExporter(endpoint).export_log(path)`` posts a
  whole log's spans in one request, retrying with exponential
  backoff on connection errors and on 429 / 5xx responses. A 4xx
  other than 429 is not retried: the collector rejected the payload,
  and sending it again will not help. ``export_log`` returns True
  only when the collector answered 2xx; the exporter keeps counts
  (exported_spans, dropped_spans, failed_posts) and the last error,
  so a caller can see what happened instead of guessing.
- Background thread: ``LiveExporter`` takes spans as they are
  recorded and posts them in batches (up to ``batch_size`` spans,
  or whatever has queued after ``flush_interval`` seconds). The
  Recorder starts one when it is given ``otlp_endpoint``: each kept
  step is enqueued as a span as it is written, and ``close()``
  flushes the queue before returning. A failed batch is counted as
  dropped, never retried forever — a flight recorder must not block
  the agent it is recording because its collector is down.

Spans come from ``otel.records_to_spans`` / ``record_to_span``, so a
live span is the same span the file exporter writes: same trace id
from the first record's hash, same span ids from the hash chain.
The one difference is the end time: a span sent live has no next
step yet, so it ends 1 ms after it starts (the file exporter can
look ahead to the next step's timestamp). Pure stdlib: urllib only.
"""

import json
import queue
import threading
import time
import urllib.error
import urllib.request

from .otel import export_otlp, record_to_span, records_to_spans

DEFAULT_OTLP_ENDPOINT = "http://localhost:4318/v1/traces"
DEFAULT_BATCH_SIZE = 100
DEFAULT_FLUSH_INTERVAL = 5.0
DEFAULT_MAX_RETRIES = 3
DEFAULT_TIMEOUT = 10.0
DEFAULT_BACKOFF_BASE = 0.5
_MAX_BACKOFF = 8.0

_SENTINEL = object()


def _payload_for_spans(spans, service_name="blackbox", version=None):
    """Spans wrapped in the OTLP/JSON resourceSpans shape."""
    from . import __version__
    return {
        "resourceSpans": [{
            "resource": {"attributes": [
                {"key": "service.name",
                 "value": {"stringValue": service_name}},
            ]},
            "scopeSpans": [{
                "scope": {"name": "blackbox",
                          "version": version or __version__},
                "spans": spans,
            }],
        }],
    }


class OtlpHttpExporter:
    """POST OTLP/JSON spans to a collector, with retry and backoff.

    max_retries is the number of retries after the first attempt
    (3 means up to 4 POSTs). backoff_base is the first wait in
    seconds; each retry doubles it, capped at 8 s.
    """

    def __init__(self, endpoint=DEFAULT_OTLP_ENDPOINT, timeout=DEFAULT_TIMEOUT,
                 max_retries=DEFAULT_MAX_RETRIES,
                 backoff_base=DEFAULT_BACKOFF_BASE,
                 service_name="blackbox"):
        self.endpoint = endpoint
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.service_name = service_name
        self.exported_spans = 0
        self.dropped_spans = 0
        self.posts = 0
        self.failed_posts = 0
        self.last_error = None

    def post_payload(self, payload, span_count=0):
        """POST one OTLP/JSON payload. True on a 2xx response."""
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            self.endpoint, data=body, method="POST",
            headers={"Content-Type": "application/json"})
        attempt = 0
        while True:
            self.posts += 1
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as resp:
                    resp.read()
                self.exported_spans += span_count
                return True
            except urllib.error.HTTPError as e:
                e.read()
                retryable = e.code == 429 or e.code >= 500
                self.last_error = "HTTP %s" % e.code
            except (urllib.error.URLError, OSError) as e:
                retryable = True
                self.last_error = str(e)
            if not retryable or attempt >= self.max_retries:
                self.failed_posts += 1
                self.dropped_spans += span_count
                return False
            wait = min(self.backoff_base * (2 ** attempt), _MAX_BACKOFF)
            time.sleep(wait)
            attempt += 1

    def export_spans(self, spans):
        """POST a list of span dicts. True on a 2xx response."""
        return self.post_payload(
            _payload_for_spans(spans, self.service_name),
            span_count=len(spans))

    def export_records(self, records):
        """POST spans built from records. True on a 2xx response."""
        return self.export_spans(records_to_spans(records))

    def export_log(self, path, causal=False, last=None, since=None,
                   until=None, now=None):
        """POST a whole log's spans (summary excluded, as in otel.py).

        With causal=True, span parents come from the causal trace
        reconstruction instead of the hash chain. last / since /
        until narrow the export to a window, the same window as
        `blackbox otel` — a bad bound, a reversed window, or --last
        combined with --since / --until raises ValueError. Returns
        True only when the collector answered 2xx.
        """
        data = export_otlp(path, self.service_name, causal=causal,
                           last=last, since=since, until=until, now=now)
        spans = data["resourceSpans"][0]["scopeSpans"][0]["spans"]
        return self.post_payload(data, span_count=len(spans))


class LiveExporter:
    """Background thread that batches spans and POSTs them live.

    Enqueue spans with enqueue_spans() (or one record at a time with
    enqueue_record()); the worker posts a batch when it reaches
    batch_size or when flush_interval seconds pass with spans queued.
    stop() flushes whatever is left and joins the thread, so nothing
    queued is lost on a clean shutdown. A failed batch is counted in
    dropped_spans and not requeued: the recorder must keep recording
    even when the collector is down.
    """

    def __init__(self, endpoint=DEFAULT_OTLP_ENDPOINT,
                 batch_size=DEFAULT_BATCH_SIZE,
                 flush_interval=DEFAULT_FLUSH_INTERVAL,
                 timeout=DEFAULT_TIMEOUT, max_retries=DEFAULT_MAX_RETRIES,
                 backoff_base=DEFAULT_BACKOFF_BASE,
                 service_name="blackbox"):
        self._exporter = OtlpHttpExporter(
            endpoint, timeout=timeout, max_retries=max_retries,
            backoff_base=backoff_base, service_name=service_name)
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        self._queue = queue.Queue()
        self._thread = None

    @property
    def endpoint(self):
        return self._exporter.endpoint

    @property
    def exported_spans(self):
        return self._exporter.exported_spans

    @property
    def dropped_spans(self):
        return self._exporter.dropped_spans

    @property
    def failed_posts(self):
        return self._exporter.failed_posts

    @property
    def last_error(self):
        return self._exporter.last_error

    def start(self):
        """Start the worker thread. Safe to call once."""
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name="blackbox-live-export")
        self._thread.start()

    def enqueue_spans(self, spans):
        """Queue spans for export. No-op before start() is not an error:
        spans queue up and go out once the thread starts."""
        for span in spans:
            self._queue.put(span)

    def enqueue_record(self, record, trace_id, end=None):
        """Queue one recorded step as a span."""
        self._queue.put(record_to_span(record, trace_id, end=end))

    def stop(self):
        """Flush the queue and stop the thread. Idempotent."""
        if self._thread is None:
            return
        self._queue.put(_SENTINEL)
        self._thread.join()
        self._thread = None

    def _flush(self, batch):
        if batch:
            self._exporter.export_spans(batch)

    def _run(self):
        batch = []
        while True:
            try:
                item = self._queue.get(timeout=self.flush_interval)
            except queue.Empty:
                self._flush(batch)
                batch = []
                continue
            if item is _SENTINEL:
                self._flush(batch)
                return
            batch.append(item)
            if len(batch) >= self.batch_size:
                self._flush(batch)
                batch = []

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.stop()
        return False
