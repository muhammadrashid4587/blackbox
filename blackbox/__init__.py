"""blackbox: a flight recorder for AI agents.

Every step an agent takes - model calls, tool calls with arguments,
tool results, timestamps, token counts - is written to a tamper-evident
append-only log you can verify and replay.

Pure Python 3 standard library. No dependencies.
"""

__version__ = "0.41.0"

from .compact import compact_log
from .costs import cost_for, price_for, register_pricing
from .live_export import (DEFAULT_OTLP_ENDPOINT, LiveExporter,
                          OtlpHttpExporter)
from .mcp_server import MCPServer
from .otel import (export_otlp, export_otlp_file, export_otlp_metrics,
                   export_otlp_metrics_file, records_to_metrics,
                   records_to_spans, session_span, spans_to_records)
from .query import query_log, query_records
from .recorder import Recorder
from .replay import diff, diff_records, load_records, replay
from .sampling import (Sampler, SamplingDeclinedError, SamplingError,
                       SamplingNotSupportedError, SamplingResult,
                       StdioTransport)
from .tail_sampling import (DEFAULT_MAX_COST_USD, DEFAULT_SLOW_MS,
                            export_tail_sampled_otlp,
                            export_tail_sampled_otlp_file,
                            tail_sample_records)
from .trace import (build_trace, link_parents, session_trace,
                    trace_log, trace_subtree)
from .verify import verify
from .watch import watch_log

__all__ = ["DEFAULT_OTLP_ENDPOINT", "DEFAULT_MAX_COST_USD",
           "DEFAULT_SLOW_MS", "LiveExporter", "MCPServer",
           "OtlpHttpExporter", "Recorder", "Sampler",
           "SamplingDeclinedError",
           "SamplingError", "SamplingNotSupportedError", "SamplingResult",
           "StdioTransport", "build_trace", "compact_log", "cost_for",
           "diff", "diff_records", "export_otlp",
           "export_otlp_file", "export_otlp_metrics",
           "export_otlp_metrics_file", "export_tail_sampled_otlp",
           "export_tail_sampled_otlp_file", "link_parents", "load_records",
           "price_for",
           "query_log", "query_records", "records_to_metrics",
           "records_to_spans", "register_pricing",
           "replay", "session_span", "session_trace", "spans_to_records",
           "tail_sample_records", "trace_log", "trace_subtree",
           "verify", "watch_log"]
