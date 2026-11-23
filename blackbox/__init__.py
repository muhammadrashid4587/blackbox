"""blackbox: a flight recorder for AI agents.

Every step an agent takes - model calls, tool calls with arguments,
tool results, timestamps, token counts - is written to a tamper-evident
append-only log you can verify and replay.

Pure Python 3 standard library. No dependencies.
"""

__version__ = "0.13.0"

from .costs import cost_for, price_for, register_pricing
from .live_export import (DEFAULT_OTLP_ENDPOINT, LiveExporter,
                          OtlpHttpExporter)
from .mcp_server import MCPServer
from .otel import (export_otlp, export_otlp_file, export_otlp_metrics,
                   export_otlp_metrics_file, records_to_metrics,
                   records_to_spans, spans_to_records)
from .recorder import Recorder
from .replay import diff, load_records, replay
from .sampling import (Sampler, SamplingDeclinedError, SamplingError,
                       SamplingNotSupportedError, SamplingResult,
                       StdioTransport)
from .tail_sampling import (DEFAULT_MAX_COST_USD, DEFAULT_SLOW_MS,
                            export_tail_sampled_otlp,
                            export_tail_sampled_otlp_file,
                            tail_sample_records)
from .verify import verify

__all__ = ["DEFAULT_OTLP_ENDPOINT", "DEFAULT_MAX_COST_USD",
           "DEFAULT_SLOW_MS", "LiveExporter", "MCPServer",
           "OtlpHttpExporter", "Recorder", "Sampler",
           "SamplingDeclinedError",
           "SamplingError", "SamplingNotSupportedError", "SamplingResult",
           "StdioTransport", "cost_for", "diff", "export_otlp",
           "export_otlp_file", "export_otlp_metrics",
           "export_otlp_metrics_file", "export_tail_sampled_otlp",
           "export_tail_sampled_otlp_file", "load_records", "price_for",
           "records_to_metrics", "records_to_spans", "register_pricing",
           "replay", "spans_to_records", "tail_sample_records", "verify"]
