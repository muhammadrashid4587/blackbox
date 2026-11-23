"""blackbox: a flight recorder for AI agents.

Every step an agent takes - model calls, tool calls with arguments,
tool results, timestamps, token counts - is written to a tamper-evident
append-only log you can verify and replay.

Pure Python 3 standard library. No dependencies.
"""

__version__ = "0.11.0"

from .costs import cost_for, price_for, register_pricing
from .live_export import (DEFAULT_OTLP_ENDPOINT, LiveExporter,
                          OtlpHttpExporter)
from .mcp_server import MCPServer
from .otel import export_otlp, export_otlp_file, records_to_spans, spans_to_records
from .recorder import Recorder
from .replay import diff, load_records, replay
from .sampling import (Sampler, SamplingDeclinedError, SamplingError,
                       SamplingNotSupportedError, SamplingResult,
                       StdioTransport)
from .verify import verify

__all__ = ["DEFAULT_OTLP_ENDPOINT", "LiveExporter", "MCPServer",
           "OtlpHttpExporter", "Recorder", "Sampler", "SamplingDeclinedError",
           "SamplingError", "SamplingNotSupportedError", "SamplingResult",
           "StdioTransport", "cost_for", "diff", "export_otlp",
           "export_otlp_file", "load_records", "price_for",
           "records_to_spans", "register_pricing", "replay",
           "spans_to_records", "verify"]
