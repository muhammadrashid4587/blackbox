"""blackbox: a flight recorder for AI agents.

Every step an agent takes - model calls, tool calls with arguments,
tool results, timestamps, token counts - is written to a tamper-evident
append-only log you can verify and replay.

Pure Python 3 standard library. No dependencies.
"""

from .recorder import Recorder
from .replay import diff, load_records, replay
from .verify import verify

__all__ = ["Recorder", "diff", "load_records", "replay", "verify"]
__version__ = "0.1.0"
