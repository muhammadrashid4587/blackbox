"""Demo: a scripted fake agent run. No real model calls, no API keys.

The fake agent "researches" flight recorders for AI agents in 8 steps:
plan, search, read a result, plan again, fetch a page, read it, then
summarise. One tool call carries a fake API key on purpose, to show
that redaction strips it before the record is written.

Run:  make demo
Then:  python -m blackbox verify data/demo.jsonl
       python -m blackbox replay data/demo.jsonl
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from blackbox import Recorder

LOG = os.path.join(os.path.dirname(__file__), "..", "data", "demo.jsonl")


def main():
    with Recorder(LOG, task="demo: research flight recorders",
                  model="demo-model") as rec:
        rec.record("model_call", {
            "model": "demo-model",
            "prompt": "Plan how to research flight recorders for AI agents.",
        }, tokens=120)
        rec.record("tool_call", {
            "tool": "search",
            "args": {"query": "flight recorder AI agents observability"},
        })
        rec.record("tool_result", {
            "tool": "search",
            "result": "3 results, top hit: langfuse docs",
            "latency_ms": 84,
        })
        rec.record("model_call", {
            "model": "demo-model",
            "prompt": "Fetch the top result and note what it records.",
        }, tokens=210)
        rec.record("tool_call", {
            "tool": "fetch",
            "args": {
                "url": "https://example.com/langfuse",
                # Deliberately fake, and deliberately shaped like a real
                # key, so the demo proves redaction works.
                "api_key": "sk-proj-FAKEDEMOKEY1234567890",
            },
        })
        rec.record("tool_result", {
            "tool": "fetch",
            "result": "200 OK, 1842 bytes",
            "latency_ms": 132,
        })
        rec.record("model_call", {
            "model": "demo-model",
            "prompt": "Summarise the findings in three bullets.",
        }, tokens=180)
        rec.record("note", {
            "text": "Run complete. Findings written up by the caller.",
        })
    print("Wrote %s (8 steps + summary)" % os.path.normpath(LOG))


if __name__ == "__main__":
    main()
