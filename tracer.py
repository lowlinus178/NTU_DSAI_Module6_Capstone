"""
tracer.py — Structured per-query trace logger.

Each query produces one Tracer instance. Steps are recorded with
timestamps and arbitrary payload dicts. On flush(), the complete
trace is appended as a single JSON line to traces/trace.log (NDJSON
format — one self-contained JSON object per line, easy to grep/parse).

Usage:
    tracer = Tracer(query="How many sick days do I get?")
    tracer.step("retrieve", chunks=[...], latency_ms=42)
    tracer.step("rerank", rerank_used=True, scores=[...], latency_ms=18)
    tracer.flush()
    trace_dict = tracer.to_dict()   # for the UI
"""

import json
import os
import time
import uuid
from datetime import datetime, timezone

TRACE_DIR = "traces"
TRACE_FILE = os.path.join(TRACE_DIR, "trace.log")


class Tracer:
    def __init__(self, query: str):
        self.trace_id = str(uuid.uuid4())[:8]           # short 8-char ID for readability
        self.query = query
        self.started_at = time.perf_counter()
        self.timestamp = datetime.now(timezone.utc).isoformat()
        self.steps: list[dict] = []
        self._step_start = time.perf_counter()           # tracks latency between steps

    # ── Core recording API ────────────────────────────────────────────────────

    def step(self, name: str, **payload) -> None:
        """
        Record a named pipeline step with an arbitrary payload.
        Automatically attaches:
          - elapsed_ms  : ms since the previous step (or trace start)
          - total_ms    : ms since trace start
        """
        now = time.perf_counter()
        elapsed_ms = round((now - self._step_start) * 1000, 1)
        total_ms = round((now - self.started_at) * 1000, 1)
        self._step_start = now

        self.steps.append({
            "step": name,
            "elapsed_ms": elapsed_ms,
            "total_ms": total_ms,
            **payload,
        })

    def to_dict(self) -> dict:
        """Return the complete trace as a serialisable dict."""
        total_ms = round((time.perf_counter() - self.started_at) * 1000, 1)
        return {
            "trace_id": self.trace_id,
            "timestamp": self.timestamp,
            "query": self.query,
            "total_ms": total_ms,
            "steps": self.steps,
        }

    def flush(self) -> None:
        """Append this trace as one JSON line to traces/trace.log."""
        os.makedirs(TRACE_DIR, exist_ok=True)
        record = self.to_dict()
        with open(TRACE_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
