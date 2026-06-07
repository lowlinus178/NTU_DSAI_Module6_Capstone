"""
eval_runner.py — Runs the 10-question eval suite after every Q&A.

Runs synchronously (Streamlit background threads cannot safely update
session_state). LLM-judge scoring is skipped during eval runs to keep
runtime manageable — heuristic scores are used, which are instant.
Results are written to traces/metrics.log (one JSON line per run).
"""

import json
import os
import time
import uuid
from datetime import datetime, timezone

from eval_suite import EVAL_QUESTIONS
from evaluator import heuristic_scores
from rag import answer as rag_answer

METRICS_LOG = os.path.join("traces", "metrics.log")
METRICS = ("relevance", "accuracy", "faithfulness", "completeness")


def _run_eval(collection) -> dict:
    """Run all 10 eval questions with heuristic scoring only (fast, no extra LLM calls)."""
    run_id = str(uuid.uuid4())[:8]
    timestamp = datetime.now(timezone.utc).isoformat()
    question_results = []

    for eq in EVAL_QUESTIONS:
        t0 = time.perf_counter()
        result = rag_answer(eq["question"], collection)   # no tracer
        latency_ms = round((time.perf_counter() - t0) * 1000, 1)

        h = heuristic_scores(
            question=eq["question"],
            answer=result["answer"],
            expected_answer=eq["expected_answer"],
            keywords=eq["keywords"],
            source_doc=eq["source_doc"],
            retrieved_chunks=result["sources"],
        )

        question_results.append({
            "id":              eq["id"],
            "question":        eq["question"],
            "expected_answer": eq["expected_answer"],
            "system_answer":   result["answer"],
            "source_doc":      eq["source_doc"],
            "latency_ms":      latency_ms,
            "scores": {
                "heuristic": h,
                "llm_judge": None,
                "combined":  h,
                "llm_judge_available": False,
            },
        })

    aggregate = {}
    for m in METRICS:
        vals = [qr["scores"]["combined"][m] for qr in question_results]
        aggregate[m] = round(sum(vals) / len(vals), 3)

    return {
        "run_id":    run_id,
        "timestamp": timestamp,
        "aggregate": aggregate,
        "questions": question_results,
    }


def _flush_metrics(run: dict) -> None:
    os.makedirs("traces", exist_ok=True)
    with open(METRICS_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(run) + "\n")


def load_metrics_history() -> list[dict]:
    """Load all past eval runs from metrics.log. Returns [] if file absent."""
    if not os.path.exists(METRICS_LOG):
        return []
    runs = []
    with open(METRICS_LOG, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    runs.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return runs


def run_eval_and_store(collection, session_state) -> None:
    """
    Run eval synchronously and store result in session_state + metrics.log.
    Call this directly from app.py inside a st.spinner().
    """
    run = _run_eval(collection)
    _flush_metrics(run)
    if "eval_runs" not in session_state:
        session_state["eval_runs"] = []
    session_state["eval_runs"].append(run)
