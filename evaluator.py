"""
evaluator.py — Score a RAG answer on 4 quality metrics.

Metrics (each scored 0.0–1.0):
  relevance    : Does the answer address the question? (heuristic: keyword overlap in question
                 vs answer; LLM: does it actually answer what was asked?)
  accuracy     : Does the answer match the ground-truth? (heuristic: expected keyword hits;
                 LLM: factual correctness vs reference answer)
  faithfulness : Is the answer grounded in the retrieved sources? (heuristic: source_doc name
                 present in chunk metadata; LLM: no hallucination beyond provided context)
  completeness : Does the answer cover all key points? (heuristic: fraction of keywords hit;
                 LLM: are all important aspects of the reference answer addressed?)

Two-stage scoring:
  1. Heuristic pass  — instant, deterministic, runs synchronously
  2. LLM-judge pass  — sends a single structured prompt to Ollama, returns refined scores

Final score = average(heuristic, llm) per metric.
"""

import json
import re
import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2"


# ── Heuristic scoring ─────────────────────────────────────────────────────────

def _keyword_hit_rate(keywords: list[str], text: str) -> float:
    """Fraction of keywords present in text (case-insensitive)."""
    if not keywords:
        return 0.0
    text_lower = text.lower()
    hits = sum(1 for kw in keywords if kw.lower() in text_lower)
    return round(hits / len(keywords), 3)


def heuristic_scores(question: str, answer: str, expected_answer: str,
                     keywords: list[str], source_doc: str,
                     retrieved_chunks: list[dict]) -> dict:
    """Fast deterministic scores — no LLM call required."""

    # Relevance: do question terms appear in the answer?
    q_words = [w for w in re.findall(r"\w+", question.lower()) if len(w) > 3]
    relevance = _keyword_hit_rate(q_words, answer)

    # Accuracy: do expected keywords appear in the answer?
    accuracy = _keyword_hit_rate(keywords, answer)

    # Faithfulness: is the correct source doc among retrieved chunks?
    retrieved_docs = [c.get("doc_name", "") for c in retrieved_chunks]
    faithfulness = 1.0 if any(source_doc in d for d in retrieved_docs) else 0.0

    # Completeness: fraction of expected answer keywords covered
    expected_words = [w for w in re.findall(r"\w+", expected_answer.lower()) if len(w) > 3]
    completeness = _keyword_hit_rate(expected_words, answer)

    return {
        "relevance":    round(min(relevance, 1.0), 3),
        "accuracy":     round(min(accuracy, 1.0), 3),
        "faithfulness": faithfulness,
        "completeness": round(min(completeness, 1.0), 3),
    }


# ── LLM-as-judge scoring ──────────────────────────────────────────────────────

JUDGE_PROMPT_TEMPLATE = """You are an objective RAG evaluation judge. Score the SYSTEM ANSWER on 4 metrics.

QUESTION: {question}
REFERENCE ANSWER: {expected_answer}
SYSTEM ANSWER: {answer}
RETRIEVED CONTEXT (sources used): {context_summary}

Score each metric from 0.0 to 1.0:
- relevance: Does the system answer address the question that was asked?
- accuracy: Is the system answer factually correct compared to the reference answer?
- faithfulness: Is the system answer grounded only in the retrieved context (no hallucination)?
- completeness: Does the system answer cover all key points from the reference answer?

Respond ONLY with valid JSON, no explanation, no markdown:
{{"relevance": 0.0, "accuracy": 0.0, "faithfulness": 0.0, "completeness": 0.0}}"""


def llm_scores(question: str, answer: str, expected_answer: str,
               retrieved_chunks: list[dict]) -> dict | None:
    """
    Ask Ollama to score the answer. Returns dict of scores or None on failure.
    Failure is non-fatal — caller falls back to heuristic scores only.
    """
    context_summary = "; ".join(
        f"{c.get('doc_name','?')} § {c.get('section','?')}"
        for c in retrieved_chunks
    )
    prompt = JUDGE_PROMPT_TEMPLATE.format(
        question=question,
        expected_answer=expected_answer,
        answer=answer,
        context_summary=context_summary,
    )
    try:
        resp = requests.post(
            OLLAMA_URL,
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
            timeout=60,
        )
        resp.raise_for_status()
        raw = resp.json().get("response", "").strip()
        # Strip markdown fences if present
        raw = re.sub(r"```[a-z]*", "", raw).replace("```", "").strip()
        scores = json.loads(raw)
        return {
            "relevance":    round(float(scores.get("relevance", 0)), 3),
            "accuracy":     round(float(scores.get("accuracy", 0)), 3),
            "faithfulness": round(float(scores.get("faithfulness", 0)), 3),
            "completeness": round(float(scores.get("completeness", 0)), 3),
        }
    except Exception:
        return None


# ── Combined scorer ───────────────────────────────────────────────────────────

def score_answer(question: str, answer: str, expected_answer: str,
                 keywords: list[str], source_doc: str,
                 retrieved_chunks: list[dict]) -> dict:
    """
    Run heuristic + LLM scoring and return combined result.
    Final score per metric = mean(heuristic, llm) when both available,
    else heuristic only.
    """
    h = heuristic_scores(question, answer, expected_answer, keywords, source_doc, retrieved_chunks)
    l = llm_scores(question, answer, expected_answer, retrieved_chunks)

    if l:
        combined = {
            m: round((h[m] + l[m]) / 2, 3)
            for m in ("relevance", "accuracy", "faithfulness", "completeness")
        }
    else:
        combined = h

    return {
        "heuristic": h,
        "llm_judge": l,
        "combined":  combined,
        "llm_judge_available": l is not None,
    }
