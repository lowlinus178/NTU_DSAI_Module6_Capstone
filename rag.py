"""
rag.py — Core RAG logic: retrieve → rerank → generate.

Pipeline:
  1. ChromaDB retrieves TOP_K chunks by cosine similarity (sentence-transformers).
  2. Cohere rerank-v3 rescores those chunks against the query; order may change.
     If Cohere is unavailable, falls back to original ChromaDB order with a warning.
  3. Score-threshold check: if best rerank score < SCORE_THRESHOLD, the query is
     rewritten by Ollama and retrieval+rerank is retried (capped at MAX_REWRITES).
  4. Ollama llama3 generates the final answer from the top-ranked chunks.
"""

import os
from dotenv import load_dotenv
load_dotenv()
import time
import requests
import chromadb
from chromadb.utils import embedding_functions
from tracer import Tracer

CHROMA_DIR = "chroma_db"
EMBED_MODEL = "all-MiniLM-L6-v2"
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2"
TOP_K = 3
SCORE_THRESHOLD = 0.5      # minimum acceptable best rerank relevance score
MAX_REWRITES = 2           # max query-rewrite retries before giving up

COHERE_API_KEY = os.environ.get("COHERE_API_KEY", "")
COHERE_RERANK_URL = "https://api.cohere.com/v2/rerank"
COHERE_RERANK_MODEL = "rerank-v3.5"


def get_collection():
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBED_MODEL
    )
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    return client.get_collection(name="policy_docs", embedding_function=ef)


def retrieve(query: str, collection, top_k: int = TOP_K, tracer: Tracer = None, attempt: int = 0) -> list[dict]:
    """Return top-k most relevant chunks with metadata."""
    t0 = time.perf_counter()
    results = collection.query(
        query_texts=[query],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    chunks = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        chunks.append({
            "text": doc,
            "source": meta.get("source", "unknown"),
            "doc_name": meta.get("doc_name", "unknown"),
            "section": meta.get("section", ""),
            "score": round(1 - dist, 3),   # cosine similarity
        })

    if tracer:
        tracer.step(
            f"retrieve (attempt {attempt})",
            query=query,
            top_k=top_k,
            latency_ms=round((time.perf_counter() - t0) * 1000, 1),
            chunks=[
                {
                    "doc_name": c["doc_name"],
                    "section": c["section"],
                    "embed_score": c["score"],
                    "text": c["text"],
                }
                for c in chunks
            ],
        )
    return chunks


def rerank(query: str, chunks: list[dict], tracer: Tracer = None, attempt: int = 0) -> tuple[list[dict], bool, str | None]:
    """
    Rerank chunks with Cohere rerank-v3.5.

    Returns:
        (reranked_chunks, rerank_used, warning_message)
        - reranked_chunks : chunks in new order, each with a `rerank_score` key added
        - rerank_used     : False if Cohere was skipped/failed (fallback mode)
        - warning_message : human-readable reason when rerank_used is False, else None
    """
    t0 = time.perf_counter()

    if not COHERE_API_KEY:
        fallback = [dict(c, rerank_score=None, embed_score=c["score"]) for c in chunks]
        msg = "Cohere API key not set — skipped reranking (set COHERE_API_KEY env var)."
        if tracer:
            tracer.step(
                f"rerank (attempt {attempt})",
                rerank_used=False,
                warning=msg,
                latency_ms=round((time.perf_counter() - t0) * 1000, 1),
                order_unchanged=True,
            )
        return fallback, False, msg

    try:
        payload = {
            "model": COHERE_RERANK_MODEL,
            "query": query,
            "documents": [c["text"] for c in chunks],
            "top_n": len(chunks),
            "return_documents": False,
        }
        headers = {
            "Authorization": f"Bearer {COHERE_API_KEY}",
            "Content-Type": "application/json",
        }
        resp = requests.post(COHERE_RERANK_URL, json=payload, headers=headers, timeout=15)
        resp.raise_for_status()
        results = resp.json().get("results", [])

        reranked = []
        for r in results:
            chunk = dict(chunks[r["index"]])
            chunk["rerank_score"] = round(r["relevance_score"], 4)
            chunk["embed_score"] = chunk.pop("score")
            reranked.append(chunk)

        if tracer:
            original_order = [c["doc_name"] for c in chunks]
            new_order = [c["doc_name"] for c in reranked]
            tracer.step(
                f"rerank (attempt {attempt})",
                rerank_used=True,
                latency_ms=round((time.perf_counter() - t0) * 1000, 1),
                model=COHERE_RERANK_MODEL,
                order_changed=(original_order != new_order),
                original_order=original_order,
                new_order=new_order,
                scores=[
                    {
                        "doc_name": c["doc_name"],
                        "section": c["section"],
                        "rerank_score": c["rerank_score"],
                        "embed_score": c["embed_score"],
                    }
                    for c in reranked
                ],
            )
        return reranked, True, None

    except requests.exceptions.ConnectionError:
        msg = "Cohere API unreachable — falling back to ChromaDB order."
    except requests.exceptions.Timeout:
        msg = "Cohere API timed out — falling back to ChromaDB order."
    except requests.exceptions.HTTPError as e:
        msg = f"Cohere API error ({e.response.status_code}) — falling back to ChromaDB order."
    except Exception as e:
        msg = f"Reranking failed ({e}) — falling back to ChromaDB order."

    fallback = [dict(c, rerank_score=None, embed_score=c["score"]) for c in chunks]
    if tracer:
        tracer.step(
            f"rerank (attempt {attempt})",
            rerank_used=False,
            warning=msg,
            latency_ms=round((time.perf_counter() - t0) * 1000, 1),
            order_unchanged=True,
        )
    return fallback, False, msg


def rewrite_query(original_query: str, attempt: int, tracer: Tracer = None) -> str:
    """
    Ask the LLM to rephrase the query to improve retrieval.
    `attempt` (1-based) lets the prompt nudge toward broader rewrites on later tries.
    """
    nudge = (
        "Make it more specific and formal."
        if attempt == 1
        else "Try a broader, more general rephrasing using different keywords."
    )
    prompt = (
        f"You are a search query optimizer for an HR and IT policy knowledge base.\n"
        f"The query below failed to retrieve relevant results. Rewrite it to improve retrieval.\n"
        f"{nudge}\n"
        f"Return ONLY the rewritten query — no explanation, no quotes, no punctuation changes.\n\n"
        f"Original query: {original_query}\n"
        f"Rewritten query:"
    )
    t0 = time.perf_counter()
    rewritten = generate(prompt).strip().strip('"').strip("'")
    if not rewritten or len(rewritten) > 300:
        rewritten = original_query

    if tracer:
        tracer.step(
            f"rewrite (attempt {attempt})",
            original_query=original_query,
            rewritten_query=rewritten,
            nudge=nudge,
            rewrite_prompt=prompt,
            latency_ms=round((time.perf_counter() - t0) * 1000, 1),
        )
    return rewritten


def build_prompt(query: str, chunks: list[dict]) -> str:
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        context_parts.append(
            f"[Source {i}: {chunk['doc_name']} — {chunk['section']}]\n{chunk['text']}"
        )
    context = "\n\n---\n\n".join(context_parts)

    return f"""You are a helpful HR and IT policy assistant.
Answer the employee's question using ONLY the policy excerpts provided below.
If the answer is not in the excerpts, say so clearly — do not make things up.
Be concise and practical.

POLICY EXCERPTS:
{context}

EMPLOYEE QUESTION:
{query}

ANSWER:"""


def generate(prompt: str, model: str = OLLAMA_MODEL, tracer: Tracer = None) -> str:
    """Call Ollama and return the response text."""
    payload = {"model": model, "prompt": prompt, "stream": False}
    t0 = time.perf_counter()
    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=120)
        resp.raise_for_status()
        text = resp.json().get("response", "").strip()
        if tracer:
            tracer.step(
                "generate",
                model=model,
                prompt=prompt,
                response=text,
                latency_ms=round((time.perf_counter() - t0) * 1000, 1),
            )
        return text
    except requests.exceptions.ConnectionError:
        msg = (
            "⚠️ Could not connect to Ollama. "
            "Make sure Ollama is running (`ollama serve`) and the model is pulled "
            f"(`ollama pull {model}`)."
        )
        if tracer:
            tracer.step("generate", error=msg, latency_ms=round((time.perf_counter() - t0) * 1000, 1))
        return msg
    except Exception as e:
        msg = f"⚠️ Error calling Ollama: {e}"
        if tracer:
            tracer.step("generate", error=msg, latency_ms=round((time.perf_counter() - t0) * 1000, 1))
        return msg


def answer(query: str, collection, tracer: Tracer = None) -> dict:
    """
    Full RAG pipeline: retrieve → rerank → (score-check + optional rewrite) → generate.

      1. Retrieve TOP_K chunks from ChromaDB.
      2. Rerank with Cohere (fallback to original order if unavailable).
      3. Check best rerank score against SCORE_THRESHOLD.
         - Above threshold → generate answer.
         - Below threshold + rewrites remaining → rewrite query, go to 1.
         - Rewrites exhausted → generate with best chunks seen, set low_confidence=True.
    """
    if tracer:
        tracer.step("start", original_query=query, score_threshold=SCORE_THRESHOLD, max_rewrites=MAX_REWRITES)

    current_query = query
    best_chunks = None
    rewrite_log = []
    low_confidence = False
    rerank_used = False
    rerank_warning = None

    for attempt in range(MAX_REWRITES + 1):
        chunks = retrieve(current_query, collection, tracer=tracer, attempt=attempt)
        chunks, rerank_used, rerank_warning = rerank(current_query, chunks, tracer=tracer, attempt=attempt)

        best_score = (
            chunks[0]["rerank_score"]
            if chunks and chunks[0]["rerank_score"] is not None
            else (chunks[0]["embed_score"] if chunks else 0.0)
        )

        if best_chunks is None:
            best_chunks = chunks
        else:
            prev_best = (
                best_chunks[0]["rerank_score"]
                if best_chunks[0]["rerank_score"] is not None
                else best_chunks[0]["embed_score"]
            )
            if best_score > prev_best:
                best_chunks = chunks

        if best_score >= SCORE_THRESHOLD:
            if tracer:
                tracer.step("score_check", attempt=attempt, best_score=best_score,
                            threshold=SCORE_THRESHOLD, passed=True)
            break

        if tracer:
            tracer.step("score_check", attempt=attempt, best_score=best_score,
                        threshold=SCORE_THRESHOLD, passed=False)

        if attempt < MAX_REWRITES:
            rewritten = rewrite_query(query, attempt + 1, tracer=tracer)
            rewrite_log.append({
                "attempt": attempt + 1,
                "rewritten_query": rewritten,
                "score_before_rewrite": best_score,
            })
            current_query = rewritten
        else:
            low_confidence = True
            if tracer:
                tracer.step("low_confidence", reason="max rewrites exhausted", best_score=best_score)

    prompt = build_prompt(query, best_chunks)
    response = generate(prompt, tracer=tracer)

    if tracer:
        tracer.flush()

    return {
        "answer": response,
        "sources": best_chunks,
        "low_confidence": low_confidence,
        "rewrite_log": rewrite_log,
        "final_query": current_query,
        "rerank_used": rerank_used,
        "rerank_warning": rerank_warning,
        "trace": tracer.to_dict() if tracer else None,
    }
