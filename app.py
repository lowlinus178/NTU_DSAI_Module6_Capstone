"""
app.py — Streamlit web UI for the company policy RAG assistant.
Usage: streamlit run app.py
"""

import json
import streamlit as st
import pandas as pd
from rag import answer, get_collection
from tracer import Tracer, TRACE_FILE
from eval_runner import run_eval_and_store, load_metrics_history, METRICS_LOG

METRICS = ("relevance", "accuracy", "faithfulness", "completeness")
METRIC_COLOURS = {
    "relevance":    "#4C9BE8",
    "accuracy":     "#56C596",
    "faithfulness": "#F5A623",
    "completeness": "#E86C6C",
}

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Policy Assistant", page_icon="📋", layout="wide")

# ── Load collection ───────────────────────────────────────────────────────────
@st.cache_resource
def load_collection():
    try:
        return get_collection()
    except Exception:
        return None

collection = load_collection()
if collection is None:
    st.error("⚠️ Vector store not found. Please run `python ingest.py` first.")
    st.stop()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
        **Policy documents indexed:**
        - `expense_claims.md`
        - `hr_leave_policy.md`
        - `incident_escalation.md`
        - `it_support_policy.md`

        **Stack:**
        - Embeddings: `all-MiniLM-L6-v2`
        - Vector store: ChromaDB
        - Reranker: Cohere `rerank-v3.5`
        - LLM: Ollama `llama3.2`
        - Retrieval: top-3 chunks
    """)
    st.divider()
    show_chunks = st.toggle("Show retrieved chunks / full prompts", value=False)

# ── Session state defaults ────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []
if "eval_runs" not in st.session_state:
    st.session_state.eval_runs = load_metrics_history()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_chat, tab_metrics = st.tabs(["💬 Chat", "📊 Metrics"])

# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — CHAT
# ═════════════════════════════════════════════════════════════════════════════
with tab_chat:
    st.title("📋 Company Policy Assistant")
    st.caption("Ask anything about HR, IT, Expenses, or Incident policies.")

    # Example questions
    st.markdown("**Try asking:**")
    example_cols = st.columns(2)
    examples = [
        "How many sick days do I get?",
        "What's the meal allowance when travelling?",
        "How do I escalate a SEV1 incident?",
        "How do I request new software?",
    ]
    for i, ex in enumerate(examples):
        if example_cols[i % 2].button(ex, use_container_width=True):
            st.session_state["prefill"] = ex

    # Query input
    prefill = st.session_state.pop("prefill", "")
    query = st.chat_input("Ask a policy question…")
    active_query = query or (prefill if prefill else None)

    if active_query:
        st.session_state.history.append({"role": "user", "content": active_query})

        with st.spinner("Searching policies and generating answer…"):
            tracer = Tracer(query=active_query)
            result = answer(active_query, collection, tracer=tracer)

        st.session_state.history.append({
            "role":          "assistant",
            "content":       result["answer"],
            "sources":       result["sources"],
            "low_confidence": result["low_confidence"],
            "rewrite_log":   result["rewrite_log"],
            "rerank_used":   result["rerank_used"],
            "rerank_warning": result["rerank_warning"],
            "trace":         result["trace"],
        })

        # Run eval suite synchronously after every Q&A
        with st.spinner("Running eval suite against 10 test questions (heuristic scoring)…"):
            run_eval_and_store(collection, st.session_state)

    # Render conversation
    for msg in st.session_state.history:
        if msg["role"] == "user":
            with st.chat_message("user"):
                st.write(msg["content"])
        else:
            with st.chat_message("assistant"):
                if msg.get("rerank_warning"):
                    st.warning(f"⚠️ **Reranker unavailable** — {msg['rerank_warning']}", icon="⚠️")
                if msg.get("low_confidence"):
                    st.warning(
                        "⚠️ **Low confidence** — best chunk scored below 50% even after rewrites.",
                        icon="⚠️",
                    )

                st.write(msg["content"])

                # ── Trace panel ──────────────────────────────────────────
                trace = msg.get("trace")
                if trace:
                    with st.expander(
                        f"🔍 Trace `{trace['trace_id']}` · `{trace['total_ms']} ms total`",
                        expanded=False,
                    ):
                        for step in trace["steps"]:
                            sname = step["step"]
                            st.markdown(f"**`{sname}`** &nbsp; +{step['elapsed_ms']} ms &nbsp; _(@ {step['total_ms']} ms)_")
                            skip = {"step", "elapsed_ms", "total_ms"}
                            payload = {k: v for k, v in step.items() if k not in skip}

                            if sname.startswith("retrieve"):
                                chunks_data = payload.pop("chunks", [])
                                for k, v in payload.items():
                                    st.caption(f"{k}: `{v}`")
                                for i, c in enumerate(chunks_data, 1):
                                    st.markdown(f"&nbsp;&nbsp;Chunk {i} · **{c['doc_name']}** — *{c['section']}* · embed `{c['embed_score']}`")
                                    if show_chunks:
                                        st.code(c["text"], language=None)
                            elif sname.startswith("rerank"):
                                scores = payload.pop("scores", [])
                                for k, v in payload.items():
                                    st.caption(f"{k}: `{v}`")
                                for i, s in enumerate(scores, 1):
                                    st.markdown(f"&nbsp;&nbsp;{i}. **{s['doc_name']}** — *{s['section']}* · rerank `{s['rerank_score']}` · embed `{s['embed_score']}`")
                            elif sname.startswith("rewrite"):
                                st.caption(f"original: `{payload.get('original_query')}`")
                                st.caption(f"rewritten: `{payload.get('rewritten_query')}`")
                                if show_chunks:
                                    st.code(payload.get("rewrite_prompt", ""), language=None)
                            elif sname == "generate":
                                if show_chunks:
                                    st.text("── prompt ──")
                                    st.code(payload.get("prompt", ""), language=None)
                                    st.text("── response ──")
                                    st.code(payload.get("response", payload.get("error", "")), language=None)
                                else:
                                    st.caption(f"model: `{payload.get('model')}` · latency: `{payload.get('latency_ms')} ms`")
                            else:
                                for k, v in payload.items():
                                    st.caption(f"{k}: `{v}`")
                            st.divider()
                        st.caption(f"📁 Full trace written to `{TRACE_FILE}`")

                # ── Rewrite log ──────────────────────────────────────────
                rewrite_log = msg.get("rewrite_log", [])
                if rewrite_log:
                    with st.expander("🔄 Query rewrites attempted", expanded=False):
                        for entry in rewrite_log:
                            st.markdown(f"**Attempt {entry['attempt']}** (score before: `{int(entry['score_before_rewrite']*100)}%`)")
                            st.code(entry["rewritten_query"], language=None)

                # ── Sources ───────────────────────────────────────────────
                sources = msg.get("sources", [])
                rerank_used = msg.get("rerank_used", False)
                if sources:
                    with st.expander("📎 Sources used", expanded=False):
                        for i, src in enumerate(sources, 1):
                            if rerank_used and src.get("rerank_score") is not None:
                                badge = f"`rerank: {src['rerank_score']:.2f}` · `embed: {int(src['embed_score']*100)}%`"
                            else:
                                badge = f"`embed: {int(src.get('embed_score', src.get('score', 0))*100)}%`"
                            st.markdown(f"**{i}. {src['doc_name']}** — *{src['section']}* &nbsp; {badge}")
                            if show_chunks:
                                st.caption(src["text"])
                            st.divider()

# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — METRICS
# ═════════════════════════════════════════════════════════════════════════════
with tab_metrics:
    st.title("📊 RAG Quality Metrics")
    st.caption(
        "Evaluated after every Q&A against 10 ground-truth questions. "
        "Each metric is scored 0–1 (heuristic + LLM-as-judge averaged)."
    )

    runs = st.session_state.get("eval_runs", [])

    if not runs:
        st.info("No eval runs yet. Ask a question in the Chat tab to trigger the first run.")
        st.stop()

    latest = runs[-1]

    # ── Latest run: metric gauge cards ───────────────────────────────────────
    st.subheader(f"Latest run · `{latest['run_id']}` · {latest['timestamp'][:19].replace('T',' ')} UTC")

    agg = latest["aggregate"]
    cols = st.columns(4)
    for col, m in zip(cols, METRICS):
        score = agg[m]
        pct = int(score * 100)
        colour = METRIC_COLOURS[m]
        # Colour-coded delta vs previous run
        delta_str = ""
        if len(runs) >= 2:
            prev = runs[-2]["aggregate"][m]
            delta = score - prev
            sign = "▲" if delta >= 0 else "▼"
            delta_str = f"{sign} {abs(delta):.3f} vs prev"
        col.metric(
            label=m.capitalize(),
            value=f"{pct}%",
            delta=delta_str if delta_str else None,
        )

    # ── Trend chart ───────────────────────────────────────────────────────────
    st.divider()
    st.subheader("Trend across all runs")

    chart_data = []
    for run in runs:
        row = {"run": run["run_id"], "timestamp": run["timestamp"][:19].replace("T", " ")}
        for m in METRICS:
            row[m] = run["aggregate"][m]
        chart_data.append(row)

    df = pd.DataFrame(chart_data)
    df_melted = df.melt(id_vars=["run", "timestamp"], var_name="metric", value_name="score")
    df_melted["label"] = df_melted["timestamp"] + " (" + df_melted["run"] + ")"

    st.line_chart(
        df.set_index("timestamp")[list(METRICS)],
        height=320,
        use_container_width=True,
    )

    # ── Per-question breakdown for latest run ─────────────────────────────────
    st.divider()
    st.subheader("Per-question breakdown (latest run)")

    for qr in latest["questions"]:
        scores = qr["scores"]["combined"]
        worst = min(scores, key=scores.get)
        with st.expander(
            f"**{qr['id']}** · {qr['question']} &nbsp; "
            f"avg `{round(sum(scores.values())/4, 2)}`",
            expanded=False,
        ):
            q_cols = st.columns(4)
            for qcol, m in zip(q_cols, METRICS):
                qcol.metric(m.capitalize(), f"{int(scores[m]*100)}%")

            st.markdown("**Question:** " + qr["question"])
            st.markdown("**Expected:** " + qr["expected_answer"])
            st.markdown("**Got:** " + qr["system_answer"])

            h = qr["scores"]["heuristic"]
            l = qr["scores"]["llm_judge"]
            st.caption(
                f"Heuristic — relevance: {h['relevance']} · accuracy: {h['accuracy']} · "
                f"faithfulness: {h['faithfulness']} · completeness: {h['completeness']}"
            )
            if l:
                st.caption(
                    f"LLM judge — relevance: {l['relevance']} · accuracy: {l['accuracy']} · "
                    f"faithfulness: {l['faithfulness']} · completeness: {l['completeness']}"
                )
            else:
                st.caption("LLM judge: unavailable (heuristic only)")
            st.caption(f"Latency: {qr['latency_ms']} ms · Source doc: `{qr['source_doc']}`")

    # ── Raw log link ──────────────────────────────────────────────────────────
    st.divider()
    st.caption(f"📁 Full metrics log written to `{METRICS_LOG}`")
    if st.button("Show raw latest run JSON"):
        st.json(latest)
