# Company Policy RAG Assistant

A fully local Retrieval-Augmented Generation (RAG) system for answering employee questions against company policy documents.

## Stack

| Layer | Tool |
|---|---|
| Embeddings | `sentence-transformers` — `all-MiniLM-L6-v2` (local, no API) |
| Vector store | ChromaDB (persistent, on-disk) |
| LLM | Ollama + `llama3` (local, no API) |
| UI | Streamlit |
| Chunking | Fixed-size ~256 tokens, 40-token overlap |
| Retrieval | Top-3 chunks by cosine similarity |

## Project Structure

```
rag_app/
├── docs/                    # Your policy markdown files
│   ├── expense_claims.md
│   ├── hr_leave_policy.md
│   ├── incident_escalation.md
│   └── it_support_policy.md
├── chroma_db/               # Auto-created by ingest.py
├── ingest.py                # One-time indexing script
├── rag.py                   # Core RAG logic
├── app.py                   # Streamlit web UI
├── requirements.txt
└── README.md
```

## Setup

### 1. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 2. Install and start Ollama

```bash
# Install Ollama from https://ollama.com
ollama pull llama3
ollama serve          # runs on http://localhost:11434
```

### 3. Index your documents

```bash
python ingest.py
```

This loads all `.md` files from `docs/`, chunks them, embeds them with `all-MiniLM-L6-v2`, and persists them to `chroma_db/`. Re-run whenever you update documents.

### 4. Launch the web UI

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

## Adding or Updating Documents

1. Drop new `.md` files into the `docs/` folder.
2. Re-run `python ingest.py` — it will rebuild the collection from scratch.
3. Restart the Streamlit app (or it will pick up changes on next cold start).

## How It Works

```
User question
     │
     ▼
Embed with all-MiniLM-L6-v2
     │
     ▼
Query ChromaDB → top-3 most similar chunks
     │
     ▼
Build prompt: system instructions + chunks + question
     │
     ▼
Send to Ollama (llama3) → generate answer
     │
     ▼
Display answer + source attribution in Streamlit
```

## Tuning Tips

- **Chunk size**: Increase `CHUNK_SIZE` in `ingest.py` for more context per chunk; decrease for more precise retrieval.
- **Top-k**: Change `TOP_K` in `rag.py` to retrieve more or fewer chunks.
- **Model**: Swap `OLLAMA_MODEL` in `rag.py` to any model you have pulled (e.g. `mistral`, `gemma3`).
- **Embedding model**: Swap `EMBED_MODEL` in both files — re-run `ingest.py` after changing.
