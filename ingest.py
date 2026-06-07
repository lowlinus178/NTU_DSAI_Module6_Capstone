"""
ingest.py — Run once (or whenever docs change) to build the ChromaDB vector store.
Usage: python ingest.py
"""

import os
import re
import chromadb
from chromadb.utils import embedding_functions

DOCS_DIR = "docs"
CHROMA_DIR = "chroma_db"
CHUNK_SIZE = 256       # tokens (approx characters / 4)
CHUNK_OVERLAP = 40     # token overlap between chunks
EMBED_MODEL = "all-MiniLM-L6-v2"


def load_markdown_files(docs_dir: str) -> list[dict]:
    """Load all .md files, returning list of {filename, content}."""
    docs = []
    for fname in sorted(os.listdir(docs_dir)):
        if fname.endswith(".md"):
            path = os.path.join(docs_dir, fname)
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            docs.append({"filename": fname, "content": content})
            print(f"  Loaded: {fname} ({len(content)} chars)")
    return docs


def extract_section(text: str, char_pos: int) -> str:
    """Find the nearest markdown heading above a character position."""
    segment = text[:char_pos]
    headings = re.findall(r"^#{1,3} .+", segment, re.MULTILINE)
    if headings:
        return headings[-1].lstrip("#").strip()
    return "Introduction"


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[dict]:
    """
    Split text into fixed-size chunks (measured in approximate tokens = chars/4)
    with overlap. Returns list of {text, char_start}.
    """
    char_chunk = chunk_size * 4
    char_overlap = overlap * 4
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + char_chunk, len(text))
        chunk_text = text[start:end].strip()
        if chunk_text:
            chunks.append({"text": chunk_text, "char_start": start})
        if end == len(text):
            break
        start += char_chunk - char_overlap
    return chunks


def build_vector_store():
    print("\n── Loading documents ──")
    docs = load_markdown_files(DOCS_DIR)

    print("\n── Chunking documents ──")
    all_chunks = []
    all_ids = []
    all_metadata = []

    for doc in docs:
        doc_name = doc["filename"].replace(".md", "")
        chunks = chunk_text(doc["content"], CHUNK_SIZE, CHUNK_OVERLAP)
        for i, chunk in enumerate(chunks):
            section = extract_section(doc["content"], chunk["char_start"])
            chunk_id = f"{doc_name}_chunk_{i}"
            all_chunks.append(chunk["text"])
            all_ids.append(chunk_id)
            all_metadata.append({
                "source": doc["filename"],
                "doc_name": doc_name,
                "section": section,
                "chunk_index": i,
            })
        print(f"  {doc['filename']}: {len(chunks)} chunks")

    print("\n── Building ChromaDB collection ──")
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBED_MODEL
    )
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    # Drop and recreate to allow re-ingestion
    try:
        client.delete_collection("policy_docs")
    except Exception:
        pass

    collection = client.create_collection(
        name="policy_docs",
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )

    # Upsert in batches of 50
    batch_size = 50
    for i in range(0, len(all_chunks), batch_size):
        collection.add(
            documents=all_chunks[i : i + batch_size],
            ids=all_ids[i : i + batch_size],
            metadatas=all_metadata[i : i + batch_size],
        )

    print(f"\n✅ Done! {len(all_chunks)} chunks stored in '{CHROMA_DIR}/'")
    print(f"   Embedding model : {EMBED_MODEL}")
    print(f"   Collection name : policy_docs")


if __name__ == "__main__":
    build_vector_store()
