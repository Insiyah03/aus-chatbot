"""
build_index.py

Embeds every chunk in data/processed/chunks.json using a local
Ollama model (mxbai-embed-large by default) and stores the
resulting vectors in a persistent ChromaDB collection.

Requires:
    - Ollama running locally (`ollama serve`)
    - The embedding model pulled: `ollama pull mxbai-embed-large`
    - pip install chromadb ollama

Usage:
    python -m src.embeddings.build_index
    python -m src.embeddings.build_index --force        # re-embed everything
    python -m src.embeddings.build_index --limit 50      # smoke test on 50 chunks
    python -m src.embeddings.build_index --query "prerequisites for MTH 221"
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

# ============================================================
# PATHS / CONFIG
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DIR = BASE_DIR / "data" / "processed"
CHUNKS_FILE = PROCESSED_DIR / "chunks.json"

CHROMA_DIR = BASE_DIR / "data" / "chroma_db"
COLLECTION_NAME = "aus_catalog"

EMBED_MODEL = "mxbai-embed-large"
BATCH_SIZE = 32

# mxbai-embed-large is asymmetric: document text is embedded as-is,
# but at QUERY time (when you embed the user's question later, for
# retrieval) the model card recommends prefixing the query with:
#   "Represent this sentence for searching relevant passages: "
# This does not apply to indexing, only to embedding a live search
# query, which is why it's applied in embed_query() below and not
# in embed_batch().
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


# ============================================================
# LOADING
# ============================================================

def load_chunks() -> list[dict]:
    if not CHUNKS_FILE.exists():
        raise FileNotFoundError(
            f"Could not find {CHUNKS_FILE}\n"
            "Run preprocess.py first to generate chunks.json."
        )

    with CHUNKS_FILE.open(encoding="utf-8") as f:
        chunks = json.load(f)

    if not isinstance(chunks, list):
        raise ValueError("chunks.json must contain a list of chunks.")

    return chunks


# ============================================================
# METADATA FLATTENING
# ============================================================
#
# Chroma metadata values must be str, int, float, or bool -- no
# None, no nested dicts/lists. chunks.json has a nested "metadata"
# dict (course format, prerequisites list, etc.) that needs
# flattening before it can be stored.

def flatten_metadata(chunk: dict) -> dict[str, Any]:
    nested = chunk.get("metadata") or {}

    prerequisites = nested.get("prerequisites")
    prerequisites_str = ", ".join(prerequisites) if isinstance(prerequisites, list) else ""

    return {
        "document_id": chunk.get("document_id") or "",
        "type": chunk.get("type") or "",
        "section": chunk.get("section") or "",
        "page": chunk.get("page") if isinstance(chunk.get("page"), int) else 0,
        "course_code": chunk.get("course_code") or nested.get("course_code") or "",
        "title": chunk.get("title") or nested.get("title") or "",
        "format": nested.get("format") or "",
        "prerequisites": prerequisites_str,
    }


# ============================================================
# OLLAMA EMBEDDING
# ============================================================

def get_ollama_client():
    try:
        import ollama
    except ImportError as exc:
        raise ImportError(
            "The 'ollama' package is required. Install it with:\n"
            "    pip install ollama"
        ) from exc

    try:
        ollama.list()
    except Exception as exc:
        raise ConnectionError(
            "Could not reach a local Ollama server.\n"
            "Start it with `ollama serve`, and make sure the model is "
            f"pulled: `ollama pull {EMBED_MODEL}`"
        ) from exc

    return ollama


def embed_batch(client, texts: list[str]) -> list[list[float]]:
    response = client.embed(model=EMBED_MODEL, input=texts)
    embeddings = response.get("embeddings") if isinstance(response, dict) else response["embeddings"]

    if len(embeddings) != len(texts):
        raise RuntimeError(
            f"Ollama returned {len(embeddings)} embeddings for {len(texts)} inputs."
        )

    return embeddings


def embed_query(client, query: str) -> list[float]:
    """Embed a user question for retrieval (adds the query prefix)."""

    return embed_batch(client, [QUERY_PREFIX + query])[0]


# ============================================================
# CHROMA
# ============================================================

def get_collection():
    import chromadb

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    return chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


# ============================================================
# INDEXING
# ============================================================

def index_chunks(
    chunks: list[dict],
    collection,
    client,
    batch_size: int = BATCH_SIZE,
    force: bool = False,
) -> None:

    total = len(chunks)
    embedded_count = 0
    skipped_count = 0
    start = time.time()

    for batch_start in range(0, total, batch_size):

        batch = chunks[batch_start : batch_start + batch_size]
        batch_ids = [c["id"] for c in batch]

        if not force:
            existing = collection.get(ids=batch_ids)
            already_present = set(existing.get("ids") or [])
            batch = [c for c in batch if c["id"] not in already_present]
            skipped_count += len(batch_ids) - len(batch)

        if not batch:
            continue

        texts = [c["text"] for c in batch]
        ids = [c["id"] for c in batch]
        metadatas = [flatten_metadata(c) for c in batch]

        vectors = embed_batch(client, texts)

        collection.upsert(
            ids=ids,
            embeddings=vectors,
            documents=texts,
            metadatas=metadatas,
        )

        embedded_count += len(batch)

        done = min(batch_start + batch_size, total)
        elapsed = time.time() - start
        rate = embedded_count / elapsed if elapsed > 0 else 0

        print(
            f"  {done}/{total} chunks processed "
            f"({embedded_count} embedded, {skipped_count} already indexed, "
            f"{rate:.1f} chunks/sec)"
        )

    print()
    print(f"Done. Embedded {embedded_count} new chunks, skipped {skipped_count} already indexed.")
    print(f"Collection now holds {collection.count()} chunks total.")


# ============================================================
# QUICK MANUAL QUERY (sanity check)
# ============================================================

def run_query(collection, client, query: str, n_results: int = 5) -> None:
    vector = embed_query(client, query)

    results = collection.query(
        query_embeddings=[vector],
        n_results=n_results,
    )

    print()
    print(f"Top {n_results} results for: {query!r}")
    print("-" * 70)

    ids = results["ids"][0]
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    distances = results["distances"][0]

    for rank, (chunk_id, doc, meta, dist) in enumerate(
        zip(ids, docs, metas, distances), start=1
    ):
        print(f"{rank}. [{chunk_id}] (distance={dist:.4f}) page {meta.get('page')}")
        print(f"   {doc[:200].strip()}...")
        print()


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Embed catalog chunks into ChromaDB.")
    parser.add_argument("--force", action="store_true", help="Re-embed all chunks, even if already indexed.")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N chunks (for testing).")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--query", type=str, default=None, help="Run a sanity-check query after indexing.")
    args = parser.parse_args()

    print("=" * 70)
    print("EMBEDDING + CHROMADB INDEXING")
    print("=" * 70)

    chunks = load_chunks()
    if args.limit:
        chunks = chunks[: args.limit]

    print(f"Loaded {len(chunks)} chunks from {CHUNKS_FILE}")
    print(f"Model: {EMBED_MODEL}")
    print(f"Chroma path: {CHROMA_DIR}")
    print()

    client = get_ollama_client()
    collection = get_collection()

    index_chunks(chunks, collection, client, batch_size=args.batch_size, force=args.force)

    if args.query:
        run_query(collection, client, args.query)


if __name__ == "__main__":
    main()