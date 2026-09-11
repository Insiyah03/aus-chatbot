"""
test_retrieval.py

Runs a batch of representative catalog questions against the
Chroma collection built by build_index.py, so you can eyeball
whether the embeddings are actually retrieving the right chunks
before wiring up llama3.2 for answer generation.

Requires the same setup as build_index.py (Ollama running,
mxbai-embed-large pulled, chromadb + ollama installed) and that
you've already run build_index.py at least once.

Usage:
    python -m src.embeddings.test_retrieval
    python -m src.embeddings.test_retrieval --n 3
    python -m src.embeddings.test_retrieval --query "what is COE 486"
"""

from __future__ import annotations

import argparse

from src.embeddings.build_index import embed_query, get_collection, get_ollama_client

# A spread of question types covering the different chunk types
# in the catalog, so a weak spot in one category shows up clearly
# instead of being averaged out.
TEST_QUERIES = [
    "What are the prerequisites for MTH 221?",
    "What is COE 486?",
    "What courses are required for Computer Science?",
    "What scholarships are available?",
    "How do I contact the Academic Support Center?",
    "When does the Fall Semester 2025 add/drop period end?",
    "What is the tuition per credit hour?",
    "What is the proposed sequence of study for Architecture?",
    "Can I repeat a course I failed?",
    "What is the deadline to pay Fall Semester 2025 tuition fees?",
]


def run_one(collection, client, query: str, n_results: int) -> None:
    vector = embed_query(client, query)

    results = collection.query(
        query_embeddings=[vector],
        n_results=n_results,
    )

    print(f"\nQ: {query}")
    print("-" * 70)

    ids = results["ids"][0]
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    distances = results["distances"][0]

    if not ids:
        print("  (no results -- is the collection empty? run build_index.py first)")
        return

    for rank, (chunk_id, doc, meta, dist) in enumerate(
        zip(ids, docs, metas, distances), start=1
    ):
        label = meta.get("course_code") or meta.get("type") or ""
        page = meta.get("page")

        snippet = doc[:].replace("\n", " ").strip()
        print(f"  {rank}. [{label} | page {page} | dist {dist:.3f}] {snippet}...")


def main() -> None:
    parser = argparse.ArgumentParser(description="Test retrieval quality against real catalog questions.")
    parser.add_argument("--n", type=int, default=1, help="Results to show per query.")
    parser.add_argument("--query", type=str, default=None, help="Run a single custom query instead of the default batch.")
    args = parser.parse_args()

    client = get_ollama_client()
    collection = get_collection()

    count = collection.count()
    print(f"Collection has {count} chunks indexed.")

    if count == 0:
        print("Nothing indexed yet -- run build_index.py first.")
        return

    queries = [args.query] if args.query else TEST_QUERIES

    for query in queries:
        run_one(collection, client, query, args.n)

    print()


if __name__ == "__main__":
    main()