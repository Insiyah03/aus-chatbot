"""
hybrid_search.py
Claude

Combines the BM25 keyword index (build_bm25_index.py) and the
Chroma vector index (build_index.py) via Reciprocal Rank Fusion
(RRF), so exact identifiers ("COE 486") and semantic/paraphrased
questions ("what courses teach machine learning") both retrieve
well.

Why RRF and not a weighted score blend: BM25 scores and cosine
distances live on completely different, unbounded scales -- there's
no principled way to add "distance 0.23" to "BM25 score 14.3"
without arbitrary normalization. RRF sidesteps this by only using
each result's RANK in its own list:

    rrf_score(doc) = sum over each list containing doc of  1 / (k + rank)

A doc that ranks well in either list (or both) scores highly,
regardless of the two systems' raw score scales. k (default 60) is
a standard damping constant from the original RRF paper -- it
flattens the difference between e.g. rank 1 and rank 2 so a single
list's top pick doesn't totally dominate.

Requires build_bm25_index.py and build_index.py to have already
been run (indexes must exist on disk).

Usage:
    python -m src.retrieval.hybrid_search --query "prerequisites for MTH 221"
    python -m src.retrieval.hybrid_search --compare   # runs the standard test batch, shows BM25-only / vector-only / hybrid side by side
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.embeddings.build_index import (
    embed_query,
    get_collection,
    get_ollama_client,
    load_chunks as load_chunks_embeddings,  # same file, kept as alias for clarity
)
from src.retrieval.build_bm25_index import (
    check_index_freshness,
    load_bm25_index,
    search as bm25_search,
)

RRF_K = 60
TOP_K_EACH = 20  # how deep each individual method searches before fusion
TOP_N = 8  # how many fused results to return


# ============================================================
# INDIVIDUAL RETRIEVERS
# ============================================================

def run_bm25(bm25, ids, query: str, top_k: int) -> list[tuple[str, float]]:
    """Returns [(chunk_id, bm25_score), ...] ranked best-first."""

    return bm25_search(bm25, ids, query, top_k=top_k)


def run_vector(collection, client, query: str, top_k: int) -> list[tuple[str, float]]:
    """Returns [(chunk_id, cosine_distance), ...] ranked best-first (lowest distance first)."""

    vector = embed_query(client, query)

    results = collection.query(
        query_embeddings=[vector],
        n_results=top_k,
    )

    ids = results["ids"][0]
    distances = results["distances"][0]

    return list(zip(ids, distances))


# ============================================================
# RECIPROCAL RANK FUSION
# ============================================================

def reciprocal_rank_fusion(
    ranked_lists: list[list[tuple[str, float]]],
    k: int = RRF_K,
) -> list[tuple[str, float, set[int]]]:
    """
    ranked_lists: one list per retrieval method, each already
    sorted best-first as [(chunk_id, raw_score), ...]. Raw scores
    are ignored -- only rank position matters for RRF.

    Returns [(chunk_id, rrf_score, {method_indices_that_found_it}), ...]
    sorted by rrf_score descending.
    """

    fused: dict[str, float] = {}
    sources: dict[str, set[int]] = {}

    for method_index, ranked in enumerate(ranked_lists):
        for rank, (chunk_id, _raw_score) in enumerate(ranked, start=1):
            fused[chunk_id] = fused.get(chunk_id, 0.0) + 1.0 / (k + rank)
            sources.setdefault(chunk_id, set()).add(method_index)

    combined = [
        (chunk_id, score, sources[chunk_id]) for chunk_id, score in fused.items()
    ]
    combined.sort(key=lambda item: item[1], reverse=True)

    return combined


# ============================================================
# HYBRID SEARCH
# ============================================================

def hybrid_search(
    query: str,
    bm25,
    bm25_ids,
    collection,
    client,
    top_k_each: int = TOP_K_EACH,
    top_n: int = TOP_N,
    rrf_k: int = RRF_K,
) -> list[tuple[str, float, set[int]]]:
    """
    Returns [(chunk_id, rrf_score, source_methods), ...], top_n
    results, best first. source_methods is a set of {0, 1} where
    0 = found by BM25, 1 = found by vector search (useful for
    understanding *why* a result showed up).
    """

    bm25_results = run_bm25(bm25, bm25_ids, query, top_k_each)
    vector_results = run_vector(collection, client, query, top_k_each)

    fused = reciprocal_rank_fusion([bm25_results, vector_results], k=rrf_k)

    return fused[:top_n]


# ============================================================
# DISPLAY HELPERS
# ============================================================

def method_label(sources: set[int]) -> str:
    if sources == {0, 1}:
        return "both"
    if sources == {0}:
        return "bm25"
    if sources == {1}:
        return "vector"
    return "?"


def print_results(
    title: str,
    results: list[tuple[str, float, set[int]]],
    chunks_by_id: dict[str, dict],
    score_label: str = "rrf",
) -> None:
    print(f"\n{title}")
    print("-" * 70)

    if not results:
        print("  (no results)")
        return

    for rank, (chunk_id, score, sources) in enumerate(results, start=1):
        chunk = chunks_by_id.get(chunk_id, {})
        label = chunk.get("course_code") or chunk.get("type") or ""
        page = chunk.get("page")
        source = method_label(sources)
        snippet = (chunk.get("text") or "")[:150].replace("\n", " ").strip()
        print(f"  {rank}. [{label} | page {page} | {score_label} {score:.4f} | via {source}] {snippet}...")


# ============================================================
# COMPARISON MODE (bm25-only vs vector-only vs hybrid, side by side)
# ============================================================

COMPARE_QUERIES = [
    "What are the prerequisites for MTH 221?",
    "What is COE 486?",
    "What courses teach machine learning?",
    "What scholarships are available?",
    "Academic Support Center phone number",
]


def run_comparison(bm25, bm25_ids, collection, client, chunks_by_id, top_n: int) -> None:
    for query in COMPARE_QUERIES:
        print("\n" + "=" * 70)
        print(f"QUERY: {query}")
        print("=" * 70)

        bm25_only = [(cid, score, {0}) for cid, score in run_bm25(bm25, bm25_ids, query, top_n)]
        vector_only = [(cid, score, {1}) for cid, score in run_vector(collection, client, query, top_n)]
        fused = hybrid_search(query, bm25, bm25_ids, collection, client, top_n=top_n)

        print_results("BM25 only:", bm25_only, chunks_by_id, score_label="bm25 score")
        print_results("Vector only:", vector_only, chunks_by_id, score_label="distance")
        print_results("Hybrid (RRF):", fused, chunks_by_id, score_label="rrf")


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid (BM25 + vector) retrieval over catalog chunks.")
    parser.add_argument("--query", type=str, default=None)
    parser.add_argument("--top-n", type=int, default=TOP_N)
    parser.add_argument("--top-k-each", type=int, default=TOP_K_EACH)
    parser.add_argument("--rrf-k", type=int, default=RRF_K)
    parser.add_argument("--compare", action="store_true", help="Run the standard comparison batch (bm25 vs vector vs hybrid).")
    args = parser.parse_args()

    print("Loading indexes...")

    bm25, bm25_ids, signature = load_bm25_index()
    if not check_index_freshness(bm25_ids, signature):
        print(
            "WARNING: the BM25 index on disk doesn't match the current "
            "chunks.json. Re-run build_bm25_index.py to refresh it."
        )

    collection = get_collection()
    if collection.count() == 0:
        print("WARNING: the Chroma collection is empty. Run build_index.py first.")

    client = get_ollama_client()

    chunks = load_chunks_embeddings()
    chunks_by_id = {c["id"]: c for c in chunks}

    if args.compare:
        run_comparison(bm25, bm25_ids, collection, client, chunks_by_id, args.top_n)
        return

    if not args.query:
        print("Pass --query \"...\" or --compare to run the standard test batch.")
        return

    fused = hybrid_search(
        args.query,
        bm25,
        bm25_ids,
        collection,
        client,
        top_k_each=args.top_k_each,
        top_n=args.top_n,
        rrf_k=args.rrf_k,
    )
    print_results(f"Hybrid results for: {args.query!r}", fused, chunks_by_id)


if __name__ == "__main__":
    main()