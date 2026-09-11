"""
reranker.py

Second-stage reranking for the AUS Undergraduate Catalog.

Pipeline:
    Query
      -> ChromaDB + BM25
      -> Reciprocal Rank Fusion (hybrid_search.py)
      -> Cross-encoder reranker
      -> top N results

The reranker does NOT modify:
    - chunks.json
    - ChromaDB
    - BM25 index
    - hybrid_search.py

Model:
    BAAI/bge-reranker-base

The model is downloaded from Hugging Face the first time it is used,
then cached locally by sentence-transformers.

Requirements:
    pip install sentence-transformers

Usage:
    python -m src.retrieval.reranker --query "COE 486"
    python -m src.retrieval.reranker --query "prerequisites for MTH 221"
    python -m src.retrieval.reranker --query "Computer Science required courses"

Optional:
    --candidate-k 15   retrieve 15 hybrid candidates before reranking
    --top-k 5          return the best 5 after reranking
"""

from __future__ import annotations

import argparse
from typing import Any


# ============================================================
# CONFIG
# ============================================================

RERANK_MODEL = "BAAI/bge-reranker-base"

DEFAULT_CANDIDATE_K = 15
DEFAULT_TOP_K = 5


# ============================================================
# MODEL
# ============================================================

def get_reranker():
    """Load the local cross-encoder reranking model."""
    try:
        from sentence_transformers import CrossEncoder
    except ImportError as exc:
        raise ImportError(
            "The 'sentence-transformers' package is required.\n"
            "Install it with:\n"
            "    pip install sentence-transformers"
        ) from exc

    print(f"Loading reranker model: {RERANK_MODEL}")
    return CrossEncoder(RERANK_MODEL)


# ============================================================
# RERANKING
# ============================================================

def rerank(
    query: str,
    results: list[dict[str, Any]],
    model,
    top_k: int = DEFAULT_TOP_K,
) -> list[dict[str, Any]]:
    """
    Rerank hybrid-search results using a cross-encoder.

    A cross-encoder receives the query and each candidate passage
    together and produces a relevance score.

    Higher score = more relevant.
    """
    if not query.strip() or not results:
        return []

    pairs = [
        (query, (result.get("text") or "").strip())
        for result in results
    ]

    scores = model.predict(pairs)

    reranked = []

    for result, score in zip(results, scores):
        updated = {
            **result,
            "reranking": {
                "score": float(score),
            },
        }
        reranked.append(updated)

    reranked.sort(
        key=lambda result: result["reranking"]["score"],
        reverse=True,
    )

    return reranked[:top_k]


# ============================================================
# DISPLAY
# ============================================================

def print_results(query: str, results: list[dict[str, Any]]) -> None:
    """Print reranked results in a readable format."""
    print()
    print("=" * 90)
    print(f"RERANKED SEARCH: {query!r}")
    print("=" * 90)

    if not results:
        print("No results found.")
        return

    for rank, result in enumerate(results, start=1):
        retrieval = result.get("retrieval") or {}
        reranking = result.get("reranking") or {}
        metadata = result.get("metadata") or {}

        course_code = (
            result.get("course_code")
            or metadata.get("course_code")
            or "-"
        )

        title = (
            result.get("title")
            or metadata.get("title")
            or "-"
        )

        page = result.get("page") or metadata.get("page") or "-"

        print()
        print(
            f"{rank}. [{result.get('id')}] "
            f"Reranker={reranking.get('score', 0):.4f}"
        )

        print(
            f"   RRF={retrieval.get('rrf_score', 0):.6f} | "
            f"Vector rank={retrieval.get('vector_rank')} | "
            f"BM25 rank={retrieval.get('bm25_rank')}"
        )

        print(f"   Course: {course_code}")
        print(f"   Title: {title}")
        print(f"   Page: {page}")

        text = (result.get("text") or "").strip().replace("\n", " ")

        if len(text) > 450:
            text = text[:450] + "..."

        print(f"   {text}")


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Hybrid retrieval followed by cross-encoder reranking."
    )

    parser.add_argument(
        "--query",
        type=str,
        required=True,
        help="Search query.",
    )

    parser.add_argument(
        "--candidate-k",
        type=int,
        default=DEFAULT_CANDIDATE_K,
        help=(
            "Number of hybrid results to pass to the reranker "
            f"(default: {DEFAULT_CANDIDATE_K})."
        ),
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help=(
            "Number of results to return after reranking "
            f"(default: {DEFAULT_TOP_K})."
        ),
    )

    args = parser.parse_args()

    if args.candidate_k < 1:
        parser.error("--candidate-k must be at least 1")

    if args.top_k < 1:
        parser.error("--top-k must be at least 1")

    if args.top_k > args.candidate_k:
        parser.error("--top-k cannot be greater than --candidate-k")

    # Import here so the existing hybrid_search.py remains untouched.
    from .hybrid_search import hybrid_search

    print("=" * 90)
    print("HYBRID RETRIEVAL + RERANKING")
    print("=" * 90)
    print(f"Query: {args.query}")
    print(f"Hybrid candidates: {args.candidate_k}")
    print(f"Final results: {args.top_k}")

    print()
    print("Step 1/2: Running hybrid search...")

    candidates = hybrid_search(
        query=args.query,
        vector_k=args.candidate_k,
        bm25_k=args.candidate_k,
        top_k=args.candidate_k,
    )

    if not candidates:
        print("No hybrid results found.")
        return

    print(f"Retrieved {len(candidates)} hybrid candidates.")

    print()
    print("Step 2/2: Reranking candidates...")

    model = get_reranker()

    results = rerank(
        query=args.query,
        results=candidates,
        model=model,
        top_k=args.top_k,
    )

    print_results(args.query, results)


if __name__ == "__main__":
    main()
