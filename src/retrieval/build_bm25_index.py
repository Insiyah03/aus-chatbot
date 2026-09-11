"""
build_bm25_index.py

Builds a BM25 keyword index over data/processed/chunks.json and
persists it to disk (like the Chroma vector index), so the
Streamlit app can load it instantly instead of re-tokenizing and
re-fitting BM25 statistics on every run.

BM25 is the complement to the vector search in build_index.py:
vector search is strong on semantic/paraphrased questions but weak
on exact identifiers (embedding models don't represent alphanumeric
codes like "COE 486" distinctively). BM25 is the opposite -- exact
term matches score highly, paraphrases don't. Hybrid retrieval
(next step) combines both.

Requires:
    pip install rank-bm25

Usage:
    python -m src.retrieval.build_bm25_index
    python -m src.retrieval.build_bm25_index --query "COE 486"
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import re
import time
from pathlib import Path
from src.config import (CHUNKS_PATH,INDEX_DIR)

# ============================================================
# PATHS / CONFIG
# ============================================================


CHUNKS_FILE = CHUNKS_PATH
INDEX_FILE = INDEX_DIR / "bm25.pkl"


# ============================================================
# TOKENIZATION
# ============================================================
#
# Course codes need special handling: "COE 486" should match a
# query for "COE486", "coe 486", or "COE-486" equally. We tokenize
# words/numbers normally AND additionally emit a fused token
# (e.g. "coe486") for every course-code-shaped span, so both a
# loose search ("COE") and an exact one ("COE 486") work.

WORD_RE = re.compile(r"[a-z]+|\d+")
COURSE_CODE_RE = re.compile(r"\b([a-z]{2,4})[\s-]?(\d{3}[a-z]{0,2})\b")


def tokenize(text: str) -> list[str]:
    text = (text or "").lower()

    tokens = WORD_RE.findall(text)
    tokens.extend(
        f"{prefix}{number}" for prefix, number in COURSE_CODE_RE.findall(text)
    )

    return tokens


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


def corpus_signature(chunk_ids: list[str]) -> str:
    """
    Fingerprint of exactly which chunk ids the index was built
    from, so a stale on-disk index (built before a chunks.json
    change) can be detected and flagged rather than silently used.
    """

    joined = "|".join(sorted(chunk_ids))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


# ============================================================
# BUILD / SAVE / LOAD
# ============================================================

def build_bm25_index(chunks: list[dict]):
    from rank_bm25 import BM25Okapi

    ids = [c["id"] for c in chunks]
    corpus = [tokenize(c["text"]) for c in chunks]

    bm25 = BM25Okapi(corpus)

    return bm25, ids


def save_bm25_index(bm25, ids: list[str]) -> None:
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    payload = {
        "bm25": bm25,
        "ids": ids,
        "signature": corpus_signature(ids),
    }

    with INDEX_FILE.open("wb") as f:
        pickle.dump(payload, f)


def load_bm25_index() -> tuple[object, list[str], str]:
    if not INDEX_FILE.exists():
        raise FileNotFoundError(
            f"No BM25 index found at {INDEX_FILE}\n"
            "Run `python -m src.retrieval.build_bm25_index` first."
        )

    with INDEX_FILE.open("rb") as f:
        payload = pickle.load(f)

    return payload["bm25"], payload["ids"], payload["signature"]


def check_index_freshness(ids: list[str], signature: str) -> bool:
    """Returns True if the on-disk index matches the current chunks.json."""

    try:
        chunks = load_chunks()
    except FileNotFoundError:
        return True  # can't check, assume fine

    current_ids = [c["id"] for c in chunks]
    return corpus_signature(current_ids) == signature


# ============================================================
# SEARCH
# ============================================================

def search(bm25, ids: list[str], query: str, top_k: int = 10) -> list[tuple[str, float]]:
    """Returns [(chunk_id, score), ...] sorted by score descending."""

    tokens = tokenize(query)
    scores = bm25.get_scores(tokens)

    ranked = sorted(zip(ids, scores), key=lambda pair: pair[1], reverse=True)

    return ranked[:top_k]


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Build a BM25 keyword index over catalog chunks.")
    parser.add_argument("--query", type=str, default=None, help="Run a sanity-check query after building.")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    print("=" * 70)
    print("BM25 INDEX BUILD")
    print("=" * 70)

    start = time.time()

    chunks = load_chunks()
    print(f"Loaded {len(chunks)} chunks from {CHUNKS_FILE}")

    bm25, ids = build_bm25_index(chunks)
    save_bm25_index(bm25, ids)

    elapsed = time.time() - start
    print(f"Built and saved BM25 index to {INDEX_FILE} in {elapsed:.2f}s")

    if args.query:
        chunks_by_id = {c["id"]: c for c in chunks}
        results = search(bm25, ids, args.query, top_k=args.top_k)

        print()
        print(f"Top {args.top_k} results for: {args.query!r}")
        print("-" * 70)

        for rank, (chunk_id, score) in enumerate(results, start=1):
            chunk = chunks_by_id.get(chunk_id, {})
            label = chunk.get("course_code") or chunk.get("type") or ""
            page = chunk.get("page")
            snippet = (chunk.get("text") or "")[:160].replace("\n", " ").strip()
            print(f"{rank}. [{label} | page {page} | score {score:.3f}] {snippet}...")


if __name__ == "__main__":
    main()