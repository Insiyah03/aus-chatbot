"""
answer.py

Final RAG step: takes a user question, retrieves context via
hybrid_search.py, and generates an answer with llama3.2 -- with
page citations that are guaranteed accurate regardless of whether
the model cites correctly inline, because the Sources list is
built deterministically from the retrieved chunks' own metadata,
not parsed out of the model's text.

Requires build_index.py and build_bm25_index.py to have already
been run (indexes must exist on disk), and `ollama pull llama3.2`.

Usage:
    python -m src.generation.answer --query "What are the prerequisites for MTH 221?"
    python -m src.generation.answer            # interactive REPL
"""

from __future__ import annotations

import argparse
import time

from src.embeddings.build_index import get_collection, get_ollama_client, load_chunks
from src.retrieval.build_bm25_index import check_index_freshness, load_bm25_index
from src.retrieval.hybrid_search import hybrid_search

ANSWER_MODEL = "llama3.2:1b"

# Explicit, not left to Ollama's default -- see module docstring.
# Sized to actual observed usage: ~6-8 retrieved chunks plus the
# system prompt typically runs 1000-2000 tokens. 4096 leaves solid
# headroom for longer questions/more chunks without over-allocating
# KV cache -- an oversized num_ctx (this was previously 8192) can
# force a slower fallback path on memory-constrained GPUs even
# though the actual prompt never gets close to it. If you have
# hefty VRAM and want to push --top-n higher for harder questions,
# raise this back up and re-check --timing.
NUM_CTX = 4096

# Ollama unloads a model from memory ~5 minutes after its last use
# by default. The next request then pays a full model-load cost on
# top of generation -- often the real culprit behind a "slow"
# first (or first-after-a-pause) answer. Keeping it loaded longer
# avoids that reload tax across a chat session.
KEEP_ALIVE = "30m"

DEFAULT_TOP_N = 8
MAX_CONTEXT_CHARS = 9000  # soft cap so we don't overrun NUM_CTX even on long chunks

SYSTEM_PROMPT = """You are a helpful assistant answering questions about the American \
University of Sharjah (AUS) Undergraduate Catalog 2025-2026.

Answer ONLY using the catalog excerpts provided below. Each excerpt is labeled with \
its page number in the format [p. N].

Rules:
- If the excerpts don't contain the answer, say so plainly -- do not guess or use \
outside knowledge.
- When you state a specific fact (a prerequisite, credit hours, a requirement, a \
deadline, a fee), mention the page it came from like this: [p. N].
- Be concise and factual. Don't pad the answer with generic filler.
- If multiple excerpts are relevant, synthesize them into one coherent answer rather \
than listing them separately.
"""


# ============================================================
# CONTEXT BUILDING
# ============================================================

def build_context(chunks: list[dict]) -> tuple[str, list[dict]]:
    """
    Assembles the context block sent to the model, and the
    deterministic source list used for the citation footer
    (independent of whether the model cites correctly itself).

    Stops adding chunks once MAX_CONTEXT_CHARS is reached, so a
    long tail of low-relevance chunks doesn't silently blow the
    context budget -- better to answer well from fewer, more
    relevant excerpts than truncate mid-chunk.
    """

    parts = []
    sources = []
    total_chars = 0

    for chunk in chunks:
        page = chunk.get("page")
        text = chunk.get("text") or ""
        block = f"[p. {page}] {text}"

        if total_chars + len(block) > MAX_CONTEXT_CHARS and parts:
            break

        parts.append(block)
        total_chars += len(block)

        sources.append(
            {
                "id": chunk.get("id"),
                "page": page,
                "course_code": chunk.get("course_code") or "",
                "type": chunk.get("type") or "",
                "title": chunk.get("title") or "",
            }
        )

    context = "\n\n---\n\n".join(parts)

    return context, sources


def format_sources(sources: list[dict]) -> str:
    """
    Deterministic 'Sources' footer, built from retrieved chunk
    metadata -- not from anything the model generated. This is
    what guarantees the citations are accurate even if a small
    local model forgets to cite inline or cites wrong.
    """

    if not sources:
        return ""

    seen = set()
    lines = []

    for source in sorted(sources, key=lambda s: (s["page"] or 0)):
        label = source["course_code"] or source["title"] or source["type"] or "excerpt"
        key = (label, source["page"])

        if key in seen:
            continue
        seen.add(key)

        lines.append(f"- {label} (p. {source['page']})")

    return "Sources:\n" + "\n".join(lines)


# ============================================================
# GENERATION
# ============================================================

def build_messages(query: str, context: str) -> list[dict]:
    user_content = f"Catalog excerpts:\n\n{context}\n\n---\n\nQuestion: {query}"

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def generate_answer(
    query: str,
    bm25,
    bm25_ids,
    collection,
    client,
    top_n: int = DEFAULT_TOP_N,
) -> dict:
    """
    Returns {"answer": str, "sources": [...], "chunks_used": [...]}.
    """

    chunks_by_id = {c["id"]: c for c in load_chunks()}

    fused = hybrid_search(query, bm25, bm25_ids, collection, client, top_n=top_n)
    retrieved_chunks = [chunks_by_id[chunk_id] for chunk_id, _score, _src in fused if chunk_id in chunks_by_id]

    if not retrieved_chunks:
        no_result_text = "I couldn't find anything relevant to that question in the catalog."
        return {
            "answer": no_result_text,
            "answer_text": no_result_text,
            "sources": [],
            "chunks_used": [],
        }

    context, sources = build_context(retrieved_chunks)
    messages = build_messages(query, context)

    response = client.chat(
        model=ANSWER_MODEL,
        messages=messages,
        options={"num_ctx": NUM_CTX, "temperature": 0.1},
        keep_alive=KEEP_ALIVE,
    )

    answer_text = response["message"]["content"]
    sources_footer = format_sources(sources)

    full_answer = f"{answer_text}\n\n{sources_footer}" if sources_footer else answer_text

    return {
        "answer": full_answer,
        "answer_text": answer_text,
        "sources": sources,
        "chunks_used": [c["id"] for c in retrieved_chunks],
    }


def generate_answer_stream(
    query: str,
    bm25,
    bm25_ids,
    collection,
    client,
    top_n: int = DEFAULT_TOP_N,
):
    """
    Streaming counterpart to generate_answer(). The model takes the
    same total time to generate either way -- this doesn't make
    generation faster, it makes the wait feel much shorter, because
    text appears token-by-token instead of all at once after a
    long silent pause. This is what a UI (app.py) should use;
    generate_answer() remains for the CLI and for callers that just
    want the final string.

    Returns (token_generator, sources, chunks_used). Retrieval
    happens eagerly (sources/chunks_used are ready immediately);
    only generation is deferred to the generator, since sources
    don't depend on the model's output.
    """

    chunks_by_id = {c["id"]: c for c in load_chunks()}

    fused = hybrid_search(query, bm25, bm25_ids, collection, client, top_n=top_n)
    retrieved_chunks = [chunks_by_id[chunk_id] for chunk_id, _score, _src in fused if chunk_id in chunks_by_id]

    if not retrieved_chunks:
        no_result_text = "I couldn't find anything relevant to that question in the catalog."

        def empty_stream():
            yield no_result_text

        return empty_stream(), [], []

    context, sources = build_context(retrieved_chunks)
    messages = build_messages(query, context)

    def token_stream():
        for part in client.chat(
            model=ANSWER_MODEL,
            messages=messages,
            options={"num_ctx": NUM_CTX, "temperature": 0.1},
            keep_alive=KEEP_ALIVE,
            stream=True,
        ):
            piece = part["message"]["content"]
            if piece:
                yield piece

    return token_stream(), sources, [c["id"] for c in retrieved_chunks]


# ============================================================
# MAIN
# ============================================================

def load_everything():
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

    return bm25, bm25_ids, collection, client


def ask_and_print(query: str, bm25, bm25_ids, collection, client, top_n: int, stream: bool, timing: bool) -> None:
    if stream:
        t0 = time.perf_counter()
        token_gen, sources, _chunks_used = generate_answer_stream(
            query, bm25, bm25_ids, collection, client, top_n=top_n
        )
        t_retrieval = time.perf_counter() - t0

        print()
        first_token_time = None
        for piece in token_gen:
            if first_token_time is None:
                first_token_time = time.perf_counter() - t0
            print(piece, end="", flush=True)
        total = time.perf_counter() - t0

        footer = format_sources(sources)
        if footer:
            print(f"\n\n{footer}")

        if timing:
            print(
                f"\n\n[retrieval: {t_retrieval:.2f}s | "
                f"time to first token: {(first_token_time or total):.2f}s | "
                f"total: {total:.2f}s]"
            )
        return

    t0 = time.perf_counter()
    result = generate_answer(query, bm25, bm25_ids, collection, client, top_n=top_n)
    total = time.perf_counter() - t0

    print()
    print(result["answer"])

    if timing:
        print(f"\n[total: {total:.2f}s (retrieval + full generation, non-streaming)]")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask a question about the AUS catalog.")
    parser.add_argument("--query", type=str, default=None)
    parser.add_argument("--top-n", type=int, default=DEFAULT_TOP_N)
    parser.add_argument("--stream", action="store_true", help="Print tokens as they're generated instead of waiting for the full answer.")
    parser.add_argument("--timing", action="store_true", help="Print how long retrieval vs generation took.")
    args = parser.parse_args()

    print("Loading indexes...")
    bm25, bm25_ids, collection, client = load_everything()

    if args.query:
        ask_and_print(args.query, bm25, bm25_ids, collection, client, args.top_n, args.stream, args.timing)
        return

    print("Interactive mode. Type a question, or 'quit' to exit.")
    while True:
        try:
            query = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not query or query.lower() in {"quit", "exit"}:
            break

        ask_and_print(query, bm25, bm25_ids, collection, client, args.top_n, args.stream, args.timing)


if __name__ == "__main__":
    main()
