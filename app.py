"""
app.py

Streamlit chat UI for the AUS Undergraduate Catalog RAG assistant.
Ties together the BM25 index, Chroma vector index, and llama3.2
generation built in the previous steps.

Requires all prior steps to have been run first:
    python -m src.ingestion.preprocess
    python -m src.embeddings.build_index
    python -m src.retrieval.build_bm25_index
    ollama pull llama3.2

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import streamlit as st

from src.embeddings.build_index import get_collection, get_ollama_client
from src.generation.answer import DEFAULT_TOP_N, generate_answer_stream
from src.retrieval.build_bm25_index import check_index_freshness, load_bm25_index

st.set_page_config(
    page_title="AUS Catalog Assistant",
    page_icon="🎓",
    layout="wide",
)


# ============================================================
# BACKEND LOADING (cached so it only runs once per server process)
# ============================================================

@st.cache_resource(show_spinner="Loading indexes and connecting to Ollama...")
def load_backend():
    """
    Raises on failure rather than returning a sentinel, so the
    caller can show a specific, actionable error message instead
    of a generic crash. Each of these calls already raises a clear
    error (missing index file, Ollama not running, empty
    collection) -- we just need to surface it in the UI.
    """

    bm25, bm25_ids, signature = load_bm25_index()
    is_fresh = check_index_freshness(bm25_ids, signature)

    collection = get_collection()
    client = get_ollama_client()

    return {
        "bm25": bm25,
        "bm25_ids": bm25_ids,
        "bm25_fresh": is_fresh,
        "collection": collection,
        "client": client,
    }


def render_backend_error(exc: Exception) -> None:
    st.error("Couldn't start the assistant.")

    message = str(exc)
    st.code(message, language=None)

    st.markdown(
        "**Checklist:**\n"
        "1. Run the ingestion + indexing steps first, in order:\n"
        "   - `python -m src.ingestion.preprocess`\n"
        "   - `python -m src.embeddings.build_index`\n"
        "   - `python -m src.retrieval.build_bm25_index`\n"
        "2. Make sure Ollama is running: `ollama serve`\n"
        "3. Make sure both models are pulled:\n"
        "   - `ollama pull mxbai-embed-large`\n"
        "   - `ollama pull llama3.2`"
    )


# ============================================================
# SIDEBAR
# ============================================================

def render_sidebar(backend: dict) -> int:
    with st.sidebar:
        st.header("🎓 AUS Catalog Assistant")
        st.caption("Undergraduate Catalog 2025–2026")

        st.divider()

        st.subheader("Index status")
        st.metric("Chunks embedded (vector)", backend["collection"].count())
        st.metric("Chunks indexed (BM25)", len(backend["bm25_ids"]))

        if not backend["bm25_fresh"]:
            st.warning(
                "The BM25 index looks stale compared to the current "
                "chunks.json. Re-run build_bm25_index.py to refresh it.",
                icon="⚠️",
            )

        st.divider()

        top_n = st.slider(
            "Chunks to retrieve per question",
            min_value=3,
            max_value=15,
            value=DEFAULT_TOP_N,
            help="More chunks = more context for the model, but a slower, "
            "more expensive prompt. 6-8 works well for most questions.",
        )

        st.divider()

        if st.button("🗑️ Clear conversation"):
            st.session_state.messages = []
            st.rerun()

        st.divider()
        st.caption(
            "Answers are generated from catalog excerpts only. "
            "Always verify important details (prerequisites, deadlines, "
            "fees) against the official catalog PDF."
        )

    return top_n


# ============================================================
# CHAT
# ============================================================

def render_sources(sources: list[dict]) -> None:
    if not sources:
        return

    seen = set()
    rows = []
    for source in sorted(sources, key=lambda s: (s["page"] or 0)):
        label = source["course_code"] or source["title"] or source["type"] or "excerpt"
        key = (label, source["page"])
        if key in seen:
            continue
        seen.add(key)
        rows.append((label, source["page"], source["type"]))

    with st.expander(f"📄 Sources ({len(rows)})"):
        for label, page, chunk_type in rows:
            st.markdown(f"- **{label}** — page {page} *({chunk_type})*")


def render_history() -> None:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and message.get("sources"):
                render_sources(message["sources"])


def handle_new_question(backend: dict, top_n: int) -> None:
    query = st.chat_input("Ask about courses, prerequisites, programs, scholarships...")

    if not query:
        return

    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Searching the catalog..."):
            try:
                stream, sources, chunks_used = generate_answer_stream(
                    query,
                    backend["bm25"],
                    backend["bm25_ids"],
                    backend["collection"],
                    backend["client"],
                    top_n=top_n,
                )
            except Exception as exc:
                error_text = f"Something went wrong generating an answer: {exc}"
                st.error(error_text)
                st.session_state.messages.append({"role": "assistant", "content": error_text, "sources": []})
                return

        # st.write_stream renders each piece as it arrives (instead of
        # waiting for the full answer) and returns the concatenated
        # final text, which we still need for the chat history.
        answer_text = st.write_stream(stream)
        render_sources(sources)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer_text,
            "sources": sources,
        }
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    try:
        backend = load_backend()
    except Exception as exc:
        render_backend_error(exc)
        return

    top_n = render_sidebar(backend)

    st.title("AUS Undergraduate Catalog Assistant")
    st.caption(
        "Ask about course prerequisites, program requirements, scholarships, "
        "deadlines, fees, and more -- answers are grounded in the official "
        "catalog with page citations."
    )

    if not st.session_state.messages:
        st.info(
            "Try asking things like:\n"
            "- What are the prerequisites for MTH 221?\n"
            "- What is COE 486?\n"
            "- What courses are required for Computer Science?\n"
            "- What scholarships are available?\n"
            "- What's the phone number for the Academic Support Center?"
        )

    render_history()
    handle_new_question(backend, top_n)


if __name__ == "__main__":
    main()