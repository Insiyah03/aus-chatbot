"""
app.py

Streamlit chat UI for the AUS Undergraduate Catalog RAG assistant.
Ties together the BM25 index, Chroma vector index, and llama3.2
generation built in the previous steps.

Beyond the basic chat loop, this version adds:
    - Multi-turn conversational memory (follow-up questions work)
    - A model switcher (llama3.2 / llama3.2:1b / custom) so you can
      trade quality for speed without editing code
    - Retrieval method attribution per source (bm25 / vector / both)
    - Live per-answer timing (retrieval / time-to-first-token / total)
    - A debug view of the actual retrieved excerpt text, for
      verifying the model is grounded in the right content
    - Regenerate-last-answer and export-conversation-to-markdown

Requires all prior steps to have been run first:
    python -m src.ingestion.preprocess
    python -m src.embeddings.build_index
    python -m src.retrieval.build_bm25_index
    ollama pull llama3.2

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import itertools
import threading
from datetime import datetime

import streamlit as st

from src.embeddings.build_index import get_collection, get_ollama_client, load_chunks
from src.generation.answer import ANSWER_MODEL, DEFAULT_TOP_N, NUM_CTX, generate_answer_stream
from src.retrieval.build_bm25_index import check_index_freshness, load_bm25_index

st.set_page_config(
    page_title="AUS Catalog Assistant",
    page_icon="🎓",
    layout="wide",
)

MODEL_OPTIONS = ["llama3.2", "llama3.2:1b", "Custom..."]
METHOD_ICON = {"bm25": "🔤", "vector": "🧠"}

# Cycled while waiting for the model's first token -- on CPU this can
# be tens of seconds with the actual Ollama call producing nothing to
# show, so a static spinner reads as "frozen." Rotating phrases (like
# Claude's own "Thinking...", "Pondering...") signal the app is still
# working without claiming to know real progress it doesn't have.
LOADING_PHRASES = [
    "Thinking...",
    "Digging through the catalog...",
    "Mulling it over...",
    "Consulting the course descriptions...",
    "Cross-referencing pages...",
    "Piecing the answer together...",
    "Still working on it...",
    "Searching the catalog...",
    "Checking the details...",
    "Looking through the catalog...",
    "Finding the relevant section...",
    "Connecting the dots...",
    "Checking the course records...",
    "Going through the fine print...",
    "Double-checking the details...",
    "Tracing that through the catalog...",
    "Looking for the exact answer...",
    "Scanning the relevant pages...",
    "Comparing the catalog entries...",
    "Checking the requirements...",
    "Reviewing the course information...",
    "Making sure the details line up...",
    "Finding the right page...",
    "Putting the pieces together...",
    "Checking one more thing...",
    "Reading between the pages...",
    "Doing a quick catalog check...",
    "Following the catalog trail...",
    "Digging a little deeper...",
    "Almost there...",
]


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


@st.cache_resource(show_spinner=False)
def load_chunks_by_id() -> dict[str, dict]:
    return {c["id"]: c for c in load_chunks()}


def render_backend_error(exc: Exception) -> None:
    st.error("Couldn't start the assistant.")

    st.code(str(exc), language=None)

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

def render_sidebar(backend: dict) -> dict:
    """Returns the current generation settings as a dict."""

    with st.sidebar:
        st.header("🎓 AUS Catalog Assistant")
        st.caption("Undergraduate Catalog 2025–2026")

        st.divider()

        st.subheader("Index status")
        col1, col2 = st.columns(2)
        col1.metric("Vector chunks", backend["collection"].count())
        col2.metric("BM25 chunks", len(backend["bm25_ids"]))

        if not backend["bm25_fresh"]:
            st.warning(
                "BM25 index looks stale vs. current chunks.json. "
                "Re-run build_bm25_index.py.",
                icon="⚠️",
            )

        st.divider()

        st.subheader("Model")
        model_choice = st.selectbox("Answer model", MODEL_OPTIONS, index=0)
        if model_choice == "Custom...":
            model = st.text_input("Model tag", value=ANSWER_MODEL)
        else:
            model = model_choice

        if model != ANSWER_MODEL:
            st.caption(f"Using **{model}** instead of the default ({ANSWER_MODEL}).")

        top_n = st.slider(
            "Chunks retrieved per question",
            min_value=3,
            max_value=15,
            value=DEFAULT_TOP_N,
            help="More chunks = more context, but a slower prompt.",
        )

        with st.expander("Advanced settings"):
            temperature = st.slider(
                "Temperature",
                min_value=0.0,
                max_value=1.0,
                value=0.1,
                step=0.05,
                help="Lower = sticks closer to the retrieved text. Higher = more creative phrasing, more risk of drifting from the source.",
            )
            num_ctx = st.number_input(
                "Context window (num_ctx)",
                min_value=1024,
                max_value=32768,
                value=NUM_CTX,
                step=1024,
                help="Must be large enough for the system prompt + retrieved chunks + conversation history. "
                "Oversizing this can slow down or destabilize generation on memory-constrained hardware.",
            )
            memory_on = st.checkbox(
                "Remember conversation (multi-turn)",
                value=True,
                help="Include recent turns so follow-up questions like 'what about its prerequisites?' work. Adds a bit to each prompt.",
            )

        st.divider()

        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Clear", use_container_width=True):
                st.session_state.messages = []
                st.session_state.last_query = None
                st.rerun()
        with col2:
            has_history = bool(st.session_state.get("messages"))
            markdown_export = build_markdown_export(st.session_state.get("messages", []))
            st.download_button(
                "💾 Export",
                data=markdown_export,
                file_name=f"aus_catalog_chat_{datetime.now():%Y%m%d_%H%M}.md",
                mime="text/markdown",
                disabled=not has_history,
                use_container_width=True,
            )

        st.divider()
        st.caption(
            "Answers are generated from catalog excerpts only. "
            "Always verify important details (prerequisites, deadlines, "
            "fees) against the official catalog PDF."
        )

    return {
        "model": model,
        "top_n": top_n,
        "temperature": temperature,
        "num_ctx": int(num_ctx),
        "memory_on": memory_on,
    }


# ============================================================
# EXPORT
# ============================================================

def build_markdown_export(messages: list[dict]) -> str:
    lines = [f"# AUS Catalog Assistant — conversation export", f"_{datetime.now():%Y-%m-%d %H:%M}_", ""]

    for message in messages:
        role = "**You**" if message["role"] == "user" else "**Assistant**"
        lines.append(f"{role}: {message['content']}")

        if message["role"] == "assistant" and message.get("sources"):
            for source in message["sources"]:
                label = source["course_code"] or source["title"] or source["type"] or "excerpt"
                lines.append(f"  - {label} (p. {source['page']})")

        lines.append("")

    return "\n".join(lines)


# ============================================================
# SOURCES / DEBUG DISPLAY
# ============================================================

def method_badge(methods: list[str]) -> str:
    if not methods:
        return "?"
    if set(methods) == {"bm25", "vector"}:
        return "🔀 both"
    return " ".join(f"{METHOD_ICON.get(m, '')} {m}".strip() for m in methods)


def dedupe_sources(sources: list[dict]) -> list[dict]:
    seen = set()
    deduped = []
    for source in sorted(sources, key=lambda s: (s["page"] or 0)):
        label = source["course_code"] or source["title"] or source["type"] or "excerpt"
        key = (label, source["page"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append({**source, "label": label})
    return deduped


def render_sources(sources: list[dict]) -> None:
    if not sources:
        return

    rows = dedupe_sources(sources)

    with st.expander(f"📄 Sources ({len(rows)})"):
        for source in rows:
            badge = method_badge(source.get("methods") or [])
            st.markdown(f"- **{source['label']}** — page {source['page']} · {badge} · _{source['type']}_")


def render_debug_excerpts(chunk_ids: list[str]) -> None:
    if not chunk_ids:
        return

    chunks_by_id = load_chunks_by_id()

    with st.expander(f"🔍 Retrieved excerpts ({len(chunk_ids)}) — for verifying grounding"):
        for chunk_id in chunk_ids:
            chunk = chunks_by_id.get(chunk_id)
            if not chunk:
                continue
            label = chunk.get("course_code") or chunk.get("title") or chunk.get("type")
            st.markdown(f"**{label}** — page {chunk.get('page')}")
            st.text(chunk.get("text", "")[:800])
            st.divider()


def stream_with_loading_indicator(token_gen, placeholder) -> str:
    """
    Renders token_gen into `placeholder`, showing a rotating
    Claude-style status phrase while waiting for the first token
    and a normal incremental render (with a blinking cursor) once
    tokens start arriving.

    Why this needs a thread: fetching the first item from token_gen
    is what actually triggers Ollama's request and blocks until the
    model produces something -- on CPU that can be tens of seconds
    with nothing to show otherwise. A plain st.spinner() can only
    show one static message for that whole span. Running that first
    next() call in a background thread lets the main thread keep
    updating the visible phrase every ~1.2s while waiting, then hand
    off to normal rendering the moment the real content arrives.

    The background thread only touches plain Python (iterating the
    generator, which ultimately calls the Ollama client) -- all
    Streamlit UI calls stay on the main thread, which is required.
    """

    result: dict = {}

    def fetch_first():
        try:
            result["first"] = next(token_gen)
        except StopIteration:
            result["first"] = None
        except Exception as exc:  # noqa: BLE001 - surfaced to the caller below
            result["error"] = exc

    worker = threading.Thread(target=fetch_first, daemon=True)
    worker.start()

    phrases = itertools.cycle(LOADING_PHRASES)
    while worker.is_alive():
        placeholder.markdown(f"_{next(phrases)}_")
        worker.join(timeout=1.2)

    if "error" in result:
        raise result["error"]

    accumulated = result.get("first") or ""
    placeholder.markdown(accumulated + "▌")

    for piece in token_gen:
        accumulated += piece
        placeholder.markdown(accumulated + "▌")

    placeholder.markdown(accumulated)

    return accumulated


def render_timing(stats: dict) -> None:
    if not stats:
        return

    retrieval = stats.get("retrieval_s")
    first_token = stats.get("first_token_s")
    total = stats.get("total_s")

    parts = []
    if retrieval is not None:
        parts.append(f"retrieval {retrieval:.2f}s")
    if first_token is not None:
        parts.append(f"first token {first_token:.2f}s")
    if total is not None:
        parts.append(f"total {total:.2f}s")

    if parts:
        st.caption("⏱️ " + " · ".join(parts))


# ============================================================
# CHAT
# ============================================================

def build_history(messages: list[dict]) -> list[dict]:
    """Strip UI-only fields (sources, stats) down to role/content for the model."""

    return [{"role": m["role"], "content": m["content"]} for m in messages]


def render_history() -> None:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                render_timing(message.get("stats") or {})
                render_sources(message.get("sources") or [])
                render_debug_excerpts(message.get("chunks_used") or [])


def run_turn(backend: dict, settings: dict, query: str, history_messages: list[dict]) -> None:
    """Runs one question through retrieval + generation and renders it live."""

    with st.chat_message("assistant"):
        stats: dict = {}

        with st.spinner("Searching the catalog..."):
            try:
                history = build_history(history_messages) if settings["memory_on"] else None
                stream, sources, chunks_used = generate_answer_stream(
                    query,
                    backend["bm25"],
                    backend["bm25_ids"],
                    backend["collection"],
                    backend["client"],
                    top_n=settings["top_n"],
                    history=history,
                    model=settings["model"],
                    temperature=settings["temperature"],
                    num_ctx=settings["num_ctx"],
                    stats=stats,
                )
            except Exception as exc:
                error_text = f"Something went wrong generating an answer: {exc}"
                st.error(error_text)
                st.session_state.messages.append({"role": "assistant", "content": error_text, "sources": [], "stats": {}, "chunks_used": []})
                return

        answer_placeholder = st.empty()
        try:
            answer_text = stream_with_loading_indicator(stream, answer_placeholder)
        except Exception as exc:
            error_text = f"Something went wrong while generating: {exc}"
            answer_placeholder.markdown(error_text)
            st.session_state.messages.append({"role": "assistant", "content": error_text, "sources": [], "stats": stats, "chunks_used": []})
            return

        render_timing(stats)
        render_sources(sources)
        render_debug_excerpts(chunks_used)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer_text,
            "sources": sources,
            "stats": stats,
            "chunks_used": chunks_used,
        }
    )


def handle_new_question(backend: dict, settings: dict) -> None:
    query = st.chat_input("Ask about courses, prerequisites, programs, scholarships...")

    if not query:
        return

    history_messages = list(st.session_state.messages)  # snapshot before appending the new turn

    st.session_state.messages.append({"role": "user", "content": query})
    st.session_state.last_query = query
    with st.chat_message("user"):
        st.markdown(query)

    run_turn(backend, settings, query, history_messages)


def handle_regenerate(backend: dict, settings: dict) -> None:
    last_query = st.session_state.get("last_query")

    if not last_query:
        return

    # Drop the last assistant turn (if any) so we don't end up with two
    # answers to the same question stacked in the history.
    if st.session_state.messages and st.session_state.messages[-1]["role"] == "assistant":
        st.session_state.messages.pop()

    history_messages = list(st.session_state.messages[:-1]) if st.session_state.messages else []

    run_turn(backend, settings, last_query, history_messages)
    st.rerun()


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "last_query" not in st.session_state:
        st.session_state.last_query = None

    try:
        backend = load_backend()
    except Exception as exc:
        render_backend_error(exc)
        return

    settings = render_sidebar(backend)

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
            "- What's the phone number for the Academic Support Center?\n\n"
            "With memory on, you can ask a follow-up like *'what about its prerequisites?'* "
            "right after asking about a course."
        )

    render_history()

    if st.session_state.messages and st.session_state.messages[-1]["role"] == "assistant":
        if st.button("🔄 Regenerate last answer"):
            handle_regenerate(backend, settings)

    handle_new_question(backend, settings)


if __name__ == "__main__":
    main()