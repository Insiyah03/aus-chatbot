# AUS Undergraduate Catalog RAG Assistant — Project Summary

A fully local, hybrid-retrieval RAG chatbot that answers questions about the AUS
Undergraduate Catalog 2025–2026, with page citations, running entirely on Ollama
(no cloud API calls, no data leaving the machine).

---

## Pipeline Overview

```
PDF
 ↓
pages.json + tables.json          (extract_pdf.py / extract_tables.py)
 ↓
preprocess.py
 ↓
courses.json + records.json + documents.json + chunks.json
 ↓
build_index.py            → ChromaDB (vector search, mxbai-embed-large)
build_bm25_index.py       → BM25 index (keyword search)
 ↓
hybrid_search.py          → Reciprocal Rank Fusion (combines both)
 ↓
answer.py                 → llama3.2 (generation + deterministic citations)
 ↓
app.py                    → Streamlit chat UI
```

---

## 1. Ingestion & Chunking (`src/ingestion/preprocess.py`)

Started from an existing pipeline and fixed real bugs found by inspecting actual
output, not just reading the code:

- **`tables.json` was defined but never loaded.** Directory, calendar, and
  curriculum-sequence content was being scraped line-by-line from flattened page
  text instead, which threw away row structure — e.g. a directory entry's name,
  phone, and email became three disconnected chunks. Rewired the pipeline to build
  proper row-level (directory, calendar) and whole-table (curriculum sequences,
  tuition/fees, placement tests) documents from the real table data instead.
- **Two silent course-loss bugs**, found by diffing what should exist against what
  actually appeared in `documents.json`:
  - A `re.VERBOSE` regex bug silently stripped literal spaces from `"Bridge Program
    Elective"`, making the BPE course pattern permanently unmatchable — its content
    was getting absorbed into the preceding course (`AUS 100`) instead.
  - The course title regex excluded all parentheses, so any course with a
    parenthetical in its title (`ARA 181` "...(Seerah)", `COE 457` "...(IoT)...",
    etc.) failed to match as a course boundary and got merged into the previous
    course's description. **5 courses were completely missing** from the dataset
    before this fix.
- Added a table-quality filter to drop false-positive "tables" (multi-column page
  text misidentified as tabular data by the PDF extractor).

**Result:** 964 → 969 courses recovered, directory/calendar/curriculum content
went from fragmented and context-less to properly structured, chunk count ~2,235 →
~2,296.

---

## 2. Embeddings (`src/embeddings/build_index.py`)

- Embeds every chunk with `mxbai-embed-large` via Ollama's batched `/api/embed`,
  stored in a persistent ChromaDB collection (cosine distance).
- **Resumable**: checks which chunk IDs are already indexed before embedding, so
  re-runs after a crash or a `chunks.json` update don't re-embed everything.
- Flattens Chroma-incompatible metadata (nested `prerequisites` list → joined
  string, `None` → `""`).
- Query-time embeddings get mxbai's recommended asymmetric prefix
  (`"Represent this sentence for searching relevant passages: "`); document-side
  embeddings don't.

---

## 3. Keyword Search (`src/retrieval/build_bm25_index.py`)

- `rank-bm25` index over the same chunks, **persisted to disk** (pickled, with a
  content-hash signature to detect when it's gone stale relative to
  `chunks.json`).
- Custom tokenizer fuses course codes into a single token regardless of spacing/
  case (`"COE 486"`, `"coe486"`, `"COE-486"` all match), so exact-identifier
  lookups are reliable — this is BM25's specific strength over vector search,
  which doesn't represent alphanumeric codes distinctively.

---

## 4. Hybrid Retrieval (`src/retrieval/hybrid_search.py`)

- Runs BM25 and vector search independently, then merges via **Reciprocal Rank
  Fusion** — each result scores `1/(k + rank)` in its own list, summed across
  lists. Chosen specifically because BM25 scores and cosine distances live on
  incompatible scales; RRF only needs rank position, not raw score normalization.
- `--compare` mode shows BM25-only / vector-only / hybrid side by side for a set
  of representative test questions, for eyeballing retrieval quality.

---

## 5. Answer Generation (`src/generation/answer.py`)

- Retrieves via hybrid search, builds a context block, and generates with
  `llama3.2` (or any Ollama model — configurable per call).
- **Citations are deterministic, not model-parsed.** The `Sources:` footer is
  built directly from retrieved-chunk metadata (page, course code), not by
  trusting the model to cite correctly inline — small local models are
  unreliable at that.
- **Multi-turn conversational memory**: recent turns are included in the prompt,
  and the retrieval query folds in the previous user turn, so follow-ups like
  *"what about its prerequisites?"* work without needing a second LLM call to
  rewrite the query.
- **Retrieval method attribution**: each source is tagged with whether it came
  from BM25, vector search, or both.
- Explicit `num_ctx` and `keep_alive` rather than Ollama's defaults — Ollama
  silently truncates context past `num_ctx` with no warning, and unloads idle
  models after ~5 minutes, both of which quietly hurt either correctness or
  latency if left on defaults.
- Both streaming (`generate_answer_stream`) and non-streaming (`generate_answer`)
  variants; streaming doesn't reduce total generation time, but changes perceived
  latency dramatically since text appears as it's produced.

---

## 6. Streamlit UI (`app.py`)

- Chat interface with conversation history, index-health metrics in the sidebar,
  and a "stale index" warning if `chunks.json` has changed since the BM25 index
  was last built.
- **Model switcher** (`llama3.2` / `llama3.2:1b` / custom tag) and an advanced
  settings panel (temperature, `num_ctx`, memory on/off) — all adjustable without
  touching code.
- **Claude-style rotating loading status** ("Thinking...", "Digging through the
  catalog...", etc.) shown while waiting for the model's first token — the actual
  slow part of the pipeline on CPU-only hardware, which previously had zero
  visual feedback because the spinner only wrapped the fast retrieval step. Uses
  a background thread to fetch the first token while the main thread keeps the
  displayed phrase rotating, then hands off to normal incremental rendering with
  a blinking cursor once real content starts arriving.
- Per-answer live timing (retrieval / time-to-first-token / total), a "Retrieved
  excerpts" debug expander showing the actual chunk text used for grounding,
  regenerate-last-answer, and export-conversation-to-markdown.

---

## Performance Investigation

Profiling with `--timing` on the CLI surfaced that **generation, not retrieval,
is the bottleneck** — retrieval consistently runs under 1.5s, while
time-to-first-token was 85–100+ seconds. `ollama ps` confirmed the model was
running `100% CPU`. The machine's GPU (`Intel(R) Arc(TM) Graphics`, an integrated
GPU on newer Intel Core Ultra laptops) isn't supported by standard Ollama on
Windows, so it silently falls back to CPU. Two paths forward were identified:

1. **Switch to a smaller model** (`llama3.2:1b`) — low effort, some quality
   tradeoff, works today via the model switcher already built into the UI.
2. **Intel's IPEX-LLM Ollama build** for actual Arc GPU acceleration — bigger
   effort, Windows support less mature, not yet attempted.

A hosted API (Claude/GPT) was discussed as a faster but non-local alternative —
not implemented, since it changes the project's fully-local design goal, but the
swap would be straightforward (`ollama.chat()` → an Anthropic/OpenAI SDK call)
if wanted later.

---

## How to Run It

```powershell
# One-time setup
pip install chromadb ollama rank-bm25 streamlit
ollama pull mxbai-embed-large
ollama pull llama3.2          # or llama3.2:1b for faster CPU inference

# Build the pipeline (in order)
python -m src.ingestion.preprocess
python -m src.embeddings.build_index
python -m src.retrieval.build_bm25_index

# Use it
streamlit run app.py                                    # UI
python -m src.generation.answer --query "..."            # CLI
python -m src.generation.answer --query "..." --stream --timing
python -m src.retrieval.hybrid_search --compare          # retrieval QA
```

---

## Known Limitations / Possible Future Work

- Curriculum-sequence tables aren't tagged with which major/program they belong
  to — a heading-proximity heuristic was tried and rejected (it returned things
  like "FIRST YEAR" instead of an actual major name); doing this properly needs
  an explicit page-range → major mapping.
- One residual garbled table (page 28 of the catalog) survives the quality
  filter — a false positive from the PDF table extractor that a length-based
  heuristic couldn't safely catch without also dropping legitimate tables with
  naturally long cell values (e.g. fee descriptions).
- No GPU acceleration on this machine's Intel Arc iGPU (see Performance
  Investigation above) — CPU inference is the current ceiling unless IPEX-LLM or
  a hosted API is adopted.
- A "stop generation" button was considered but not implemented — stock
  Streamlit can't cleanly interrupt a synchronous blocking stream mid-flight
  without more involved threading infrastructure than was justified here.