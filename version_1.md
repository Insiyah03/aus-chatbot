# AUS Undergraduate Catalog RAG Assistant

## Project Goal

I am building a **local RAG chatbot** that answers questions using the **AUS Undergraduate Catalog 2025–2026 PDF**.

Example questions:
- What are the prerequisites for MTH 221?
- What is COE 486?
- What courses are required for Computer Science?
- What scholarships are available?
- What are the requirements for a specific major?

The final system should retrieve relevant catalog information and use a local LLM to generate an answer with **PDF page citations**.

---

## Tech Stack

- Python
- PyMuPDF (`pymupdf`) — PDF text extraction
- pdfplumber — table extraction
- Ollama
- `mxbai-embed-large` — embeddings
- `llama3.2` — answer generation
- ChromaDB — vector database
- BM25 (`rank-bm25`) — keyword search
- Streamlit — final UI

Ollama models:

```powershell
ollama pull llama3.2
ollama pull llama
ollama pull mxbai-embed-large
```

---
## Project Structure
``` txt
AUS_RAG_Assistant/
├── data/
│   ├── processed/
│   │   ├── chunks.json
│   │   ├── courses.json
│   │   ├── documents.json
│   │   ├── pages.json
│   │   ├── records.json
│   │   └── tables.json
│   └── aus_ug_catalog_25-26.pdf
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   └── ingestion/
│       ├── __init__.py
│       ├── extract_pdf.py
│       ├── extract_tables.py
│       ├── clean.py
│       ├── preprocess.py
│       └── chunk.py
│
├── venv/
├── requirements.txt
└── testing.ipynb
```
---
## What the Files Do

```
PDF
 ↓
extract_pdf.py
 ↓
pages.json
 ↓
extract_tables.py
 ↓
tables.json
 ↓
preprocess.py
 ↓
courses.json + records.json
 ↓
documents.json
 ↓
chunk.py
 ↓
chunks.json
 ↓
Embeddings
 ↓
ChromaDB
 ↓
Hybrid Retrieval
 ↓
llama3.2
 ↓
Answer
```

**JSON files**  
- `pages.json` → extracted page text
- `tables.json` → extracted tables
- `courses.json` → structured course information
- `records.json` → structured non-course information
- `documents.json` → unified searchable documents
- `chunks.json` → final chunks that will be embedded
---
## Current Preprocessing Status

The PDF preprocessing pipeline is working.

Current output:
```
Courses extracted:     964
Unique course codes:   961
Structured records:    375
Total documents:       1339
Total chunks:          2235
```
Course information is being stored in a structured format such as:
```
Course: MTH 221
Title: Linear Algebra

Description:
...

Prerequisites:
...
```
Page numbers are preserved as metadata so the final chatbot can cite the source PDF page.

---
## Retrieval Strategy
We want **hybrid** retrieval instead of only vector search.

**Vector search**  
Using:
`
mxbai-embed-large
`  
Good for semantic questions.

**BM25**  
Good for exact terms such as:
```
MTH 221
COE 486
CMP 220
Computer Science
```

**Hybrid**
```
User Question
      ↓
 ┌────┴────┐
 ↓         ↓
Vector    BM25
Search    Search
 └────┬────┘
      ↓
Combined/Reranked Results
      ↓
Relevant Catalog Context
      ↓
llama3.2
      ↓
Answer + Sources
```

---
## Important Chunking Approach

Courses should preferably remain as one coherent chunk/document, rather than being arbitrarily split.  
For example:  
```
Course: MTH 221
Title: Linear Algebra

Description:
...

Prerequisites:
...
```
Program information and long prose should be split into logical chunks.  
Tables should retain context so that rows are understandable when retrieved. 

---
## Current Task

The PDF extraction/preprocessing is essentially done.  
**Next step:**

Review and finalize `chunk.py` and `chunks.json`.

We should check that:

- course information stays together
- prerequisites remain attached to courses
- program information is chunked sensibly
- tables retain their context
- metadata/page numbers are preserved
- chunks are not too small or too large

After that:

1) Generate embeddings with mxbai-embed-large
2) Store them in ChromaDB
3) Build BM25 search
4) Implement hybrid retrieval
5) Connect llama3.2
6) Add source/page citations
7) Build the Streamlit interface
8) Test the RAG system with real catalog questions

---
## Current Position

We are not starting from scratch.  
The ingestion and preprocessing pipeline is already built.

The next thing to work on is: `src/ingestion/chunk.py` and: `data/processed/chunks.json`

Then we move into the actual RAG/retrieval stage.