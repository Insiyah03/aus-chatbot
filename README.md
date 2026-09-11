# AUS Catalog RAG Assistant

A local Retrieval-Augmented Generation (RAG) assistant for answering questions about the **American University of Sharjah (AUS) Undergraduate Catalog 2025–2026**.

The system retrieves relevant information from the catalog and uses a local Ollama language model to generate answers grounded in the retrieved content.

## Features

* Search the AUS Undergraduate Catalog using natural language
* Hybrid retrieval using:

  * Vector search
  * BM25 keyword search
* Cross-encoder reranking
* Local LLM generation using Ollama
* Streamlit web interface
* Source information for retrieved answers
* No external LLM API required

## Tech Stack

* **Python**
* **Streamlit**
* **Ollama**
* **ChromaDB**
* **mxbai-embed-large** for embeddings
* **BAAI/bge-reranker-base** for reranking
* **rank-bm25** for keyword search

## Project Structure

```text

```

## How It Works

```text
AUS Catalog PDF
      ↓
Text & Table Extraction
      ↓
Preprocessing & Chunking
      ↓
Vector Index + BM25 Index
      ↓
Hybrid Search
      ↓
Reranking
      ↓
LLM
      ↓
Answer
```

## Setup

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd AUS_RAG_Assistant
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Install Ollama

Install [Ollama](https://ollama.com/) and download the required models:

```bash
ollama pull mxbai-embed-large
ollama pull llama3.2
```

## Running the Assistant

From the project root:

```bash
streamlit run src/ui/app.py
```

The Streamlit interface will open in your browser.

## Example Questions

```text
What is COE 486?

What are the prerequisites for MTH 104?

How many credits is MTH 221?

What courses are required for the Computer Engineering minor?

What are the requirements for the Computer Science major?
```

## Data

The assistant is designed to answer questions using only the **AUS Undergraduate Catalog 2025–2026** provided in the `data` directory.

The processed catalog data and search indexes are generated locally.

## Status

The core retrieval pipeline is working. Current development is focused on improving evidence selection and answer reliability, particularly for course-specific questions and prerequisite relationships.
