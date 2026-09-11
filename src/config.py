from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

PDF_PATH = ROOT / "data" / "aus_ug_catalog_25-26.pdf"

PAGES_PATH = ROOT / "data" / "processed" / "pages.json"
TABLES_PATH = ROOT / "data" / "processed" / "tables.json"
CHUNKS_PATH = ROOT / "data" / "processed" / "chunks.json"
COURSES_PATH = ROOT / "data"/"processed" / "courses.json"
RECORDS_PATH = ROOT / "data"/"processed" / "records.json"
DOCUMENTS_PATH = ROOT / "data"/"processed" / "documents.json"

CHROMA_DIR = ROOT / "data" / "chroma"
BM25_PATH = ROOT / "data" / "processed" / "bm25.json"

INDEX_DIR = ROOT / "data" / "bm25_index"

COLLECTION_NAME = "aus_ug_catalog"

LLM_MODEL = "llama3.2"
EMBED_MODEL = "mxbai-embed-large"
EMBED_QUERY_PREFIX = "Represent this sentence for searching: "

COURSE_SECTION_PDF_START = 187
FACULTY_SECTION_PDF_START = 249

MAX_POLICY_CHARS = 1800
POLICY_OVERLAP_CHARS = 250

# RETRIEVE_K = 8
# BM25_K = 20
# VECTOR_K = 20
# RRF_K = 60
