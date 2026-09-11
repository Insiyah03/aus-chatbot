import json
import re
from pathlib import Path

from src.config import (
    CHUNKS_PATH,
    COURSE_SECTION_PDF_START,
    FACULTY_SECTION_PDF_START,
    MAX_POLICY_CHARS,
    PAGES_PATH,
    POLICY_OVERLAP_CHARS,
    TABLES_PATH,
)
from src.ingestion.clean import extract_catalog_page, normalize_whitespace, strip_running_headers

COURSE_START = re.compile(
    r"(?:(?<=\n)|(?<=^))"
    r"([A-Z]{2,4})\s+(\d{3}[A-Z]{0,2})\s+"
    r"([A-Z][^\n()]{2,90}(?:\n[A-Za-z][^\n()]{2,60})?)\s*"
    r"(\((?:\d+-\d+-\d+|[^)]{0,40}credit[^)]{0,20}|[^)]{0,20}hours?[^)]{0,20})\))\.\s+",
)


def load_json(path: Path):
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def table_to_markdown(rows: list[list[str]]) -> str:
    cleaned = [[(cell or "").strip() for cell in row] for row in rows]
    cleaned = [row for row in cleaned if any(cell for cell in row)]
    if not cleaned:
        return ""

    width = max(len(row) for row in cleaned)
    cleaned = [row + [""] * (width - len(row)) for row in cleaned]
    header = cleaned[0]
    body = cleaned[1:] or [[""] * width]

    def fmt(row):
        return "| " + " | ".join(row) + " |"

    lines = [fmt(header), "| " + " | ".join("---" for _ in header) + " |"]
    lines.extend(fmt(row) for row in body)
    return "\n".join(lines)


def is_useful_table(rows: list[list[str]]) -> bool:
    cleaned = [[(cell or "").strip() for cell in row] for row in rows]
    nonempty_rows = [row for row in cleaned if any(row)]
    if len(nonempty_rows) < 2:
        return False
    width = max(sum(1 for cell in row if cell) for row in nonempty_rows)
    if width < 2:
        return False
    cells = [cell for row in nonempty_rows for cell in row if cell]
    if len(cells) < 4:
        return False
    unique = {cell.lower() for cell in cells}
    return len(unique) >= 3


def annotate_pages(pages: list[dict], start: int, end: int) -> tuple[str, list[dict]]:
    parts = []
    markers = []
    cursor = 0
    for page in pages:
        if page["page"] < start or page["page"] >= end:
            continue
        cleaned = strip_running_headers(page.get("text") or "")
        if not cleaned:
            continue
        catalog_page = extract_catalog_page(page.get("text") or "")
        marker = f"\n[[[PDF:{page['page']}|CAT:{catalog_page or ''}]]]\n"
        parts.append(marker + cleaned)
        markers.append(
            {
                "offset": cursor,
                "pdf_page": page["page"],
                "catalog_page": catalog_page,
            }
        )
        cursor += len(marker) + len(cleaned)
    return "\n\n".join(parts), markers


def page_at(offset: int, markers: list[dict]) -> dict:
    current = markers[0] if markers else {"pdf_page": None, "catalog_page": None}
    for marker in markers:
        if marker["offset"] <= offset:
            current = marker
        else:
            break
    return current


def split_courses(text: str, markers: list[dict]) -> list[dict]:
    matches = list(COURSE_START.finditer(text))
    chunks = []
    for i, match in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        prefix, number, title, hours = match.groups()
        title = normalize_whitespace(title.replace("\n", " "))
        if title.lower().startswith("and ") or "Independent Study" in title:
            continue
        code = f"{prefix} {number}"
        body = normalize_whitespace(text[match.end() : end].replace("\n", " "))
        start_meta = page_at(match.start(), markers)
        end_meta = page_at(end - 1 if end else match.start(), markers)
        full = f"{code} {title} {hours}. {body}".strip()
        chunks.append(
            {
                "chunk_type": "course",
                "section": "Undergraduate Course Descriptions",
                "course_code": code,
                "title": title,
                "text": full,
                "page_start": start_meta["pdf_page"],
                "page_end": end_meta["pdf_page"],
                "catalog_page_start": start_meta["catalog_page"],
                "catalog_page_end": end_meta["catalog_page"],
            }
        )
    return chunks


def page_section_label(cleaned: str, fallback: str) -> str:
    for line in cleaned.splitlines()[:4]:
        stripped = line.strip()
        if 4 <= len(stripped) <= 70 and not stripped.endswith("."):
            return stripped
    return fallback


def window_text(text: str, max_chars: int, overlap: int) -> list[str]:
    text = normalize_whitespace(text.replace("\n", " "))
    if len(text) <= max_chars:
        return [text] if text else []
    windows = []
    start = 0
    while start < len(text):
        end = min(len(text), start + max_chars)
        if end < len(text):
            cut = text.rfind(". ", start, end)
            if cut > start + max_chars // 2:
                end = cut + 1
        windows.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(0, end - overlap)
    return windows


def split_policies(pages: list[dict], start: int, end: int, chunk_type: str = "policy") -> list[dict]:
    chunks = []
    last_section = "Undergraduate Catalog"
    for page in pages:
        if page["page"] < start or page["page"] >= end:
            continue
        catalog_page = extract_catalog_page(page.get("text") or "")
        cleaned = strip_running_headers(page.get("text") or "")
        if not cleaned:
            continue
        section = page_section_label(cleaned, last_section)
        last_section = section
        for window in window_text(cleaned, MAX_POLICY_CHARS, POLICY_OVERLAP_CHARS):
            chunks.append(
                {
                    "chunk_type": chunk_type,
                    "section": section,
                    "course_code": "",
                    "title": section,
                    "text": window,
                    "page_start": page["page"],
                    "page_end": page["page"],
                    "catalog_page_start": catalog_page,
                    "catalog_page_end": catalog_page,
                }
            )
    return chunks


def split_tables(tables: list[dict]) -> list[dict]:
    chunks = []
    for table in tables:
        rows = table.get("rows") or []
        if not is_useful_table(rows):
            continue
        markdown = table_to_markdown(rows)
        if not markdown:
            continue
        page = table["page"]
        chunks.append(
            {
                "chunk_type": "table",
                "section": "Table",
                "course_code": "",
                "title": f"Table on PDF page {page}",
                "text": f"Table from undergraduate catalog (PDF page {page}):\n{markdown}",
                "page_start": page,
                "page_end": page,
                "catalog_page_start": "",
                "catalog_page_end": "",
            }
        )
    return chunks


def assign_ids(chunks: list[dict]) -> list[dict]:
    numbered = []
    for i, chunk in enumerate(chunks):
        item = dict(chunk)
        item["id"] = f"{chunk['chunk_type']}-{i:05d}"
        numbered.append(item)
    return numbered


def build_chunks() -> list[dict]:
    pages = load_json(PAGES_PATH)
    tables = load_json(TABLES_PATH)

    course_text, course_markers = annotate_pages(
        pages, COURSE_SECTION_PDF_START, FACULTY_SECTION_PDF_START
    )
    courses = split_courses(course_text, course_markers)
    policies = split_policies(pages, 1, COURSE_SECTION_PDF_START)
    faculty = split_policies(
        pages, FACULTY_SECTION_PDF_START, 10_000, chunk_type="policy"
    )
    table_chunks = split_tables(tables)

    chunks = assign_ids(policies + courses + faculty + table_chunks)
    CHUNKS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CHUNKS_PATH.open("w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)
    return chunks


if __name__ == "__main__":
    built = build_chunks()
    counts: dict[str, int] = {}
    for chunk in built:
        counts[chunk["chunk_type"]] = counts.get(chunk["chunk_type"], 0) + 1
    print(f"Wrote {len(built)} chunks to {CHUNKS_PATH}")
    print(counts)
