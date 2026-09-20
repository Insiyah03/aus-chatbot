"""
preprocess.py

Structure-aware preprocessing for the AUS Undergraduate Catalog.

Input:
    data/processed/pages.json
    data/processed/tables.json  

Output:
    data/processed/courses.json
    data/processed/records.json
    data/processed/documents.json
    data/processed/chunks.json

The catalog is divided into sections using PDF viewer page numbers.

Important:
- Course extraction is ONLY performed on the actual Course Descriptions
  section (PDF pages 36-37 and 187-248).
- TOC, index, charts, maps, etc. are excluded.
- Course boundaries are detected using the "(x-x-x)." course format.
- Other catalog content is converted into structured semantic records.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any
from src.config import (PAGES_PATH,TABLES_PATH,COURSES_PATH,RECORDS_PATH,DOCUMENTS_PATH,CHUNKS_PATH)
from src.config import (MAX_POLICY_CHARS,POLICY_OVERLAP_CHARS)

# ============================================================
# PATHS
# ============================================================


PAGES_FILE = PAGES_PATH
TABLES_FILE = TABLES_PATH

COURSES_FILE = COURSES_PATH
RECORDS_FILE = RECORDS_PATH
DOCUMENTS_FILE = DOCUMENTS_PATH
CHUNKS_FILE = CHUNKS_PATH


# ============================================================
# PAGE RANGES
# ============================================================

# PDF viewer page numbers.
#
# These are intentionally based on the actual PDF pages rather
# than the printed catalog page numbers.

SECTION_RANGES = {
    "general": [(9, 34)],
    "achievement_courses": [(35, 37)],
    "admissions": [(38, 44)],
    "policies": [(45, 67)],
    "tuition_scholarships": [(68, 71)],
    "programs": [(72, 186)],
    "courses": [(187, 248)],
    "faculty": [(249, 254)],
    "directory": [(16, 16)],
}


# Pages that should not be embedded.
#
# These are mostly images, charts, navigation material, etc.

EXCLUDED_PAGES = set()

for start, end in [
    (1, 8),       # image/front matter
    (11, 13),     # organizational charts
    (17, 17),     # campus map
    (20, 20),     # blank
    (22, 23),     # table of contents
    (255, 266),   # index
]:
    EXCLUDED_PAGES.update(range(start, end + 1))


# ============================================================
# COURSE REGEX
# ============================================================

# Standard course:
#
#     ARC 201 Architectural Design I (3-0-0).
#     CSE 312A Something (3-0-0).
#     MTH 103 Mathematics (3-0-0).
# The key feature is the "(x-x-x)." marker.
# We deliberately DO NOT require the course code to appear at
# the beginning of a line because PDF extraction can flatten
# columns/lines.

COURSE_START_RE = re.compile(
    r"""
    (?P<code>
        [A-Z]{2,4}
        \s+
        \d{3}[A-Z]?
        (?:-\d{2})?
    )
    \s+
    (?P<title>
        [A-Z][^\n]{1,150}?
    )
    \s+
    \(
        (?P<credits>
            \d+(?:\.\d+)?-
            \d+(?:\.\d+)?-
            \d+(?:\.\d+)?
        )
    \)
    \.
    """,
    re.VERBOSE,
)


# BPE has a special unnumbered course-like entry.
BPE_UNNUMBERED_RE = re.compile(
    r"""
    (?P<code>BPE)
    \s+
    (?P<title>Bridge\s+Program\s+Elective)
    \s+
    \(
        (?P<credits>
            \d+(?:\.\d+)?-
            \d+(?:\.\d+)?-
            \d+(?:\.\d+)?
        )
    \)
    \.
    """,
    re.VERBOSE,
)


COURSE_CODE_RE = re.compile(
    r"\b[A-Z]{2,4}\s+\d{3}[A-Z]?(?:-\d{2})?\b"
)


# ============================================================
# GENERAL REGEXES
# ============================================================

PREREQUISITE_RE = re.compile(
    r"(?i)\bprerequisite[s]?\s*:\s*(.+?)(?="
    r"\b(?:corequisite|co-requisite|notes?|course description)\b|$)"
)


COURSE_REFERENCE_RE = re.compile(
    r"\b[A-Z]{2,4}\s+\d{3}[A-Z]?(?:-\d{2})?\b"
)


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def load_json(path: Path, default: Any = None) -> Any:
    """Load JSON safely."""

    if not path.exists():
        return default

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: Any) -> None:
    """Save JSON with readable formatting."""

    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            data,
            f,
            indent=2,
            ensure_ascii=False,
        )


def clean_whitespace(text: str) -> str:
    """Normalize whitespace."""

    if not text:
        return ""

    text = text.replace("\u00a0", " ")
    text = text.replace("\u200b", "")
    text = text.replace("\r", "\n")

    # Normalize repeated spaces/tabs.
    text = re.sub(r"[ \t]+", " ", text)

    # Normalize excessive blank lines.
    text = re.sub(r"\n\s*\n+", "\n\n", text)

    return text.strip()


def normalize_for_matching(text: str) -> str:
    """
    Normalize text for regex matching.

    Unlike the earlier version, we preserve sentence boundaries
    as much as possible instead of blindly removing every newline.
    """

    if not text:
        return ""

    text = text.replace("\u00a0", " ")
    text = text.replace("\r", "\n")

    # Remove excessive whitespace.
    text = re.sub(r"[ \t]+", " ", text)

    # A newline inside a sentence becomes a space.
    text = re.sub(r"\n+", "\n", text)

    return text.strip()


def normalize_course_text(text: str) -> str:
    """Normalize text specifically for course descriptions."""

    if not text:
        return ""

    text = text.replace("\u00a0", " ")
    text = text.replace("\r", "\n")

    # Join lines that were broken by PDF extraction.
    text = re.sub(r"[ \t]*\n[ \t]*", " ", text)

    # Collapse whitespace.
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def page_in_ranges(page_number: int, ranges: list[tuple[int, int]]) -> bool:
    """Check whether a page belongs to one of the supplied ranges."""

    for start, end in ranges:
        if start <= page_number <= end:
            return True

    return False


def get_section(page_number: int) -> str:
    """Return the section associated with a PDF page."""

    # Courses must be checked first because pages 35-37 are special.
    if page_in_ranges(page_number, SECTION_RANGES["achievement_courses"]):
        return "achievement_courses"

    if page_in_ranges(page_number, SECTION_RANGES["courses"]):
        return "courses"

    if page_in_ranges(page_number, SECTION_RANGES["faculty"]):
        return "faculty"

    if page_in_ranges(page_number, SECTION_RANGES["directory"]):
        return "directory"

    if page_in_ranges(page_number, SECTION_RANGES["admissions"]):
        return "admissions"

    if page_in_ranges(page_number, SECTION_RANGES["policies"]):
        return "policies"

    if page_in_ranges(page_number, SECTION_RANGES["tuition_scholarships"]):
        return "tuition_scholarships"

    if page_in_ranges(page_number, SECTION_RANGES["programs"]):
        return "programs"

    if page_in_ranges(page_number, SECTION_RANGES["general"]):
        return "general"

    return "unknown"


# ============================================================
# PAGE TEXT EXTRACTION
# ============================================================

def get_page_number(page: dict, fallback: int) -> int:
    """Get page number from different possible pages.json formats."""

    for key in ("page", "page_number", "page_num", "number"):
        value = page.get(key)

        if isinstance(value, int):
            return value

        if isinstance(value, str) and value.isdigit():
            return int(value)

    return fallback


def get_page_text(page: dict) -> str:
    """Extract text from a page object."""

    possible_keys = [
        "text",
        "content",
        "page_text",
        "full_text",
    ]

    for key in possible_keys:
        value = page.get(key)

        if isinstance(value, str) and value.strip():
            return value

    # Some extraction scripts store blocks instead of one text field.
    blocks = page.get("blocks")

    if isinstance(blocks, list):
        texts = []

        for block in blocks:
            if isinstance(block, dict):
                text = block.get("text")

                if isinstance(text, str):
                    texts.append(text)

            elif isinstance(block, str):
                texts.append(block)

        return "\n".join(texts)

    return ""


# ============================================================
# COLUMN HANDLING
# ============================================================

def extract_columns_from_page(page: dict) -> str:
    """
    Extract PDF blocks in approximate reading order.

    This is useful because most catalog pages contain 3 columns.

    Expected block format:

        {
            "x0": ...,
            "y0": ...,
            "x1": ...,
            "y1": ...,
            "text": "..."
        }

    If blocks are not available, fall back to normal page text.
    """

    blocks = page.get("blocks")

    if not isinstance(blocks, list):
        return get_page_text(page)

    usable_blocks = []

    for block in blocks:

        if not isinstance(block, dict):
            continue

        text = block.get("text", "")

        if not isinstance(text, str):
            continue

        text = text.strip()

        if not text:
            continue

        try:
            x0 = float(block.get("x0", 0))
            y0 = float(block.get("y0", 0))
            x1 = float(block.get("x1", x0))
            y1 = float(block.get("y1", y0))
        except (TypeError, ValueError):
            continue

        usable_blocks.append(
            {
                "text": text,
                "x0": x0,
                "y0": y0,
                "x1": x1,
                "y1": y1,
            }
        )

    if not usable_blocks:
        return get_page_text(page)

    # --------------------------------------------------------
    # Estimate columns.
    # --------------------------------------------------------

    page_width = max(
        block["x1"]
        for block in usable_blocks
    )

    # Rough column assignment.
    #
    # This assumes the PDF is approximately letter/A4 width.
    # We use relative position rather than fixed coordinates.

    def column_number(block: dict) -> int:
        center = (block["x0"] + block["x1"]) / 2

        ratio = center / max(page_width, 1)

        if ratio < 0.34:
            return 0

        if ratio < 0.67:
            return 1

        return 2

    for block in usable_blocks:
        block["column"] = column_number(block)

    # --------------------------------------------------------
    # Sort top-to-bottom within each column.
    # --------------------------------------------------------

    ordered = []

    for column in (0, 1, 2):

        column_blocks = [
            block
            for block in usable_blocks
            if block["column"] == column
        ]

        column_blocks.sort(
            key=lambda b: (
                round(b["y0"], 1),
                round(b["x0"], 1),
            )
        )

        ordered.extend(
            block["text"]
            for block in column_blocks
        )

    return "\n".join(ordered)


# ============================================================
# COURSE VALIDATION
# ============================================================

BAD_COURSE_TITLE_PHRASES = [
    "prerequisite:",
    "prerequisites:",
    "students are",
    "students who",
    "students may",
    "students must",
    "students can",
    "course descriptions",
    "academic policies",
    "program requirements",
    "degree requirements",
    "the university",
    "admission to",
    "achievement academy",
]


def is_valid_course_title(title: str) -> bool:
    """
    Reject obvious false-positive course matches.
    """

    if not title:
        return False

    title = clean_whitespace(title)

    # Course titles should not be enormous.
    if len(title) < 2 or len(title) > 160:
        return False

    lower_title = title.lower()

    # Reject obvious prose.
    for phrase in BAD_COURSE_TITLE_PHRASES:
        if phrase in lower_title:
            return False

    # A title containing another course code is suspicious.
    codes = COURSE_REFERENCE_RE.findall(title)

    if len(codes) > 0:
        return False

    # A title containing too many words is probably prose.
    if len(title.split()) > 25:
        return False

    # Avoid titles ending in obvious sentence-like patterns.
    if title.endswith(":"):
        return False

    return True


# ============================================================
# COURSE PARSER
# ============================================================

def extract_prerequisites(description: str) -> list[str]:
    """Extract prerequisite text from a course description."""

    if not description:
        return []

    match = PREREQUISITE_RE.search(description)

    if not match:
        return []

    prereq_text = match.group(1).strip()

    # Extract individual course codes.
    codes = COURSE_REFERENCE_RE.findall(prereq_text)

    # Deduplicate while preserving order.
    seen = set()
    result = []

    for code in codes:
        normalized = re.sub(r"\s+", " ", code).strip()

        if normalized not in seen:
            seen.add(normalized)
            result.append(normalized)

    return result


def find_course_matches(text: str) -> list[re.Match]:
    """
    Find standard course headers.

    We use finditer rather than line-by-line matching because
    PDF extraction can flatten multiple columns.
    """

    return list(COURSE_START_RE.finditer(text))


def extract_courses_from_page(
    text: str,
    page_number: int,
) -> list[dict]:
    """
    Extract all valid courses from a course-description page.
    """

    normalized = normalize_course_text(text)

    if not normalized:
        return []

    matches = find_course_matches(normalized)

    courses = []

    for index, match in enumerate(matches):

        code = match.group("code")
        title = match.group("title").strip()
        credits = match.group("credits")

        # ----------------------------------------------------
        # Validate title.
        # ----------------------------------------------------

        if not is_valid_course_title(title):
            continue

        # ----------------------------------------------------
        # Determine description boundary.
        #
        # The description ends immediately before the next
        # valid course header.
        # ----------------------------------------------------

        start = match.end()

        if index + 1 < len(matches):
            end = matches[index + 1].start()
        else:
            end = len(normalized)

        description = normalized[start:end].strip()

        # ----------------------------------------------------
        # Remove obvious trailing junk.
        # ----------------------------------------------------

        description = re.sub(
            r"\s+Course Descriptions.*$",
            "",
            description,
            flags=re.IGNORECASE,
        ).strip()

        # ----------------------------------------------------
        # Basic sanity checks.
        # ----------------------------------------------------

        if len(description) > 5000:
            description = description[:5000]

        prerequisites = extract_prerequisites(description)

        courses.append(
            {
                "type": "course",
                "course_code": code,
                "title": title,
                "format": credits,
                "description": description,
                "prerequisites": prerequisites,
                "page": page_number,
                "text": (
                    f"{code} {title} "
                    f"({credits}). "
                    f"{description}"
                ).strip(),
            }
        )

    return courses


# ============================================================
# SPECIAL BPE COURSE
# ============================================================

def extract_bpe_unnumbered(
    text: str,
    page_number: int,
) -> list[dict]:
    """Extract BPE Bridge Program Elective."""

    normalized = normalize_course_text(text)

    matches = list(BPE_UNNUMBERED_RE.finditer(normalized))

    results = []

    for index, match in enumerate(matches):

        start = match.end()

        if index + 1 < len(matches):
            end = matches[index + 1].start()
        else:
            end = len(normalized)

        description = normalized[start:end].strip()

        results.append(
            {
                "type": "course",
                "course_code": "BPE",
                "title": "Bridge Program Elective",
                "format": match.group("credits"),
                "description": description,
                "prerequisites": extract_prerequisites(description),
                "page": page_number,
                "text": (
                    f"BPE Bridge Program Elective "
                    f"({match.group('credits')}). "
                    f"{description}"
                ).strip(),
            }
        )

    return results


# ============================================================
# CALENDAR EXTRACTION
# ============================================================

def extract_calendar_records(
    text: str,
    page_number: int,
) -> list[dict]:
    """
    Extract calendar-related information.

    We intentionally keep this fairly conservative because
    calendar pages often contain tables.
    """

    records = []

    lines = [
        clean_whitespace(line)
        for line in text.splitlines()
        if clean_whitespace(line)
    ]

    for line in lines:

        lower = line.lower()

        if any(
            keyword in lower
            for keyword in [
                "semester",
                "registration",
                "classes begin",
                "classes end",
                "last day",
                "withdrawal",
                "final examination",
                "holiday",
                "break",
                "add/drop",
            ]
        ):
            records.append(
                {
                    "type": "calendar_event",
                    "page": page_number,
                    "title": "Academic Calendar",
                    "text": line,
                }
            )

    return records


# ============================================================
# PROGRAM / CURRICULUM EXTRACTION
# ============================================================

def extract_curriculum_records(
    text: str,
    page_number: int,
) -> list[dict]:
    """
    Extract course-like curriculum rows from program pages.

    This does NOT treat them as course descriptions.

    Example:
        CSE 210 Data Structures 3

    These become curriculum_course records.
    """

    records = []

    lines = [
        clean_whitespace(line)
        for line in text.splitlines()
        if clean_whitespace(line)
    ]

    for line in lines:

        match = re.match(
            r"^"
            r"(?P<code>[A-Z]{2,4}\s+\d{3}[A-Z]?(?:-\d{2})?)"
            r"\s+"
            r"(?P<rest>.+)"
            r"$",
            line,
        )

        if not match:
            continue

        code = match.group("code")
        rest = match.group("rest").strip()

        # Avoid interpreting arbitrary prose as curriculum.
        if len(rest) < 2:
            continue

        if len(rest.split()) > 20:
            continue

        # Only accept lines that look like curriculum rows.
        if not (
            re.search(r"\b\d+(?:\.\d+)?\b", rest)
            or "elective" in rest.lower()
            or "required" in rest.lower()
        ):
            continue

        records.append(
            {
                "type": "curriculum_course",
                "course_code": code,
                "page": page_number,
                "text": line,
            }
        )

    return records


# ============================================================
# GENERAL PROSE RECORDS
# ============================================================

def extract_prose_records(
    text: str,
    page_number: int,
    section: str,
) -> list[dict]:
    """
    Convert prose into paragraph-level records.

    This is intentionally conservative.
    """

    records = []

    cleaned = clean_whitespace(text)

    if not cleaned:
        return records

    paragraphs = re.split(
        r"\n\s*\n",
        cleaned,
    )

    for paragraph in paragraphs:

        paragraph = clean_whitespace(paragraph)

        if not paragraph:
            continue

        # Ignore very short fragments.
        if len(paragraph) < 40:
            continue

        # Ignore obvious headers.
        if len(paragraph) < 100 and not re.search(
            r"[.!?]",
            paragraph,
        ):
            continue

        records.append(
            {
                "type": "general_prose",
                "section": section,
                "page": page_number,
                "text": paragraph,
            }
        )

    return records


# ============================================================
# FACULTY EXTRACTION
# ============================================================

def extract_faculty_records(
    text: str,
    page_number: int,
) -> list[dict]:
    """
    Convert faculty page content into searchable records.

    We keep the original extracted text instead of aggressively
    guessing fields because faculty pages use multi-column
    layouts.
    """

    records = []

    cleaned = clean_whitespace(text)

    if not cleaned:
        return records

    # Split into reasonably sized blocks.
    blocks = re.split(
        r"\n\s*\n",
        cleaned,
    )

    for block in blocks:

        block = clean_whitespace(block)

        if len(block) < 10:
            continue

        records.append(
            {
                "type": "faculty",
                "page": page_number,
                "text": block,
            }
        )

    return records


# ============================================================
# DIRECTORY EXTRACTION
# ============================================================

def extract_directory_records(
    text: str,
    page_number: int,
) -> list[dict]:
    """Create searchable directory records."""

    records = []

    lines = [
        clean_whitespace(line)
        for line in text.splitlines()
        if clean_whitespace(line)
    ]

    for line in lines:

        if len(line) < 3:
            continue

        records.append(
            {
                "type": "directory",
                "page": page_number,
                "text": line,
            }
        )

    return records


# ============================================================
# TABLE-AWARE EXTRACTION (tables.json)
# ============================================================
#
# tables.json holds properly parsed rows/columns. Previously it
# was loaded nowhere in this script, so directory, calendar and
# curriculum content was instead scraped line-by-line from
# flattened page text -- which throws away row structure (e.g.
# a directory entry's name, phone and email become three
# unrelated records instead of one).
#
# The functions below turn tables.json rows into properly
# contextualized records instead.

DIRECTORY_HEADER_CELL_RE = re.compile(
    r"^e?partment$|^telephone$|^email$", re.IGNORECASE
)


def load_tables() -> list[dict]:
    """Load tables.json, tolerating its absence."""

    tables = load_json(TABLES_FILE, default=[])

    if not isinstance(tables, list):
        return []

    return tables


def index_tables_by_page(tables: list[dict]) -> dict[int, list[dict]]:
    """Group tables by PDF page number."""

    by_page: dict[int, list[dict]] = {}

    for table in tables:
        page = table.get("page")

        if not isinstance(page, int):
            continue

        by_page.setdefault(page, []).append(table)

    return by_page


def is_curriculum_table(table: dict) -> bool:
    return "curriculum" in (table.get("table_id") or "")


def table_row_cells(row: list) -> list[str]:
    """Clean a table row down to its non-empty cells."""

    cells = [clean_whitespace(cell or "") for cell in row]
    return [cell for cell in cells if cell]


def table_to_markdown(headers: list[str], rows: list[list[str]]) -> str:
    """Render a table as a GitHub-style markdown table."""

    cleaned_rows = [
        [clean_whitespace(cell or "") for cell in row]
        for row in rows
    ]
    cleaned_rows = [row for row in cleaned_rows if any(row)]

    if not cleaned_rows and not headers:
        return ""

    width = max(
        [len(headers)] + [len(row) for row in cleaned_rows] or [0]
    )

    header_row = (headers or cleaned_rows[0] if not headers else headers)
    header_row = list(header_row) + [""] * (width - len(header_row))

    body_rows = cleaned_rows if headers else cleaned_rows[1:]
    body_rows = [row + [""] * (width - len(row)) for row in body_rows]

    def fmt(row: list[str]) -> str:
        return "| " + " | ".join(row) + " |"

    lines = [fmt(header_row), "| " + " | ".join("---" for _ in header_row) + " |"]
    lines.extend(fmt(row) for row in body_rows)

    return "\n".join(lines)


def build_directory_records(table: dict) -> list[dict]:
    """
    One record per directory row, columns joined so the
    department/phone/email stay together.
    """

    page = table.get("page")
    records = []

    for row in table.get("rows") or []:

        cells = table_row_cells(row)

        if len(cells) < 2:
            continue

        # Drop stray header/OCR-artifact rows (e.g. "epartment").
        if DIRECTORY_HEADER_CELL_RE.match(cells[0]):
            continue

        name = cells[0]
        rest = cells[1:]
        text = f"{name} — " + ", ".join(rest)

        records.append(
            {
                "type": "directory",
                "section": "directory",
                "page": page,
                "title": name,
                "text": text,
            }
        )

    return records


def build_calendar_records(table: dict) -> list[dict]:
    """One record per calendar row, date/day/event kept together."""

    page = table.get("page")
    records = []

    for row in table.get("rows") or []:

        cells = table_row_cells(row)

        if not cells:
            continue

        if len(cells) == 1:
            text = cells[0]
        else:
            text = ", ".join(cells)

        records.append(
            {
                "type": "calendar_event",
                "section": "calendar",
                "page": page,
                "title": "Academic Calendar",
                "text": text,
            }
        )

    return records


def build_curriculum_table_record(
    table: dict,
    pages_by_number: dict[int, dict],
) -> dict | None:
    """
    One record for an entire 'Proposed Sequence of Study' table,
    so the full course sequence stays together instead of being
    exploded into disconnected lines.

    NOTE: we deliberately do NOT attach a guessed program/major
    name here. A heading-proximity heuristic was tried and tested
    against real pages -- it returned things like "FIRST YEAR" and
    "Internship" instead of the actual major, because the catalog
    has no reliable page -> major marker near these tables. Wrong
    attribution here would actively mislead the chatbot, so page
    number is kept as the only anchor for now. If you have (or can
    build) an explicit page-range -> major mapping like
    SECTION_RANGES above, that would let this be added safely.
    """

    page = table.get("page")
    headers = table.get("headers") or []
    rows = table.get("rows") or []

    markdown = table_to_markdown(headers, rows)

    if not markdown:
        return None

    title = table.get("title") or "Proposed Sequence of Study"

    return {
        "type": "curriculum_course",
        "section": "programs",
        "page": page,
        "title": title,
        "text": f"{title} (page {page}):\n{markdown}",
    }


def build_reference_table_record(table: dict, section: str) -> dict | None:
    """
    One record for a general reference table (tuition, fees,
    placement tests, etc.) that isn't a directory/calendar/
    curriculum table.
    """

    page = table.get("page")
    headers = table.get("headers") or []
    rows = table.get("rows") or []

    markdown = table_to_markdown(headers, rows)

    if not markdown:
        return None

    title = table.get("title") or f"Table on page {page}"

    return {
        "type": "table",
        "section": section,
        "page": page,
        "title": title,
        "text": f"{title} (page {page}):\n{markdown}",
    }


def build_table_records(
    tables: list[dict],
    pages_by_number: dict[int, dict],
) -> list[dict]:
    """Dispatch each table to the right record builder by section."""

    records = []

    for table in tables:

        page = table.get("page")

        if not isinstance(page, int):
            continue

        section = get_section(page)

        if section == "directory":
            records.extend(build_directory_records(table))
            continue

        if page_in_ranges(page, [(18, 19)]):
            records.extend(build_calendar_records(table))
            continue

        if is_curriculum_table(table):
            record = build_curriculum_table_record(table, pages_by_number)
            if record:
                records.append(record)
            continue

        record = build_reference_table_record(table, section)
        if record:
            records.append(record)

    return records


# ============================================================
# PAGE PROCESSING
# ============================================================

def process_page(
    page: dict,
    fallback_page_number: int,
    table_covered_kinds: set[str] | None = None,
) -> tuple[list[dict], list[dict]]:
    """
    Process one page.

    table_covered_kinds tells this function which content on this
    page is already being supplied by a properly structured table
    (via build_table_records), so the old line-based fallback for
    that kind of content should be skipped to avoid duplicating/
    fragmenting it.

    Returns:
        courses, records
    """

    table_covered_kinds = table_covered_kinds or set()

    page_number = get_page_number(
        page,
        fallback_page_number,
    )

    if page_number in EXCLUDED_PAGES:
        return [], []

    section = get_section(page_number)

    # Use coordinate-aware extraction when available.
    text = extract_columns_from_page(page)

    if not text:
        return [], []

    courses = []
    records = []

    # ========================================================
    # COURSE DESCRIPTIONS
    # ========================================================

    if section == "courses":

        courses.extend(
            extract_courses_from_page(
                text,
                page_number,
            )
        )

        courses.extend(
            extract_bpe_unnumbered(
                text,
                page_number,
            )
        )

        return courses, records

    # ========================================================
    # ACHIEVEMENT ACADEMY / COURSE DESCRIPTIONS
    # ========================================================

    if section == "achievement_courses":

        # Page 35 is mainly Achievement Academy prose.
        # Pages 36-37 contain course descriptions.
        #
        # Therefore only parse courses when the actual
        # Course Descriptions heading is present.

        if re.search(
            r"\bCourse Descriptions\b",
            text,
            flags=re.IGNORECASE,
        ):
            courses.extend(
                extract_courses_from_page(
                    text,
                    page_number,
                )
            )

            courses.extend(
                extract_bpe_unnumbered(
                    text,
                    page_number,
                )
            )

        # Still retain useful prose.
        records.extend(
            extract_prose_records(
                text,
                page_number,
                section,
            )
        )

        return courses, records

    # ========================================================
    # CALENDAR
    # ========================================================

    if page_in_ranges(page_number, [(18, 19)]):

        if "calendar" not in table_covered_kinds:
            records.extend(
                extract_calendar_records(
                    text,
                    page_number,
                )
            )

        return courses, records

    # ========================================================
    # PROGRAMS
    # ========================================================

    if section == "programs":

        if "curriculum" not in table_covered_kinds:
            records.extend(
                extract_curriculum_records(
                    text,
                    page_number,
                )
            )

        records.extend(
            extract_prose_records(
                text,
                page_number,
                section,
            )
        )

        return courses, records

    # ========================================================
    # FACULTY
    # ========================================================

    if section == "faculty":

        records.extend(
            extract_faculty_records(
                text,
                page_number,
            )
        )

        return courses, records

    # ========================================================
    # DIRECTORY
    # ========================================================

    if section == "directory":

        if "directory" not in table_covered_kinds:
            records.extend(
                extract_directory_records(
                    text,
                    page_number,
                )
            )

        return courses, records

    # ========================================================
    # EVERYTHING ELSE
    # ========================================================

    records.extend(
        extract_prose_records(
            text,
            page_number,
            section,
        )
    )

    return courses, records


# ============================================================
# DEDUPLICATION
# ============================================================

def deduplicate_courses(
    courses: list[dict],
) -> list[dict]:
    """
    Remove duplicate course entries.

    Key:
        course code + title + page
    """

    seen = set()
    result = []

    for course in courses:

        key = (
            course.get("course_code", "").strip(),
            course.get("title", "").strip().lower(),
            course.get("page"),
        )

        if key in seen:
            continue

        seen.add(key)
        result.append(course)

    return result


def deduplicate_records(
    records: list[dict],
) -> list[dict]:
    """Remove duplicate records."""

    seen = set()
    result = []

    for record in records:

        key = (
            record.get("type"),
            record.get("page"),
            record.get("text", "").strip().lower(),
        )

        if key in seen:
            continue

        seen.add(key)
        result.append(record)

    return result


# ============================================================
# COURSE TEXT CLEANUP
# ============================================================

def clean_course_description(
    description: str,
) -> str:
    """Clean common PDF extraction artifacts."""

    description = clean_whitespace(description)

    # Remove repeated spaces.
    description = re.sub(
        r"\s+",
        " ",
        description,
    )

    # Remove spaces before punctuation.
    description = re.sub(
        r"\s+([,.!?;:])",
        r"\1",
        description,
    )

    return description.strip()


def finalize_courses(
    courses: list[dict],
) -> list[dict]:

    final = []

    for course in courses:

        description = clean_course_description(
            course.get("description", "")
        )

        course["description"] = description

        course["text"] = (
            f"{course['course_code']} "
            f"{course['title']} "
            f"({course['format']}). "
            f"{description}"
        ).strip()

        final.append(course)

    return final


# ============================================================
# UNIFIED DOCUMENTS
# ============================================================

def create_documents(
    courses: list[dict],
    records: list[dict],
) -> list[dict]:
    """
    Convert all extracted material into one unified format.

    This is the file that the retrieval pipeline will eventually
    use.
    """

    documents = []

    # --------------------------------------------------------
    # Courses
    # --------------------------------------------------------

    for index, course in enumerate(courses):

        documents.append(
            {
                "id": f"course_{index:05d}",
                "type": "course",
                "section": "course_descriptions",
                "page": course["page"],
                "course_code": course["course_code"],
                "title": course["title"],
                "metadata": {
                    "course_code": course["course_code"],
                    "title": course["title"],
                    "format": course["format"],
                    "page": course["page"],
                    "prerequisites": course.get(
                        "prerequisites",
                        [],
                    ),
                },
                "text": course["text"],
            }
        )

    # --------------------------------------------------------
    # Other records
    # --------------------------------------------------------

    for index, record in enumerate(records):

        document_id = f"record_{index:05d}"

        documents.append(
            {
                "id": document_id,
                "type": record.get(
                    "type",
                    "general_prose",
                ),
                "section": record.get(
                    "section",
                    "unknown",
                ),
                "page": record.get("page"),
                "metadata": {
                    key: value
                    for key, value in record.items()
                    if key not in {
                        "text",
                    }
                },
                "text": record.get(
                    "text",
                    "",
                ),
            }
        )

    return documents


# ============================================================
# CHUNKING
# ============================================================

def chunk_text(
    text: str,
    max_chars: int = MAX_POLICY_CHARS,
    overlap: int = POLICY_OVERLAP_CHARS,
) -> list[str]:
    """
    Generic text chunking.

    Course descriptions are already individual documents, so
    they normally won't need to be split unless unusually large.
    """

    text = clean_whitespace(text)

    if not text:
        return []

    if len(text) <= max_chars:
        return [text]

    chunks = []

    start = 0

    while start < len(text):

        end = min(
            start + max_chars,
            len(text),
        )

        # Try to end at a sentence.
        if end < len(text):

            sentence_end = max(
                text.rfind(". ", start, end),
                text.rfind("? ", start, end),
                text.rfind("! ", start, end),
            )

            if sentence_end > start + 300:
                end = sentence_end + 1

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        start = max(
            end - overlap,
            start + 1,
        )

    return chunks


def create_chunks(
    documents: list[dict],
) -> list[dict]:
    """Create chunks while preserving metadata."""

    chunks = []

    for document in documents:

        text = document.get("text", "")

        document_chunks = chunk_text(
            text,
            max_chars=MAX_POLICY_CHARS,
            overlap=POLICY_OVERLAP_CHARS,
        )

        for chunk_index, chunk in enumerate(
            document_chunks
        ):

            chunks.append(
                {
                    "id": (
                        f"{document['id']}"
                        f"_chunk_{chunk_index:03d}"
                    ),
                    "document_id": document["id"],
                    "type": document["type"],
                    "section": document["section"],
                    "page": document.get("page"),
                    "course_code": document.get(
                        "course_code"
                    ),
                    "title": document.get(
                        "title"
                    ),
                    "metadata": document.get(
                        "metadata",
                        {},
                    ),
                    "text": chunk,
                }
            )

    return chunks


# ============================================================
# VALIDATION
# ============================================================

def print_course_validation(
    courses: list[dict],
) -> None:
    """Print useful course extraction diagnostics."""

    print()
    print("=" * 70)
    print("COURSE EXTRACTION VALIDATION")
    print("=" * 70)

    print(
        f"Total courses extracted: {len(courses)}"
    )

    if not courses:
        print("WARNING: No courses were extracted.")
        return

    print()
    print("First 15 courses:")

    for course in courses[:15]:

        print(
            f"  {course['course_code']} | "
            f"{course['title']} | "
            f"page {course['page']}"
        )

    print()
    print("Last 15 courses:")

    for course in courses[-15:]:

        print(
            f"  {course['course_code']} | "
            f"{course['title']} | "
            f"page {course['page']}"
        )

    # --------------------------------------------------------
    # Specific sanity checks.
    # --------------------------------------------------------

    important_codes = [
        "MTH 103",
        "MTH 221",
        "MTH 225",
        "MTH 243",
        "CMP 220",
        "COE 486",
    ]

    print()
    print("Specific course checks:")

    course_map = {}

    for course in courses:
        course_map.setdefault(
            course["course_code"],
            [],
        ).append(course)

    for code in important_codes:

        matches = course_map.get(
            code,
            [],
        )

        if matches:

            for course in matches:

                print(
                    f"  ✓ {code}: "
                    f"{course['title']} "
                    f"(page {course['page']})"
                )

        else:

            print(
                f"  ✗ {code}: NOT FOUND"
            )

    # --------------------------------------------------------
    # Suspicious titles.
    # --------------------------------------------------------

    suspicious = []

    for course in courses:

        title = course["title"]

        if not is_valid_course_title(title):
            suspicious.append(course)

    print()
    print(
        f"Suspicious course titles: "
        f"{len(suspicious)}"
    )

    for course in suspicious[:10]:

        print(
            f"  ? {course['course_code']} | "
            f"{course['title']} | "
            f"page {course['page']}"
        )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print("=" * 70)
    print("AUS CATALOG PREPROCESSING")
    print("=" * 70)

    # --------------------------------------------------------
    # Check input.
    # --------------------------------------------------------

    if not PAGES_FILE.exists():

        raise FileNotFoundError(
            f"Could not find {PAGES_FILE}\n"
            "Run extract_pdf.py first."
        )

    pages = load_json(
        PAGES_FILE,
        default=[],
    )

    if not isinstance(pages, list):

        raise ValueError(
            "pages.json must contain a list of pages."
        )

    print(
        f"Loaded {len(pages)} pages"
    )

    # --------------------------------------------------------
    # Load tables (previously unused).
    # --------------------------------------------------------

    tables = load_tables()

    print(
        f"Loaded {len(tables)} tables"
    )

    pages_by_number: dict[int, dict] = {}

    for index, page in enumerate(pages, start=1):
        page_number = get_page_number(page, index)
        pages_by_number[page_number] = page

    tables_by_page = index_tables_by_page(tables)

    # For each page, note which kinds of content (directory,
    # calendar, curriculum) are already covered by a proper table,
    # so process_page() can skip the old line-based fallback there.
    table_covered_by_page: dict[int, set[str]] = {}

    for page_number, page_tables in tables_by_page.items():
        section = get_section(page_number)
        kinds: set[str] = set()

        for table in page_tables:
            if section == "directory":
                # The directory table is a full, clean row/column
                # extraction that covers this page's content
                # entirely -- the old line-by-line version only
                # produced disconnected single-cell fragments, so
                # it's fully replaced (not additive).
                kinds.add("directory")
            # NOTE: calendar and curriculum tables are NOT marked
            # as "covered" here. Checked against the real data:
            # the calendar table only captures 3 commencement rows
            # while the page also contains ~30 other calendar
            # lines (registration deadlines, holidays, etc.) with
            # no table at all, and curriculum pages often have
            # extra requirement notes outside the sequence table.
            # So for these two, the table record is added
            # alongside the line-based extraction rather than
            # replacing it. Deduplication removes exact repeats.

        if kinds:
            table_covered_by_page[page_number] = kinds

    table_records = build_table_records(tables, pages_by_number)

    print(
        f"Built {len(table_records)} records from tables.json"
    )

    # --------------------------------------------------------
    # Show section distribution.
    # --------------------------------------------------------

    section_counts = {}

    for index, page in enumerate(pages, start=1):

        page_number = get_page_number(
            page,
            index,
        )

        if page_number in EXCLUDED_PAGES:
            section = "exclude"
        else:
            section = get_section(
                page_number
            )

        section_counts[section] = (
            section_counts.get(section, 0)
            + 1
        )

    print()
    print("Pages by section:")

    for section, count in sorted(
        section_counts.items()
    ):

        print(
            f"  {section:<25} {count}"
        )

    # --------------------------------------------------------
    # Process pages.
    # --------------------------------------------------------

    all_courses = []
    all_records = []

    for index, page in enumerate(
        pages,
        start=1,
    ):

        page_number = get_page_number(
            page,
            index,
        )

        courses, records = process_page(
            page,
            page_number,
            table_covered_by_page.get(page_number),
        )

        all_courses.extend(courses)
        all_records.extend(records)

    all_records.extend(table_records)

    # --------------------------------------------------------
    # Deduplicate.
    # --------------------------------------------------------

    all_courses = deduplicate_courses(
        all_courses
    )

    all_courses = finalize_courses(
        all_courses
    )

    all_records = deduplicate_records(
        all_records
    )

    # --------------------------------------------------------
    # Unified documents.
    # --------------------------------------------------------

    documents = create_documents(
        all_courses,
        all_records,
    )

    # --------------------------------------------------------
    # Chunks.
    # --------------------------------------------------------

    chunks = create_chunks(
        documents
    )

    # --------------------------------------------------------
    # Save.
    # --------------------------------------------------------

    save_json(
        COURSES_FILE,
        all_courses,
    )

    save_json(
        RECORDS_FILE,
        all_records,
    )

    save_json(
        DOCUMENTS_FILE,
        documents,
    )

    save_json(
        CHUNKS_FILE,
        chunks,
    )

    # --------------------------------------------------------
    # Summary.
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("PREPROCESSING COMPLETE")
    print("=" * 70)

    print(
        f"Courses extracted      : "
        f"{len(all_courses)}"
    )

    print(
        f"Structured records     : "
        f"{len(all_records)}"
    )

    print(
        f"Total documents        : "
        f"{len(documents)}"
    )

    print(
        f"Total chunks           : "
        f"{len(chunks)}"
    )

    print()
    print("Output files:")

    print(
        f"  {COURSES_FILE}"
    )

    print(
        f"  {RECORDS_FILE}"
    )

    print(
        f"  {DOCUMENTS_FILE}"
    )

    print(
        f"  {CHUNKS_FILE}"
    )

    # --------------------------------------------------------
    # Validation.
    # --------------------------------------------------------

    print_course_validation(
        all_courses
    )

    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()