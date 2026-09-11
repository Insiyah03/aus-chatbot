import pdfplumber
import json
import re
from pathlib import Path
from src.config import (PDF_PATH, TABLES_PATH)


OUTPUT_PATH = TABLES_PATH


# ============================================================
# BASIC CLEANING
# ============================================================

def clean_cell(cell):
    if cell is None:
        return ""

    text = str(cell)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def clean_text(text):
    if not text:
        return ""

    text = re.sub(r"\s+", " ", str(text)).strip()

    return text


# ============================================================
# GENERAL TABLE HELPERS
# ============================================================

def get_nonempty_cells(row):
    """
    Return non-empty cells while preserving their left-to-right order.
    Useful because pdfplumber often creates artificial empty columns.
    """
    return [
        clean_cell(cell)
        for cell in row
        if clean_cell(cell)
    ]


def table_has_content(table):
    """
    Check whether a pdfplumber table contains meaningful content.
    """
    try:
        rows = table.extract()
    except Exception:
        return False

    if not rows:
        return False

    nonempty = 0

    for row in rows:
        for cell in row:
            if clean_cell(cell):
                nonempty += 1

    return nonempty >= 2


def is_one_column_prose(table):
    """
    Reject tables that are really ordinary page text
    accidentally detected as a table.
    """
    try:
        rows = table.extract()
    except Exception:
        return True

    if not rows:
        return True

    meaningful_rows = [
        get_nonempty_cells(row)
        for row in rows
        if get_nonempty_cells(row)
    ]

    if not meaningful_rows:
        return True

    # If every row contains only one cell, this is usually
    # a false-positive prose block.
    if all(len(row) == 1 for row in meaningful_rows):
        return True

    return False


# ============================================================
# CURRICULUM DETECTION
# ============================================================

YEAR_PATTERN = re.compile(
    r"\b(FIRST|SECOND|THIRD|FOURTH|FIFTH)\s+YEAR\s*"
    r"\((\d+)\s+credit\s+hours?\)",
    re.IGNORECASE
)


def is_curriculum_page(page):
    """
    Detect pages containing curriculum / proposed sequence tables.

    Examples:
        FIRST YEAR (30 credit hours)
        SECOND YEAR (36 credit hours)
        THIRD YEAR (33 credit hours)
    """
    text = page.extract_text() or ""

    if not text:
        return False

    has_year = bool(YEAR_PATTERN.search(text))

    # Curriculum tables normally contain this header.
    has_course_header = (
        "Course #" in text
        or "Course #" in text.replace("\n", " ")
    )

    return has_year and has_course_header


# ============================================================
# WORD / LINE RECONSTRUCTION
# ============================================================

def group_words_into_lines(words, tolerance=3):
    """
    Group PDF words into visual lines using their vertical position.

    Returns:
        [
            {
                "top": ...,
                "words": [...]
            },
            ...
        ]
    """

    if not words:
        return []

    sorted_words = sorted(
        words,
        key=lambda w: (w["top"], w["x0"])
    )

    lines = []

    for word in sorted_words:

        placed = False

        for line in lines:

            if abs(word["top"] - line["top"]) <= tolerance:
                line["words"].append(word)

                # Keep the average-ish top position stable.
                line["top"] = min(
                    line["top"],
                    word["top"]
                )

                placed = True
                break

        if not placed:
            lines.append({
                "top": word["top"],
                "words": [word]
            })

    for line in lines:
        line["words"].sort(key=lambda w: w["x0"])

    lines.sort(key=lambda line: line["top"])

    return lines


def line_text(words):
    return clean_text(
        " ".join(word["text"] for word in words)
    )


# ============================================================
# COURSE CODE CLEANING
# ============================================================

def normalize_course_code(text):
    """
    Clean common PDF extraction artifacts in course codes.

    Examples:
        GER-Cor e -> GER-Core
        GER-CO M  -> GER-COM
    """

    text = clean_text(text)

    # Common split forms caused by PDF column boundaries.
    text = re.sub(
        r"\b(GER-[A-Za-z]+)\s+([A-Za-z])\b",
        lambda m: m.group(1) + m.group(2),
        text,
    )

    return text


def looks_like_course_code(text):
    """
    Determine whether a string looks like a course identifier.

    Supports:
        ARC 201
        MTH 100
        GER-Core
        GER-COM
        GER-SCI
        FRE
        MJE
        ENG 203 or ENG 204
        ENG 203
    """

    text = clean_text(text)

    if not text:
        return False

    patterns = [
        # Standard department + number
        r"^[A-Z]{2,5}\s+\d{3}[A-Z]?$",

        # Alternatives between numbered courses.
        r"^[A-Z]{2,5}\s+\d{3}[A-Z]?\s+or\s+[A-Z]{2,5}\s+\d{3}[A-Z]?$",

        # Alternatives between a numbered course and a curriculum
        # placeholder, e.g. ARC 591 or FRE.
        r"^[A-Z]{2,5}\s+\d{3}[A-Z]?\s+or\s+(?:FRE|MJE|MCE)$",

        # General education / elective identifiers
        r"^GER-[A-Za-z]+$",
        r"^FRE$",
        r"^MJE$",
        r"^MCE$",

        # Other named curriculum placeholders
        r"^[A-Z]{2,5}-[A-Za-z]+$",
    ]

    return any(
        re.match(pattern, text, re.IGNORECASE)
        for pattern in patterns
    )


def looks_like_course_start(words):
    """
    Check whether a line appears to begin a new course row.

    We primarily inspect the first few words because course numbers
    occupy a predictable column in the PDF.
    """

    if not words:
        return False

    text = line_text(words[:6])

    text = normalize_course_code(text)

    # Standard code: ARC 201
    if re.search(
        r"\b[A-Z]{2,5}\s+\d{3}[A-Z]?\b",
        text
    ):
        return True

    # General curriculum identifiers.
    if re.search(
        r"\bGER-[A-Za-z]+\b|\bFRE\b|\bMJE\b|\bMCE\b",
        text
    ):
        return True

    return False


# ============================================================
# CURRICULUM HEADER DETECTION
# ============================================================

def is_year_heading(text):
    """
    Return match information for:
        THIRD YEAR (33 credit hours)
    """

    match = YEAR_PATTERN.search(text)

    if not match:
        return None

    return {
        "year": match.group(1).title() + " Year",
        "credit_hours": int(match.group(2)),
    }


def is_curriculum_header(text):
    """
    Detect:
        Term Course # Course Title Credit Hours
    """

    normalized = clean_text(text).lower()

    return (
        "term" in normalized
        and "course" in normalized
        and "title" in normalized
        and "credit" in normalized
    )


# ============================================================
# COLUMN INFERENCE
# ============================================================


def infer_curriculum_columns(header_words, block_left, block_right):
    """
    Infer logical curriculum column boundaries from header positions.

    The catalog headers are aligned with the start of each logical
    column. We therefore use those START positions as boundaries rather
    than assigning words to the nearest header.
    """
    words = sorted(header_words, key=lambda w: w["x0"])

    term_positions = []
    course_positions = []
    title_positions = []
    credit_positions = []

    for word in words:
        value = clean_text(word["text"]).lower()

        if value == "term":
            term_positions.append(word["x0"])
        elif value == "course":
            course_positions.append(word["x0"])
        elif value == "title":
            title_positions.append(word["x0"])
        elif value == "credit":
            credit_positions.append(word["x0"])

    # There are two "Course" headers: Course # and Course Title.
    # The second one marks the beginning of the title column.
    course_start = (
        course_positions[0]
        if course_positions
        else block_left + 35
    )

    title_start = (
        course_positions[1]
        if len(course_positions) >= 2
        else (
            title_positions[0]
            if title_positions
            else block_left + 75
        )
    )

    if credit_positions:
        credit_start = credit_positions[0] - 5
    else:
        # Some pages extract only "Hours" rather than "Credit".
        hour_positions = [
            word["x0"]
            for word in words
            if clean_text(word["text"]).lower() == "hours"
        ]
        credit_start = (
            hour_positions[0] - 5
            if hour_positions
            else block_right - 25
        )

    return {
        "term_start": (
            term_positions[0]
            if term_positions
            else block_left + 4
        ),
        "course_start": course_start,
        "title_start": title_start,
        "credit_start": credit_start,
        "left": block_left,
        "right": block_right,
    }


def assign_word_to_column(word, columns):
    """Assign a word to a curriculum column using column start positions."""
    x = word["x0"]

    if x < columns["course_start"]:
        return "term"
    if x < columns["title_start"]:
        return "course"
    if x < columns["credit_start"]:
        return "title"
    return "credit"


def words_to_curriculum_row(words, columns):
    """Convert one logical table row into [term, course, title, credits]."""
    result = {"term": [], "course": [], "title": [], "credit": []}

    for word in sorted(words, key=lambda w: (w.get("top", 0), w["x0"])):
        result[assign_word_to_column(word, columns)].append(word["text"])

    return clean_curriculum_row([
        clean_text(" ".join(result["term"])),
        normalize_course_code(clean_text(" ".join(result["course"]))),
        clean_text(" ".join(result["title"])),
        clean_text(" ".join(result["credit"])),
    ])


def get_panel_bounds(page, block):
    """Return the visual left/right panel containing a curriculum year."""
    midpoint = page.width / 2

    if block["x0"] < midpoint:
        return 0, midpoint

    return midpoint, page.width


def get_panel_horizontal_rules(page, panel_left, panel_right, min_width=100):
    """
    Find horizontal table rules for one curriculum panel.

    The catalog represents these rules as very thin PDF rectangles.
    They are used to reconstruct logical rows, including rows whose
    course number or title wraps onto multiple text lines.
    """
    rules = []

    for rect in page.rects:
        x0 = rect.get("x0", 0)
        x1 = rect.get("x1", 0)
        top = rect.get("top", 0)
        height = rect.get("height", 0)
        width = x1 - x0

        if height > 1.0 or width < min_width:
            continue

        overlap = min(x1, panel_right) - max(x0, panel_left)
        if overlap < min_width * 0.8:
            continue

        rules.append(top)

    rules.sort()

    unique = []
    for y in rules:
        if not unique or abs(y - unique[-1]) > 1.0:
            unique.append(y)

    return unique


def find_curriculum_header(lines, start_index, end_index, panel_left, panel_right):
    """
    Find the curriculum header and return its line index and header words.

    'Credit' and 'Hours' are often on separate visual lines. We collect
    only those header lines, stopping before the first data row.
    """
    for i in range(start_index, end_index):
        panel_words = [
            word for word in lines[i]["words"]
            if panel_left <= word["x0"] < panel_right
        ]

        lower = line_text(panel_words).lower()

        if "term" in lower and "course" in lower:
            header_words = []

            for j in range(i, end_index):
                words_here = [
                    word for word in lines[j]["words"]
                    if panel_left <= word["x0"] < panel_right
                ]

                text_here = line_text(words_here).lower()
                header_words.extend(words_here)

                # The header is complete once we see "hours".
                if "hours" in text_here:
                    break

                # Safety stop: do not absorb the first data row.
                if j > i + 2:
                    break

            return i, header_words

    return None, []


def extract_curriculum_rows_from_rules(
    page,
    lines,
    header_index,
    block_end_top,
    panel_left,
    panel_right,
    columns,
):
    """
    Reconstruct logical rows using the table's horizontal rules.

    This fixes cases such as:
        MTH 111 or
        MTH 103

    which are one logical row in the PDF but two visual text lines.
    """
    # Find the end of the header itself. Do not include the first data
    # row when determining header_bottom.
    header_bottom = lines[header_index]["top"]

    for i in range(header_index, min(len(lines), header_index + 4)):
        words_here = [
            word for word in lines[i]["words"]
            if panel_left <= word["x0"] < panel_right
        ]
        if "hours" in line_text(words_here).lower():
            header_bottom = lines[i]["top"]
            break

    rules = [
        y for y in get_panel_horizontal_rules(page, panel_left, panel_right)
        if header_bottom < y < block_end_top
    ]

    if len(rules) < 2:
        return []

    raw_rows = []

    for top, bottom in zip(rules, rules[1:]):
        row_words = []

        # Use individual word coordinates rather than the page-wide
        # visual-line top. Adjacent left/right curriculum tables can
        # have text on nearly identical y-coordinates, and page-wide
        # line grouping can otherwise shift a row into the wrong band.
        for line in lines:
            for word in line["words"]:
                if not (panel_left <= word["x0"] < panel_right):
                    continue

                if word["top"] <= top + 0.5:
                    continue
                if word["top"] >= bottom - 0.5:
                    continue

                row_words.append(word)

        if row_words:
            row = words_to_curriculum_row(row_words, columns)
            if any(row):
                raw_rows.append(row)

    return raw_rows


# ============================================================
# CURRICULUM ROW CLEANING
# ============================================================

def clean_curriculum_row(row):
    """
    Clean and repair one curriculum row.
    """

    term, course_number, title, credit_hours = row

    term = clean_text(term)
    course_number = normalize_course_code(course_number)
    title = clean_text(title)
    credit_hours = clean_text(credit_hours)

    # Occasionally PDF extraction places the term into
    # the course column.
    known_terms = {"Fall", "Spring", "Summer"}

    if not term and course_number in known_terms:
        term = course_number
        course_number = ""

    # Normalize common split GER identifiers.
    course_number = re.sub(
        r"\bGER-Cor\s+e\b",
        "GER-Core",
        course_number,
        flags=re.IGNORECASE,
    )

    course_number = re.sub(
        r"\bGER-CO\s+M\b",
        "GER-COM",
        course_number,
        flags=re.IGNORECASE,
    )

    return [
        term,
        course_number,
        title,
        credit_hours,
    ]


def is_total_row(row):
    combined = " ".join(row).lower()

    return (
        "total" in combined
        and not re.search(
            r"\b[A-Z]{2,5}\s+\d{3}\b",
            combined,
            re.IGNORECASE
        )
    )


def normalize_total_row(row):
    """
    Convert things such as:

        ['', '', 'Total', '15']

    into:

        ['', '', 'Total', '15']
    """

    term, course, title, credits = row

    combined = " ".join(
        x for x in [term, course, title]
        if x
    )

    if "total" in combined.lower():

        # Extract the number from any field.
        number_match = re.search(
            r"\b(\d+(?:\.\d+)?)\b",
            " ".join(row)
        )

        number = (
            number_match.group(1)
            if number_match
            else credits
        )

        return ["", "", "Total", number]

    return row


# ============================================================
# CURRICULUM BLOCK EXTRACTION
# ============================================================


def extract_curriculum_blocks(page):
    """
    Reconstruct curriculum sections directly from PDF words and
    horizontal table rules.

    Each year becomes its own logical table.

    LEFT and RIGHT panels are processed independently. This is critical
    on pages where FIRST/SECOND YEAR are on the left while
    THIRD/FOURTH/FIFTH YEAR are on the right.
    """
    words = page.extract_words(
        x_tolerance=2,
        y_tolerance=3,
        keep_blank_chars=False,
        use_text_flow=False,
    )

    if not words:
        return []

    lines = group_words_into_lines(words)
    midpoint = page.width / 2

    year_blocks = []

    for index, line in enumerate(lines):
        words_in_line = sorted(line["words"], key=lambda w: w["x0"])

        # A single visual line can contain two side-by-side headings
        # (e.g. FIRST YEAR on the left and THIRD YEAR on the right).
        for i in range(len(words_in_line) - 4):
            candidate_words = words_in_line[i:i + 5]
            candidate = " ".join(w["text"] for w in candidate_words)

            match = YEAR_PATTERN.fullmatch(candidate.strip())
            if not match:
                continue

            x0 = min(w["x0"] for w in candidate_words)
            x1 = max(w["x1"] for w in candidate_words)
            center = (x0 + x1) / 2

            year_blocks.append({
                "line_index": index,
                "top": line["top"],
                "x0": x0,
                "x1": x1,
                "panel": "left" if center < midpoint else "right",
                "year": match.group(1).title() + " Year",
                "credit_hours": int(match.group(2)),
            })

    if not year_blocks:
        return []

    panel_blocks = {
        "left": sorted(
            [b for b in year_blocks if b["panel"] == "left"],
            key=lambda b: b["top"],
        ),
        "right": sorted(
            [b for b in year_blocks if b["panel"] == "right"],
            key=lambda b: b["top"],
        ),
    }

    results = []

    for panel_name, blocks in panel_blocks.items():
        for block in blocks:
            start_line_index = block["line_index"]

            # CRITICAL: end at the next year heading in the SAME panel.
            if block is not blocks[-1]:
                block_position = blocks.index(block)
                next_block = blocks[block_position + 1]
                end_line_index = next_block["line_index"]
                block_end_top = next_block["top"]
            else:
                end_line_index = len(lines)
                block_end_top = page.height

            panel_left, panel_right = get_panel_bounds(page, block)

            header_index, header_words = find_curriculum_header(
                lines,
                start_line_index + 1,
                end_line_index,
                panel_left,
                panel_right,
            )

            if header_index is None:
                continue

            columns = infer_curriculum_columns(
                header_words,
                panel_left,
                panel_right,
            )

            raw_rows = extract_curriculum_rows_from_rules(
                page,
                lines,
                header_index,
                block_end_top,
                panel_left,
                panel_right,
                columns,
            )

            if not raw_rows:
                continue

            cleaned_rows = []
            current_term = ""

            for row in raw_rows:
                row = normalize_total_row(row)

                term, course, title, credits = row

                # Ignore ordinary prose accidentally picked up by later
                # horizontal rules on the same page.
                if (
                    course
                    and not looks_like_course_code(course)
                    and not is_total_row(row)
                ):
                    continue

                if term in {"Fall", "Spring", "Summer"}:
                    current_term = term
                elif not term:
                    term = current_term

                row = [term, course, title, credits]

                if not any(row):
                    continue

                if is_year_heading(" ".join(x for x in row if x)):
                    continue

                cleaned_rows.append(row)

            if not cleaned_rows:
                continue

            results.append({
                "table_id": f"page_{page.page_number}_curriculum_{len(results) + 1}",
                "table_index": len(results),
                "page": page.page_number,
                "title": "Proposed Sequence of Study",
                "type": "curriculum",
                "year": block["year"],
                "year_credit_hours": block["credit_hours"],
                "headers": [
                    "Term",
                    "Course #",
                    "Course Title",
                    "Credit Hours",
                ],
                "rows": cleaned_rows,
                "notes": [],
                "text": build_curriculum_text(
                    page.page_number,
                    block["year"],
                    cleaned_rows,
                ),
                "_sort_top": block["top"],
                "_sort_x": block["x0"],
            })

    # Curriculum years are more useful downstream in logical academic
    # order than in physical page order (the PDF uses two columns).
    year_order = {
        "First Year": 1,
        "Second Year": 2,
        "Third Year": 3,
        "Fourth Year": 4,
        "Fifth Year": 5,
    }

    results.sort(
        key=lambda item: (
            year_order.get(item["year"], 99),
            item["_sort_top"],
            item["_sort_x"],
        )
    )

    for index, item in enumerate(results):
        item["table_index"] = index
        item.pop("_sort_top", None)
        item.pop("_sort_x", None)
    return results


# ============================================================
# CURRICULUM TEXT
# ============================================================

def build_curriculum_text(
    page_number,
    year,
    rows,
):
    """
    Build a clean text representation specifically for RAG.
    """

    parts = [
        f"Page: {page_number}",
        "Proposed Sequence of Study",
        year,
    ]

    for row in rows:

        term, course, title, credits = row

        values = []

        if term:
            values.append(term)

        if course:
            values.append(course)

        if title:
            values.append(title)

        if credits:
            values.append(credits)

        if values:
            parts.append(" | ".join(values))

    return "\n".join(parts)


# ============================================================
# NORMAL TABLE TITLE DETECTION
# ============================================================

def detect_title(rows):
    """
    Detect titles only from known table structures.

    We deliberately DO NOT grab arbitrary text above a table,
    because that caused incorrect titles such as:

        "Placement Tests table hereafter. For"

    """

    flattened = []

    for row in rows:
        flattened.extend(
            get_nonempty_cells(row)
        )

    text = " ".join(flattened)

    # Placement Tests
    if "Required Placement Tests" in text:
        return "Required Placement Tests"

    # Tuition
    if "Tuition (in AED)" in text:
        return "Tuition (in AED)"

    # Compulsory Fees
    if "Compulsory Fees (in AED)" in text:
        return "Compulsory Fees (in AED)"

    return ""


# ============================================================
# PLACEMENT TEST TABLE
# ============================================================

def normalize_placement_tests(rows):
    """
    Normalize the Required Placement Tests table.
    """

    full_text = " ".join(
        " ".join(get_nonempty_cells(row))
        for row in rows
    )

    if "Required Placement Tests" not in full_text:
        return None

    majors = [
        "Architecture/Interior Design",
        "Biology",
        "Business Administration (all majors)",
        "Chemistry and Biochemistry",
        "Computer Science",
        "Data Science",
        "Design Management",
        "Engineering majors",
        "English Language and Literature",
        "Environmental Sciences and Sustainability",
        "Film and New Media Design/Graphic Design",
        "International Studies/Psychology",
        "Media Communication",
        "Mathematics",
        "Physics",
        "Psychology",
        "Undeclared Major",
    ]

    data_rows = []

    for row in rows:

        cells = get_nonempty_cells(row)

        if not cells:
            continue

        row_text = " ".join(cells)

        matched_major = None

        for major in majors:
            if major.lower() in row_text.lower():
                matched_major = major
                break

        if not matched_major:
            continue

        # Remove major name from beginning.
        remaining = row_text[
            len(matched_major):
        ].strip()

        values = re.findall(
            r"\b(?:Yes|No|\*|\*\*)\b",
            remaining,
            flags=re.IGNORECASE,
        )

        # More permissive fallback.
        if len(values) < 5:
            values = re.findall(
                r"Yes|No|\*{1,2}",
                remaining,
                flags=re.IGNORECASE,
            )

        if len(values) >= 5:
            data_rows.append([
                matched_major,
                *values[:5],
            ])

    if not data_rows:
        return None

    return {
        "headers": [
            "Majors",
            "Engineering Math",
            "Business Math",
            "Architecture Math",
            "Physics",
            "English",
        ],
        "rows": data_rows,
    }


# ============================================================
# GRADING TABLE
# ============================================================

def normalize_grading_table(rows):
    """
    Normalize the grading tables on page 60.

    Handles PDF grids where grade letters and grade points
    are separated into artificial columns.
    """

    flattened = [
        get_nonempty_cells(row)
        for row in rows
    ]

    text = " ".join(
        " ".join(row)
        for row in flattened
    ).lower()

    grade_words = [
        "excellent",
        "good",
        "satisfactory",
        "poor",
        "fail",
        "academic integrity violation fail",
    ]

    if not any(
        grade in text
        for grade in grade_words
    ):
        return None

    result = []

    current_category = None

    for row in flattened:

        row_text = " ".join(row)

        lower = row_text.lower()

        matched_category = None

        for category in grade_words:
            if category in lower:
                matched_category = category.title()
                break

        if matched_category:
            current_category = matched_category

        # Grade + grade point.
        grade_match = re.search(
            r"\(?([A-F][+-]?)\)?\s*"
            r"(?:equals|=)?\s*"
            r"(\d+(?:\.\d+)?)",
            row_text,
            re.IGNORECASE,
        )

        if grade_match and current_category:

            result.append([
                current_category,
                grade_match.group(1).upper(),
                grade_match.group(2),
            ])

    if not result:
        return None

    return {
        "headers": [
            "Category",
            "Grade",
            "Grade Points",
        ],
        "rows": result,
    }


# ============================================================
# GRADE NOTATION TABLE
# ============================================================

def normalize_grade_notation(rows):
    """
    Normalize AUD, I, IP, N, P, NP, TR, W, WV table.
    """

    codes = {
        "AUD",
        "I",
        "IP",
        "N",
        "P",
        "NP",
        "TR",
        "W",
        "WV",
    }

    result = []

    for row in rows:

        cells = get_nonempty_cells(row)

        if not cells:
            continue

        code = cells[0]

        if code in codes and len(cells) >= 2:
            result.append([
                code,
                " ".join(cells[1:]),
            ])

    if not result:
        return None

    return {
        "headers": [
            "Notation",
            "Meaning",
        ],
        "rows": result,
    }


# ============================================================
# TUITION TABLE
# ============================================================

def normalize_tuition(rows):
    """
    Normalize Tuition (in AED).
    """

    text = " ".join(
        " ".join(get_nonempty_cells(row))
        for row in rows
    )

    if "Tuition (in AED)" not in text:
        return None

    result = []

    for row in rows:

        cells = get_nonempty_cells(row)

        if not cells:
            continue

        row_text = " ".join(cells)

        # Skip title/header.
        if (
            "Tuition (in AED)" in row_text
            or row_text == "Regular Semester"
            or row_text == "Summer Term"
        ):
            continue

        # Preserve meaningful tuition rows.
        result.append(cells)

    return {
        "headers": [
            "Item",
            "Regular Semester",
            "Summer Term",
        ],
        "rows": result,
    }


# ============================================================
# COMPULSORY FEES
# ============================================================

def normalize_compulsory_fees(rows):
    """
    Normalize Compulsory Fees (in AED).
    """

    text = " ".join(
        " ".join(get_nonempty_cells(row))
        for row in rows
    )

    if "Compulsory Fees (in AED)" not in text:
        return None

    result = []

    for row in rows:

        cells = get_nonempty_cells(row)

        if not cells:
            continue

        row_text = " ".join(cells)

        if (
            "Compulsory Fees (in AED)" in row_text
            or row_text in {
                "Fee Type",
                "Description",
                "Regular Semester",
                "Summer Term",
            }
        ):
            continue

        result.append(cells)

    return {
        "headers": [
            "Fee Type",
            "Description",
            "Regular Semester",
            "Summer Term",
        ],
        "rows": result,
    }


# ============================================================
# GENERIC TABLE NORMALIZATION
# ============================================================

def normalize_generic_table(rows):
    """
    Generic fallback.

    Removes artificial empty PDF columns while preserving
    the original left-to-right order.
    """

    cleaned = []

    for row in rows:

        cells = get_nonempty_cells(row)

        if not cells:
            continue

        cleaned.append(cells)

    return cleaned


# ============================================================
# TABLE TEXT
# ============================================================

def build_table_text(
    page_number,
    title,
    headers,
    rows,
):
    parts = [
        f"Page: {page_number}",
    ]

    if title:
        parts.append(title)

    if headers:
        parts.append(
            " | ".join(headers)
        )

    for row in rows:

        parts.append(
            " | ".join(
                clean_cell(cell)
                for cell in row
                if clean_cell(cell)
            )
        )

    return "\n".join(parts)


# ============================================================
# NORMAL TABLE PROCESSING
# ============================================================

def process_normal_table(
    page,
    table,
    table_index,
):
    """
    Process a non-curriculum table.

    This is the existing/general pipeline.
    """

    if not table_has_content(table):
        return None

    if is_one_column_prose(table):
        return None

    try:
        raw_rows = table.extract()
    except Exception:
        return None

    if not raw_rows:
        return None

    # --------------------------------------------------------
    # Special table types
    # --------------------------------------------------------

    placement = normalize_placement_tests(
        raw_rows
    )

    if placement:

        return {
            "table_id": (
                f"page_{page.page_number}_"
                f"table_{table_index + 1}"
            ),
            "table_index": table_index,
            "page": page.page_number,
            "title": "Required Placement Tests",
            "type": "table",
            "headers": placement["headers"],
            "rows": placement["rows"],
            "notes": [],
            "text": build_table_text(
                page.page_number,
                "Required Placement Tests",
                placement["headers"],
                placement["rows"],
            ),
            "bbox": list(table.bbox),
        }

    grading = normalize_grading_table(
        raw_rows
    )

    if grading:

        return {
            "table_id": (
                f"page_{page.page_number}_"
                f"table_{table_index + 1}"
            ),
            "table_index": table_index,
            "page": page.page_number,
            "title": "",
            "type": "table",
            "headers": grading["headers"],
            "rows": grading["rows"],
            "notes": [],
            "text": build_table_text(
                page.page_number,
                "",
                grading["headers"],
                grading["rows"],
            ),
            "bbox": list(table.bbox),
        }

    notation = normalize_grade_notation(
        raw_rows
    )

    if notation:

        return {
            "table_id": (
                f"page_{page.page_number}_"
                f"table_{table_index + 1}"
            ),
            "table_index": table_index,
            "page": page.page_number,
            "title": "",
            "type": "table",
            "headers": notation["headers"],
            "rows": notation["rows"],
            "notes": [],
            "text": build_table_text(
                page.page_number,
                "",
                notation["headers"],
                notation["rows"],
            ),
            "bbox": list(table.bbox),
        }

    tuition = normalize_tuition(
        raw_rows
    )

    if tuition:

        return {
            "table_id": (
                f"page_{page.page_number}_"
                f"table_{table_index + 1}"
            ),
            "table_index": table_index,
            "page": page.page_number,
            "title": "Tuition (in AED)",
            "type": "table",
            "headers": tuition["headers"],
            "rows": tuition["rows"],
            "notes": [],
            "text": build_table_text(
                page.page_number,
                "Tuition (in AED)",
                tuition["headers"],
                tuition["rows"],
            ),
            "bbox": list(table.bbox),
        }

    compulsory = normalize_compulsory_fees(
        raw_rows
    )

    if compulsory:

        return {
            "table_id": (
                f"page_{page.page_number}_"
                f"table_{table_index + 1}"
            ),
            "table_index": table_index,
            "page": page.page_number,
            "title": "Compulsory Fees (in AED)",
            "type": "table",
            "headers": compulsory["headers"],
            "rows": compulsory["rows"],
            "notes": [],
            "text": build_table_text(
                page.page_number,
                "Compulsory Fees (in AED)",
                compulsory["headers"],
                compulsory["rows"],
            ),
            "bbox": list(table.bbox),
        }

    # --------------------------------------------------------
    # Generic table
    # --------------------------------------------------------

    rows = normalize_generic_table(
        raw_rows
    )

    if not rows:
        return None

    title = detect_title(
        raw_rows
    )

    # Use first logical row as headers only when it actually
    # looks like a header.
    headers = []

    if rows:

        first_row = rows[0]

        header_text = " ".join(
            first_row
        ).lower()

        header_keywords = [
            "name",
            "type",
            "description",
            "fee",
            "major",
            "category",
            "grade",
            "class",
            "credit",
            "course",
            "regular semester",
            "summer term",
        ]

        if any(
            keyword in header_text
            for keyword in header_keywords
        ):
            headers = first_row
            rows = rows[1:]

    return {
        "table_id": (
            f"page_{page.page_number}_"
            f"table_{table_index + 1}"
        ),
        "table_index": table_index,
        "page": page.page_number,
        "title": title,
        "type": "table",
        "headers": headers,
        "rows": rows,
        "notes": [],
        "text": build_table_text(
            page.page_number,
            title,
            headers,
            rows,
        ),
        "bbox": list(table.bbox),
    }


# ============================================================
# MAIN EXTRACTION
# ============================================================

def extract_tables():

    all_tables = []

    with pdfplumber.open(PDF_PATH) as pdf:

        for page_number, page in enumerate(
            pdf.pages,
            start=1
        ):

            # ==================================================
            # CURRICULUM PAGE
            # ==================================================

            if is_curriculum_page(page):

                curriculum_tables = (
                    extract_curriculum_blocks(page)
                )

                if curriculum_tables:

                    all_tables.extend(
                        curriculum_tables
                    )

                    print(
                        f"Page {page_number}: "
                        f"extracted "
                        f"{len(curriculum_tables)} "
                        f"curriculum sections"
                    )

                    # IMPORTANT:
                    #
                    # Do NOT also run find_tables() on this
                    # page. Otherwise we get the fragmented
                    # curriculum tables again.
                    #
                    continue

            # ==================================================
            # NORMAL TABLE PAGE
            # ==================================================

            table_settings = {
                "vertical_strategy": "lines",
                "horizontal_strategy": "lines",
                "intersection_tolerance": 5,
            }

            try:
                tables = page.find_tables(
                    table_settings
                )
            except Exception:
                tables = []

            for table_index, table in enumerate(
                tables
            ):

                processed = process_normal_table(
                    page,
                    table,
                    table_index,
                )

                if processed:
                    all_tables.append(
                        processed
                    )

    # ==========================================================
    # SAVE
    # ==========================================================

    Path(
        "data/processed"
    ).mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            all_tables,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print(
        f"Extracted {len(all_tables)} tables/sections."
    )
    print(
        f"Saved to: {OUTPUT_PATH}"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    extract_tables()