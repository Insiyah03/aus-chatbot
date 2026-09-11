import re

CATALOG_BANNER = re.compile(
    r"Undergraduate Catalog 2025[\s\S]{0,3}2026",
    re.IGNORECASE,
)
AUS_LINE = re.compile(r"^American University of Sharjah\s*$", re.IGNORECASE)
ROMAN_PAGE = re.compile(r"^[ivxlcdm]+\s*$", re.IGNORECASE)
ARABIC_PAGE = re.compile(r"^\d{1,3}\s*$")
REPEATED_HEADERS = {
    "undergraduate course descriptions",
    "academic policies and regulations",
    "college of architecture, art and design",
    "college of arts and sciences",
    "college of engineering",
    "school of business administration",
    "the university",
    "campus life",
    "table of contents",
    "full-time faculty",
}


def extract_catalog_page(text: str) -> str | None:
    for line in text.splitlines()[:18]:
        stripped = line.strip()
        if ROMAN_PAGE.match(stripped) or ARABIC_PAGE.match(stripped):
            return stripped.lower()
    return None


def strip_running_headers(text: str) -> str:
    kept = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            kept.append("")
            continue
        if CATALOG_BANNER.search(stripped):
            continue
        if AUS_LINE.match(stripped):
            continue
        if ROMAN_PAGE.match(stripped) or ARABIC_PAGE.match(stripped):
            continue
        kept.append(stripped)

    joined = "\n".join(kept)
    joined = re.sub(r"(\w)-\n(\w)", r"\1\2", joined)
    joined = re.sub(r"[ \t]+", " ", joined)
    joined = re.sub(r"\n{3,}", "\n\n", joined)
    return joined.strip()


def normalize_whitespace(text: str) -> str:
    text = text.replace("\u00ad", "")
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()
