import fitz
import json
from pathlib import Path
from src.config import (PDF_PATH, PAGES_PATH)


OUTPUT_PATH = PAGES_PATH


def extract_pdf():
    doc = fitz.open(PDF_PATH)

    pages = []

    for page_index, page in enumerate(doc):

        page_number = page_index + 1

        blocks = page.get_text("blocks")

        page_blocks = []

        for block in blocks:

            if len(block) < 5:
                continue

            x0, y0, x1, y1, text = block[:5]

            text = text.strip()

            if not text:
                continue

            page_blocks.append({
                "x0": x0,
                "y0": y0,
                "x1": x1,
                "y1": y1,
                "text": text,
            })

        # Keep the original PyMuPDF text as a fallback.
        raw_text = page.get_text("text")

        pages.append({
            "page": page_number,
            "text": raw_text,
            "blocks": page_blocks,
        })

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            pages,
            f,
            indent=2,
            ensure_ascii=False
        )

    print(f"Extracted {len(pages)} pages")
    print(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    extract_pdf()