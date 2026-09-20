import pymupdf
import json

from src.config import PDF_PATH, PAGES_PATH


OUTPUT_PATH = PAGES_PATH


def clean_headers_footers(blocks, page_height):
    """
    Remove header and footer blocks based on their vertical position.
    """

    HEADER_HEIGHT = 50
    FOOTER_HEIGHT = 50

    cleaned_blocks = []

    for block in blocks:

        y0 = block["y0"]
        y1 = block["y1"]

        # Remove header
        if y1 <= HEADER_HEIGHT:
            continue

        # Remove footer
        if y0 >= page_height - FOOTER_HEIGHT:
            continue

        cleaned_blocks.append(block)

    return cleaned_blocks


def extract_pdf():
    doc = pymupdf.open(PDF_PATH)

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

        # Remove headers and footers
        page_blocks = clean_headers_footers(
            page_blocks,
            page.rect.height
        )

        # Sort blocks into reading order
        page_blocks.sort(
            key=lambda block: (
                block["y0"],
                block["x0"]
            )
        )

        # Reconstruct cleaned page text
        cleaned_text = "\n".join(
            block["text"]
            for block in page_blocks
        )

        pages.append({
            "page": page_number,
            "text": cleaned_text,
            "blocks": page_blocks,
        })

    doc.close()

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