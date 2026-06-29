from __future__ import annotations

import argparse
from pathlib import Path

import pymupdf


def extract_text(pdf_path: Path, text_path: Path) -> None:
    doc = pymupdf.open(pdf_path)
    parts: list[str] = []
    for page_number, page in enumerate(doc, start=1):
        parts.append(f"\n\n## Page {page_number}\n\n{page.get_text('text')}")
    text_path.write_text("".join(parts), encoding="utf-8")


def render_pages(pdf_path: Path, render_dir: Path, pages: list[int] | None = None) -> list[Path]:
    doc = pymupdf.open(pdf_path)
    if pages is None:
        candidates = [1, 2, 3, 6, 10, 15, 20, doc.page_count]
        pages = sorted({p for p in candidates if 1 <= p <= doc.page_count})

    render_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for page_number in pages:
        page = doc[page_number - 1]
        pix = page.get_pixmap(matrix=pymupdf.Matrix(1.5, 1.5), alpha=False)
        out = render_dir / f"page_{page_number:02d}.png"
        pix.save(out)
        outputs.append(out)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract PDF text and render sample pages for paper reading notes.")
    parser.add_argument("pdf", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path(r"D:\TREE\paper reading\output"))
    args = parser.parse_args()

    pdf_path = args.pdf.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    text_path = output_dir / f"{pdf_path.stem}_extracted.txt"
    render_dir = output_dir / f"{pdf_path.stem}_rendered_pages"
    extract_text(pdf_path, text_path)
    rendered = render_pages(pdf_path, render_dir)

    doc = pymupdf.open(pdf_path)
    print(f"pdf={pdf_path}")
    print(f"pages={doc.page_count}")
    print(f"text={text_path}")
    print(f"render_dir={render_dir}")
    for path in rendered:
        print(f"rendered={path}")


if __name__ == "__main__":
    main()
