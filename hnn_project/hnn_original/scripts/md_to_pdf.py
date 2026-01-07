#!/usr/bin/env python3
"""
Minimal Markdown-to-PDF converter tailored for textual reports.
It performs a lightweight markdown stripping and writes a PDF with a single
Type1 font (Helvetica). The implementation avoids external dependencies so
it can run inside restricted environments.
"""

import sys
import textwrap
from pathlib import Path


PAGE_WIDTH = 612  # points (8.5in)
PAGE_HEIGHT = 792  # points (11in)
MARGIN_X = 72
MARGIN_Y = 72
FONT_SIZE = 12
LINE_HEIGHT = 14


def read_markdown(path: Path) -> list[str]:
    raw = path.read_text(encoding="utf-8").splitlines()
    lines: list[str] = []

    for line in raw:
        stripped = line.strip()
        if not stripped:
            lines.append("")
            continue
        if stripped.startswith("###"):
            header = stripped[3:].strip()
            lines.append(header.upper())
            lines.append("")
            continue
        if stripped.startswith("##"):
            header = stripped[2:].strip()
            lines.append(header.upper())
            lines.append("")
            continue
        if stripped.startswith("#"):
            header = stripped[1:].strip()
            lines.append(header.upper())
            lines.append("")
            continue
        if stripped.startswith("- "):
            content = stripped[2:].strip()
            bullet = f"- {content}"
            wrapped = textwrap.wrap(bullet, width=88)
            lines.extend(wrapped or ["-"])
            continue
        if stripped.startswith("```"):
            # We only support inline code blocks by adding a marker line.
            lines.append("")
            lines.append("[CODE BLOCK]")
            lines.append("")
            continue
        wrapped = textwrap.wrap(stripped, width=90)
        lines.extend(wrapped or [""])
    return lines


def escape_pdf_text(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def paginate(lines: list[str]) -> list[list[str]]:
    pages: list[list[str]] = []
    current: list[str] = []
    y = PAGE_HEIGHT - MARGIN_Y
    for line in lines:
        if y < MARGIN_Y:
            pages.append(current)
            current = []
            y = PAGE_HEIGHT - MARGIN_Y
        current.append(line)
        y -= LINE_HEIGHT
    if current:
        pages.append(current)
    return pages or [[]]


def lines_to_stream(page_lines: list[str]) -> str:
    parts = ["BT", f"/F1 {FONT_SIZE} Tf"]
    current_y = PAGE_HEIGHT - MARGIN_Y
    for entry in page_lines:
        if entry:
            escaped = escape_pdf_text(entry)
            parts.append(f"1 0 0 1 {MARGIN_X} {current_y} Tm ({escaped}) Tj")
        current_y -= LINE_HEIGHT
    parts.append("ET")
    return "\n".join(parts) + "\n"


def build_pdf(lines: list[str]) -> bytes:
    objects: list[str] = []

    def add_object(data: str) -> int:
        objects.append(data)
        return len(objects)

    font_obj_id = add_object("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    page_entries: list[tuple[int, int]] = []
    pages = paginate(lines)

    for page_lines in pages:
        stream = lines_to_stream(page_lines)
        content_obj_id = add_object(f"<< /Length {len(stream.encode('utf-8'))} >>\nstream\n{stream}endstream")
        page_obj_id = add_object("PAGE_PLACEHOLDER")
        page_entries.append((page_obj_id, content_obj_id))

    pages_obj_id = add_object(
        "<< /Type /Pages /Kids [{kids}] /Count {count} >>".format(
            kids=" ".join(f"{pid} 0 R" for pid, _ in page_entries),
            count=len(page_entries),
        )
    )
    catalog_obj_id = add_object(f"<< /Type /Catalog /Pages {pages_obj_id} 0 R >>")

    for page_obj_id, content_obj_id in page_entries:
        objects[page_obj_id - 1] = (
            "<< /Type /Page /Parent {parent} 0 R /MediaBox [0 0 {w} {h}] "
            "/Resources << /Font << /F1 {font} 0 R >> >> /Contents {content} 0 R >>"
        ).format(
            parent=pages_obj_id,
            w=PAGE_WIDTH,
            h=PAGE_HEIGHT,
            font=font_obj_id,
            content=content_obj_id,
        )

    offsets: list[int] = []
    pdf_bytes = bytearray(b"%PDF-1.4\n")
    for obj_id, obj in enumerate(objects, start=1):
        offsets.append(len(pdf_bytes))
        pdf_bytes.extend(f"{obj_id} 0 obj\n".encode("utf-8"))
        pdf_bytes.extend((obj + "\n").encode("utf-8"))
        pdf_bytes.extend(b"endobj\n")

    xref_offset = len(pdf_bytes)
    pdf_bytes.extend(f"xref\n0 {len(objects)+1}\n".encode("utf-8"))
    pdf_bytes.extend(b"0000000000 65535 f \n")
    for off in offsets:
        pdf_bytes.extend(f"{off:010} 00000 n \n".encode("utf-8"))

    pdf_bytes.extend(b"trailer\n")
    pdf_bytes.extend(f"<< /Size {len(objects)+1} /Root {catalog_obj_id} 0 R >>\n".encode("utf-8"))
    pdf_bytes.extend(f"startxref\n{xref_offset}\n%%EOF\n".encode("utf-8"))
    return bytes(pdf_bytes)


def main():
    if len(sys.argv) != 3:
        print("Usage: python scripts/md_to_pdf.py <input.md> <output.pdf>")
        return 1
    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])
    lines = read_markdown(input_path)
    pdf_bytes = build_pdf(lines)
    output_path.write_bytes(pdf_bytes)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
