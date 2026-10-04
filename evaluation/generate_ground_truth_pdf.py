"""Build the auditable PDF directly from the benchmark's source dataset."""

from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Iterable

from evaluation.eval_dataset import EVAL_DATASET


OUTPUT_FILE = Path("evaluation/07_RAG_Evaluation_Ground_Truth.pdf")
PAGE_WIDTH = 595
PAGE_HEIGHT = 842


def _escape_pdf_text(text: str) -> bytes:
    encoded = text.encode("cp1252", errors="replace")
    encoded = encoded.replace(b"\\", b"\\\\")
    encoded = encoded.replace(b"(", b"\\(").replace(b")", b"\\)")
    return encoded


def _page_stream(page_number: int, page_count: int, items: Iterable[dict]) -> bytes:
    lines = [
        "RAG Evaluation Ground Truth",
        f"Corpus-grounded benchmark | page {page_number} of {page_count}",
        "",
    ]
    for item in items:
        fields = (
            ("ID", item["id"]),
            ("Question", item["question"]),
            ("Ground Truth", item["ground_truth"]),
            ("Source Document", item["source_document"]),
            ("Source Pages", ", ".join(map(str, item["source_pages"])) or "Not stated"),
            ("Source Section", item["source_section"]),
            ("Category", item["category"]),
            ("Difficulty", item["difficulty"]),
        )
        for label, value in fields:
            wrapped = textwrap.wrap(f"{label}: {value}", width=88) or [f"{label}:"]
            lines.extend(wrapped)
        lines.extend(["", "-" * 72, ""])

    commands = [b"BT", b"/F1 10 Tf", b"46 792 Td", b"14 TL"]
    for line in lines:
        commands.append(b"(" + _escape_pdf_text(line) + b") Tj")
        commands.append(b"T*")
    commands.append(b"ET")
    return b"\n".join(commands)


def generate_ground_truth_pdf() -> Path:
    """Write a dependency-free PDF with two benchmark records per page."""
    records_per_page = 2
    pages = [
        EVAL_DATASET[index : index + records_per_page]
        for index in range(0, len(EVAL_DATASET), records_per_page)
    ]
    page_count = len(pages)

    objects: list[bytes] = [b""]
    objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    page_ids = [4 + 2 * index for index in range(page_count)]
    kids = b" ".join(f"{page_id} 0 R".encode("ascii") for page_id in page_ids)
    objects.append(
        b"<< /Type /Pages /Kids [" + kids + f"] /Count {page_count} >>".encode("ascii")
    )
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>")

    for index, items in enumerate(pages, start=1):
        page_id = 4 + 2 * (index - 1)
        stream_id = page_id + 1
        stream = _page_stream(index, page_count, items)
        objects.append(
            (
                f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {PAGE_WIDTH} {PAGE_HEIGHT}] "
                f"/Resources << /Font << /F1 3 0 R >> >> /Contents {stream_id} 0 R >>"
            ).encode("ascii")
        )
        objects.append(
            b"<< /Length " + str(len(stream)).encode("ascii") + b" >>\nstream\n"
            + stream + b"\nendstream"
        )

    pdf = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0]
    for object_id, body in enumerate(objects[1:], start=1):
        offsets.append(len(pdf))
        pdf.extend(f"{object_id} 0 obj\n".encode("ascii"))
        pdf.extend(body)
        pdf.extend(b"\nendobj\n")

    xref_offset = len(pdf)
    pdf.extend(f"xref\n0 {len(objects)}\n".encode("ascii"))
    pdf.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        pdf.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    pdf.extend(
        f"trailer\n<< /Size {len(objects)} /Root 1 0 R >>\n"
        f"startxref\n{xref_offset}\n%%EOF\n".encode("ascii")
    )

    OUTPUT_FILE.parent.mkdir(exist_ok=True)
    OUTPUT_FILE.write_bytes(pdf)
    return OUTPUT_FILE


if __name__ == "__main__":
    print(generate_ground_truth_pdf())
