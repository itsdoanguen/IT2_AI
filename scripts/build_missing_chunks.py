"""Build fallback RAG chunks for regulations missing from rag_chunks.jsonl.

Covers:
1. All structured_docs entries whose document_id has 0 chunks in rag_chunks.jsonl
   (typically because OCR wrote 'Diu'/'Biu' instead of 'Điều' or used non-article headings).
2. Core scholarship seed regulations cached on backend/crawl-data (`1414/QĐ-ĐHBK`,
   `28/VBHN-BGDĐT`) via pdfplumber extraction.
"""

import argparse
import json
from pathlib import Path
import re
import sys
from typing import Any
from urllib.request import urlopen

import pdfplumber

SERVICE_ROOT = Path(__file__).resolve().parent.parent
WORKSPACE_ROOT = SERVICE_ROOT.parent
DATASET_DIR = WORKSPACE_ROOT / "dut_rag_dataset_processed"

ARTICLE_RE = re.compile(
    r"(?m)^\s*(?:Điều|Điu|Diu|Biu|Biêu|Dieu)\s+(\d+[a-z]?)\s*[\.:]?\s*(.*)$"
)
END_NOISE_RE = re.compile(r"(?im)^\s*(?:Nơi nhận|Noi nhan)\s*[:.]?\s*$")


def _clean_text(raw: str) -> str:
    lines = [ln.rstrip() for ln in raw.replace("\r\n", "\n").split("\n")]
    kept: list[str] = []
    for ln in lines:
        if END_NOISE_RE.match(ln.strip()):
            break
        kept.append(ln)
    text = "\n".join(kept).strip()
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def _split_sliding(text: str, max_chars: int = 1200, overlap: int = 180) -> list[str]:
    if len(text) <= max_chars:
        return [text] if text.strip() else []
    parts: list[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + max_chars)
        if end < len(text):
            break_at = text.rfind("\n", start + max_chars // 2, end)
            if break_at <= start:
                break_at = text.rfind(" ", start + max_chars // 2, end)
            if break_at > start:
                end = break_at
        piece = text[start:end].strip()
        if len(piece) >= 60:
            parts.append(piece)
        if end >= len(text):
            break
        start = max(start + 1, end - overlap)
    return parts


def chunk_document_text(doc_meta: dict[str, Any], raw_text: str) -> list[dict[str, Any]]:
    cleaned = _clean_text(raw_text)
    if len(cleaned) < 60:
        return []

    doc_id = str(doc_meta.get("document_id") or "")
    doc_num = str(doc_meta.get("document_number") or "")
    doc_title = str(doc_meta.get("title") or "")
    field = str(doc_meta.get("field") or "")
    issuer = str(doc_meta.get("issuer") or "")
    status = str(doc_meta.get("status") or "Còn hiệu lực")

    matches = list(ARTICLE_RE.finditer(cleaned))
    sections: list[tuple[str, str, str]] = []

    if len(matches) >= 2:
        preamble = cleaned[: matches[0].start()].strip()
        if len(preamble) >= 120:
            sections.append(("Mở đầu", "", preamble))
        for idx, m in enumerate(matches):
            start = m.start()
            end = matches[idx + 1].start() if idx + 1 < len(matches) else len(cleaned)
            body = cleaned[start:end].strip()
            art_num = f"Điều {m.group(1)}"
            art_title = m.group(2).strip()[:160]
            if len(body) >= 60:
                sections.append((art_num, art_title, body))
    else:
        for idx, piece in enumerate(_split_sliding(cleaned), start=1):
            sections.append((f"Phần {idx}", "", piece))

    out: list[dict[str, Any]] = []
    chunk_counter = 0
    for art_num, art_title, body in sections:
        for sub_idx, piece in enumerate(_split_sliding(body, max_chars=1400, overlap=180), start=1):
            chunk_counter += 1
            header = f"[VĂN BẢN: {doc_num} - {doc_title}] | {art_num}"
            if art_title:
                header += f" ({art_title})"
            full_rag_text = f"{header}\n{piece}"
            out.append(
                {
                    "chunk_id": f"{doc_id}_fallback_{chunk_counter}_{sub_idx}",
                    "document_id": doc_id,
                    "document_number": doc_num,
                    "document_title": doc_title,
                    "field": field,
                    "issuer": issuer,
                    "status": status,
                    "chapter_title": "",
                    "article_number": art_num,
                    "article_title": art_title,
                    "clause_number": str(sub_idx) if len(body) > 1400 else "all",
                    "context_header": header,
                    "content": piece,
                    "full_rag_text": full_rag_text,
                    "token_count_estimate": max(1, len(piece.split())),
                }
            )
    return out


def _extract_pdf_text(pdf_bytes: bytes) -> str:
    import io

    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        pages = [(p.extract_text() or "").strip() for p in pdf.pages]
    return "\n\n".join(p for p in pages if p)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build missing chunks JSONL.")
    parser.add_argument(
        "--output",
        type=Path,
        default=DATASET_DIR / "extra_missing_chunks.jsonl",
    )
    parser.add_argument(
        "--backend-url",
        type=str,
        default="http://localhost:8080",
        help="Backend URL to fetch cached seed PDFs (1414/QĐ-ĐHBK, 28/VBHN-BGDĐT).",
    )
    args = parser.parse_args()

    structured_path = DATASET_DIR / "structured_docs.json"
    chunks_path = DATASET_DIR / "rag_chunks.jsonl"
    raw_md_dir = DATASET_DIR / "raw_markdown"

    with structured_path.open("r", encoding="utf-8") as f:
        sdocs = json.load(f)

    existing_doc_ids: set[str] = set()
    with chunks_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                existing_doc_ids.add(str(json.loads(line).get("document_id") or ""))

    md_by_doc_id: dict[str, Path] = {}
    for p in raw_md_dir.glob("*.md"):
        prefix = p.name.split("_", 1)[0]
        md_by_doc_id[prefix] = p

    all_new_chunks: list[dict[str, Any]] = []
    recovered_docs = 0

    for doc in sdocs:
        doc_id = str(doc.get("document_id") or "")
        if not doc_id or doc_id in existing_doc_ids:
            continue
        md_path = md_by_doc_id.get(doc_id)
        if not md_path or not md_path.exists():
            continue
        raw_text = md_path.read_text(encoding="utf-8", errors="ignore")
        doc_chunks = chunk_document_text(doc, raw_text)
        if doc_chunks:
            recovered_docs += 1
            all_new_chunks.extend(doc_chunks)

    print(f"Recovered {len(all_new_chunks)} chunks from {recovered_docs} markdown docs missing in rag_chunks.jsonl.")

    seed_pdfs = [
        (
            1,
            {
                "document_id": "0018",
                "document_number": "1414/QĐ-ĐHBK",
                "title": "Quy định về việc xét, cấp học bổng cho sinh viên hệ đại học chính quy tại Trường ĐHBK - ĐHĐN",
                "field": "Công tác sinh viên",
                "issuer": "Trường Đại học Bách khoa - ĐHĐN",
                "status": "Còn hiệu lực",
            },
        ),
        (
            2,
            {
                "document_id": "0009",
                "document_number": "28/VBHN-BGDĐT",
                "title": "Quyết định về học bổng khuyến khích học tập đối với học sinh, sinh viên",
                "field": "Công tác sinh viên",
                "issuer": "Bộ Giáo dục và Đào tạo",
                "status": "Còn hiệu lực",
            },
        ),
    ]

    for reg_id, meta in seed_pdfs:
        if meta["document_id"] in existing_doc_ids:
            continue
        try:
            with urlopen(f"{args.backend_url}/api/regulations/{reg_id}/pdf", timeout=15) as resp:
                pdf_bytes = resp.read()
            if pdf_bytes[:4] == b"%PDF":
                extracted = _extract_pdf_text(pdf_bytes)
                seed_chunks = chunk_document_text(meta, extracted)
                print(f"Extracted {len(seed_chunks)} chunks for seed regulation {meta['document_number']} ({len(extracted)} chars).")
                all_new_chunks.extend(seed_chunks)
        except Exception as exc:
            print(f"Skipped seed regulation {meta['document_number']}: {exc}", file=sys.stderr)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as out_f:
        for ch in all_new_chunks:
            out_f.write(json.dumps(ch, ensure_ascii=False) + "\n")

    print(f"Wrote {len(all_new_chunks)} total fallback chunks to {args.output}")


if __name__ == "__main__":
    main()
