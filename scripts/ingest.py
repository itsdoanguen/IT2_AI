"""CLI script to ingest processed DUT RAG chunks into persistent ChromaDB."""

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Optional

import chromadb

# Allow running directly via `python scripts/ingest.py` from IT2_AI root.
SERVICE_ROOT = Path(__file__).resolve().parent.parent
if str(SERVICE_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVICE_ROOT))

from app.config import settings
from app.vector_store import build_chunk_metadata, build_embedding_function


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ingest rag_chunks.jsonl into ChromaDB persistent storage."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=settings.rag_chunks_file_path,
        help="Path to rag_chunks.jsonl",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Upsert batch size (default: 64)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional max number of chunks to ingest (for quick local testing)",
    )
    parser.add_argument(
        "--fields",
        type=str,
        default=None,
        help="Semicolon-separated list of fields to ingest, e.g. 'Công tác sinh viên;Đào tạo;Khảo thí'",
    )
    parser.add_argument(
        "--field",
        action="append",
        dest="field_list",
        default=None,
        help="Single field to ingest (can be repeated multiple times)",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete existing collection before ingesting (required if changing embedding model)",
    )
    return parser.parse_args()


def load_chunks(
    input_path: Path,
    allowed_fields: Optional[set[str]] = None,
    limit: Optional[int] = None,
) -> tuple[list[str], list[str], list[dict[str, str | int]]]:
    """Read and filter chunks from JSONL into parallel ChromaDB arrays."""
    ids: list[str] = []
    documents: list[str] = []
    metadatas: list[dict[str, str | int]] = []

    with input_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if allowed_fields and item.get("field") not in allowed_fields:
                continue

            ids.append(str(item["chunk_id"]))
            documents.append(str(item["full_rag_text"]))
            metadatas.append(build_chunk_metadata(item))

            if limit is not None and len(ids) >= limit:
                break

    return ids, documents, metadatas


def main() -> None:
    args = parse_args()
    input_path: Path = args.input.resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {input_path}")

    selected_fields: set[str] = set()
    if args.fields:
        delimiter = ";" if ";" in args.fields else ","
        selected_fields.update(f.strip() for f in args.fields.split(delimiter) if f.strip())
    if args.field_list:
        selected_fields.update(f.strip() for f in args.field_list if f.strip())
    allowed_fields = selected_fields or None

    print(f"[1/4] Reading chunks from: {input_path}")
    ids, documents, metadatas = load_chunks(
        input_path=input_path,
        allowed_fields=allowed_fields,
        limit=args.limit,
    )
    print(f"      Selected {len(ids)} chunks" + (f" (fields={sorted(allowed_fields)})" if allowed_fields else ""))
    if not ids:
        print("No chunks matched the provided filters. Exiting.")
        return

    print(f"[2/4] Loading embedding model '{settings.EMBEDDING_MODEL}' on {settings.EMBEDDING_DEVICE}...")
    t0 = time.perf_counter()
    embedding_fn = build_embedding_function(
        model_name=settings.EMBEDDING_MODEL,
        device=settings.EMBEDDING_DEVICE,
    )

    persist_dir = settings.persist_dir_path
    persist_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(persist_dir))

    if args.reset:
        try:
            client.delete_collection(name=settings.CHROMA_COLLECTION_NAME)
            print(f"      Reset existing collection '{settings.CHROMA_COLLECTION_NAME}'")
        except Exception:
            pass

    collection = client.get_or_create_collection(
        name=settings.CHROMA_COLLECTION_NAME,
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"},
    )

    print(f"[3/4] Upserting {len(ids)} chunks into '{persist_dir}' (batch_size={args.batch_size})...")
    total_batches = (len(ids) + args.batch_size - 1) // args.batch_size
    for idx in range(0, len(ids), args.batch_size):
        batch_num = idx // args.batch_size + 1
        collection.upsert(
            ids=ids[idx : idx + args.batch_size],
            documents=documents[idx : idx + args.batch_size],
            metadatas=metadatas[idx : idx + args.batch_size],
        )
        if batch_num % 5 == 0 or batch_num == total_batches:
            elapsed = time.perf_counter() - t0
            print(f"      Batch {batch_num}/{total_batches} done ({elapsed:.1f}s)")

    print(f"[4/4] Completed! Collection '{settings.CHROMA_COLLECTION_NAME}' now holds {collection.count()} chunks.")


if __name__ == "__main__":
    main()
