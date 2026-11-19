"""ChromaDB client, embedding setup, and hybrid semantic + lexical search service."""

from pathlib import Path
import re
from typing import Any, Optional
import unicodedata

import chromadb
from chromadb.utils import embedding_functions

from app.config import Settings, settings
from app.schemas import ChunkMetadata, VectorSearchChunk, VectorSearchFilters

_VI_STOPWORDS = frozenset(
    {
        "la",
        "cua",
        "va",
        "cac",
        "nhung",
        "trong",
        "duoc",
        "theo",
        "nao",
        "bao",
        "nhieu",
        "khi",
        "the",
        "co",
        "khong",
        "ve",
        "cho",
        "voi",
        "tai",
        "gi",
        "se",
        "bi",
        "phai",
        "can",
        "mot",
        "hay",
        "hoac",
        "de",
        "nhu",
        "nay",
        "do",
        "sau",
        "tren",
        "duoi",
        "tu",
        "den",
        "ra",
        "vao",
        "ve",
        "quy",
        "dinh",
        "trinh",
        "sinh",
        "vien",
    }
)

_HIGH_SIGNAL_PHRASES = (
    "buoc thoi hoc",
    "canh bao hoc",
    "khoi luong hoc tap",
    "khuyen khich hoc tap",
    "hoc bong",
    "tin chi",
    "no tin chi",
    "khong dat",
    "rut hoc phan",
    "rut bot hoc phan",
    "xac nhan sinh vien",
    "diem ren luyen",
    "thang diem 4",
    "xuat sac",
    "loai gioi",
    "loai kha",
    "mien giam hoc phi",
    "phuc khao",
    "nghi hoc tam thoi",
    "bao luu",
    "tot nghiep",
    "chuan dau ra",
)


def _strip_accents_and_ocr(text: str) -> str:
    """Lowercase, strip Vietnamese accents, and normalize common OCR digit/char slips for matching."""
    if not text:
        return ""
    s = text.replace("đ", "d").replace("Đ", "D").lower()
    s = "".join(
        ch for ch in unicodedata.normalize("NFKD", s) if not unicodedata.combining(ch)
    )
    # Normalize digits embedded inside alphabetic tokens from OCR (e.g. kh6i -> khoi, h9c -> hoc, t1n -> tin).
    s = re.sub(r"(?<=[a-z])6(?=[a-z])", "o", s)
    s = re.sub(r"(?<=[a-z])9(?=[a-z])", "o", s)
    s = re.sub(r"(?<=[a-z])0(?=[a-z])", "o", s)
    s = re.sub(r"(?<=[a-z])1(?=[a-z])", "i", s)
    s = re.sub(r"(?<=[a-z])5(?=[a-z])", "o", s)
    s = re.sub(r"[^a-z0-9/]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _tokenize_query(norm_query: str) -> tuple[list[str], list[str], list[str]]:
    raw_tokens = [t for t in norm_query.split() if len(t) >= 2]
    meaningful = [t for t in raw_tokens if t not in _VI_STOPWORDS]
    if not meaningful:
        meaningful = raw_tokens
    bigrams = [f"{raw_tokens[i]} {raw_tokens[i + 1]}" for i in range(len(raw_tokens) - 1)]
    trigrams = [
        f"{raw_tokens[i]} {raw_tokens[i + 1]} {raw_tokens[i + 2]}"
        for i in range(len(raw_tokens) - 2)
    ]
    return meaningful, bigrams, trigrams


def _is_boilerplate_chunk(raw_content: str, norm_content: str) -> bool:
    if len(raw_content.strip()) < 90:
        return True
    boilerplate_markers = (
        "noi nhan",
        "chanh van phong",
        "luu vt",
        "quyet dinh nay co hieu luc ke tu ngay ky",
        "thong tu nay co hieu luc thi hanh",
        "chiu trach nhiem thi hanh quyet dinh nay",
    )
    if len(raw_content.strip()) < 360 and any(m in norm_content for m in boilerplate_markers):
        return True
    return False


def build_embedding_function(model_name: str, device: str = "cpu") -> Any:
    """Create a SentenceTransformer embedding function for ChromaDB."""
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=model_name,
        device=device,
    )


def build_chunk_metadata(item: dict[str, Any]) -> dict[str, str | int]:
    """Extract Chroma-safe metadata (no None values allowed by ChromaDB)."""
    return {
        "document_id": str(item.get("document_id") or ""),
        "document_number": str(item.get("document_number") or ""),
        "document_title": str(item.get("document_title") or ""),
        "field": str(item.get("field") or ""),
        "issuer": str(item.get("issuer") or ""),
        "status": str(item.get("status") or ""),
        "chapter_title": str(item.get("chapter_title") or ""),
        "article_number": str(item.get("article_number") or ""),
        "article_title": str(item.get("article_title") or ""),
        "clause_number": str(item.get("clause_number") or ""),
        "content": str(item.get("content") or ""),
        "token_count_estimate": int(item.get("token_count_estimate") or 0),
    }


def build_chroma_where(filters: Optional[VectorSearchFilters]) -> Optional[dict[str, Any]]:
    """Translate optional search filters into a ChromaDB `where` expression."""
    if filters is None:
        return None

    conditions: list[dict[str, str]] = []
    for key in ("field", "document_number", "status"):
        value = getattr(filters, key, None)
        if value and value.strip():
            conditions.append({key: value.strip()})

    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


def _matches_filters(meta_dict: dict[str, Any], filters: Optional[VectorSearchFilters]) -> bool:
    if filters is None:
        return True
    for key in ("field", "document_number", "status"):
        val = getattr(filters, key, None)
        if val and val.strip():
            if str(meta_dict.get(key) or "").strip() != val.strip():
                return False
    return True


def _build_chunk_obj(
    chunk_id: str,
    score: float,
    distance: float,
    doc_text: str,
    meta_dict: dict[str, Any],
) -> VectorSearchChunk:
    raw_content = str(meta_dict.get("content") or doc_text or "")
    return VectorSearchChunk(
        chunk_id=str(chunk_id),
        score=round(max(0.0, min(1.0, score)), 4),
        distance=round(max(0.0, distance), 4),
        content=raw_content,
        full_rag_text=str(doc_text or ""),
        metadata=ChunkMetadata(
            document_id=str(meta_dict.get("document_id") or ""),
            document_number=str(meta_dict.get("document_number") or ""),
            document_title=str(meta_dict.get("document_title") or ""),
            field=str(meta_dict.get("field") or ""),
            issuer=str(meta_dict.get("issuer") or ""),
            status=str(meta_dict.get("status") or ""),
            chapter_title=str(meta_dict.get("chapter_title") or ""),
            article_number=str(meta_dict.get("article_number") or ""),
            article_title=str(meta_dict.get("article_title") or ""),
            clause_number=str(meta_dict.get("clause_number") or ""),
            token_count_estimate=int(meta_dict.get("token_count_estimate") or 0),
        ),
    )


class VectorStoreService:
    """Manages a persistent ChromaDB collection and hybrid semantic + lexical queries."""

    def __init__(
        self,
        app_settings: Settings = settings,
        persist_dir: Optional[Path] = None,
        embedding_function: Optional[Any] = None,
    ) -> None:
        self._settings = app_settings
        self._persist_dir = persist_dir or app_settings.persist_dir_path
        self._embedding_function = embedding_function
        self._client: Optional[chromadb.PersistentClient] = None
        self._collection: Optional[Any] = None
        self._corpus_count: int = -1
        self._corpus_entries: list[tuple[str, str, dict[str, Any], str, bool]] = []

    def initialize(self) -> None:
        """Load the embedding model and open the persistent Chroma collection."""
        if self._collection is not None:
            return

        self._persist_dir.mkdir(parents=True, exist_ok=True)
        if self._embedding_function is None:
            self._embedding_function = build_embedding_function(
                model_name=self._settings.EMBEDDING_MODEL,
                device=self._settings.EMBEDDING_DEVICE,
            )

        self._client = chromadb.PersistentClient(path=str(self._persist_dir))
        self._collection = self._client.get_or_create_collection(
            name=self._settings.CHROMA_COLLECTION_NAME,
            embedding_function=self._embedding_function,
            # Force cosine space so distance = 1 - cosine_similarity in [0, 2].
            metadata={"hnsw:space": "cosine"},
        )

    @property
    def collection(self) -> Any:
        if self._collection is None:
            self.initialize()
        return self._collection

    def count(self) -> int:
        """Return total indexed chunks in the active collection."""
        return int(self.collection.count())

    def _ensure_lexical_corpus(self, total: int) -> list[tuple[str, str, dict[str, Any], str, bool]]:
        """Lazily cache normalized chunk text for fast hybrid lexical + OCR-tolerant matching."""
        if self._corpus_count == total and self._corpus_entries:
            return self._corpus_entries
        if not hasattr(self.collection, "get"):
            return []

        try:
            snapshot = self.collection.get(include=["documents", "metadatas"])
        except Exception:
            return []

        ids = snapshot.get("ids") or []
        documents = snapshot.get("documents") or []
        metadatas = snapshot.get("metadatas") or []

        entries: list[tuple[str, str, dict[str, Any], str, bool]] = []
        for cid, doc_text, meta in zip(ids, documents, metadatas):
            meta_dict = meta or {}
            raw_content = str(meta_dict.get("content") or doc_text or "")
            combined_text = " ".join(
                filter(
                    None,
                    [
                        str(meta_dict.get("document_number") or ""),
                        str(meta_dict.get("document_title") or ""),
                        str(meta_dict.get("article_number") or ""),
                        str(meta_dict.get("article_title") or ""),
                        raw_content,
                    ],
                )
            )
            norm_text = _strip_accents_and_ocr(combined_text)
            is_boiler = _is_boilerplate_chunk(raw_content, norm_text)
            entries.append((str(cid), str(doc_text or ""), meta_dict, norm_text, is_boiler))

        self._corpus_entries = entries
        self._corpus_count = total
        return entries

    def _lexical_score(
        self,
        norm_query: str,
        meaningful_tokens: list[str],
        bigrams: list[str],
        trigrams: list[str],
        norm_text: str,
        meta_dict: dict[str, Any],
    ) -> float:
        if not norm_text or not meaningful_tokens:
            return 0.0

        token_hits = sum(1 for tok in meaningful_tokens if tok in norm_text)
        if token_hits == 0:
            return 0.0

        token_ratio = token_hits / len(meaningful_tokens)
        score = token_ratio * 0.35

        # Exact 2-gram and 3-gram phrase hits give strong disambiguation (e.g. "no tin chi" vs "tra no").
        if bigrams:
            bg_hits = sum(1 for bg in bigrams if bg in norm_text)
            score += min(0.30, (bg_hits / len(bigrams)) * 0.35)
        if trigrams:
            tg_hits = sum(1 for tg in trigrams if tg in norm_text)
            score += min(0.30, (tg_hits / len(trigrams)) * 0.45)

        for phrase in _HIGH_SIGNAL_PHRASES:
            if phrase in norm_query and phrase in norm_text:
                score += 0.14

        doc_num_norm = _strip_accents_and_ocr(str(meta_dict.get("document_number") or ""))
        if doc_num_norm and len(doc_num_norm) >= 3 and doc_num_norm in norm_query:
            score += 0.35

        return min(0.85, score)

    def search(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.0,
        filters: Optional[VectorSearchFilters] = None,
    ) -> list[VectorSearchChunk]:
        """Run hybrid (dense vector + lexical/phrase) search against ChromaDB with doc diversity."""
        total = self.count()
        if total == 0:
            return []

        where = build_chroma_where(filters)
        dense_limit = min(max(top_k * 8, 48), total)

        raw = self.collection.query(
            query_texts=[query],
            n_results=dense_limit,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        ids = (raw.get("ids") or [[]])[0]
        documents = (raw.get("documents") or [[]])[0]
        metadatas = (raw.get("metadatas") or [[]])[0]
        distances = (raw.get("distances") or [[]])[0]

        norm_query = _strip_accents_and_ocr(query)
        if "gpa" in norm_query and "hoc bong" not in norm_query:
            norm_query = f"{norm_query} xep loai hoc luc thang diem 4 hoc bong khuyen khich hoc tap"
        is_postgrad_query = any(
            kw in norm_query for kw in ("tien si", "thac si", "nghien cuu sinh", "cao hoc", "luan van", "luan an")
        )
        meaningful_tokens, bigrams, trigrams = _tokenize_query(norm_query)

        # Candidate map: chunk_id -> (doc_text, meta_dict, dense_score, dense_dist, norm_text, is_boiler)
        candidates: dict[str, tuple[str, dict[str, Any], float, float, str, bool]] = {}
        for chunk_id, doc_text, meta, dist in zip(ids, documents, metadatas, distances):
            meta_dict = meta or {}
            distance_val = float(dist)
            dense_sim = max(0.0, 1.0 - distance_val)
            raw_content = str(meta_dict.get("content") or doc_text or "")
            norm_text = _strip_accents_and_ocr(
                f"{meta_dict.get('document_number', '')} {meta_dict.get('document_title', '')} "
                f"{meta_dict.get('article_number', '')} {meta_dict.get('article_title', '')} {raw_content}"
            )
            candidates[str(chunk_id)] = (
                str(doc_text or ""),
                meta_dict,
                dense_sim,
                distance_val,
                norm_text,
                _is_boilerplate_chunk(raw_content, norm_text),
            )

        # Merge top lexical matches from the full corpus so exact terms aren't lost to dense drift.
        corpus = self._ensure_lexical_corpus(total)
        if corpus and meaningful_tokens:
            lexical_hits: list[tuple[float, str, str, dict[str, Any], str, bool]] = []
            for cid, doc_text, meta_dict, norm_text, is_boiler in corpus:
                if not _matches_filters(meta_dict, filters):
                    continue
                lex = self._lexical_score(
                    norm_query, meaningful_tokens, bigrams, trigrams, norm_text, meta_dict
                )
                if lex >= 0.18:
                    lexical_hits.append((lex, cid, doc_text, meta_dict, norm_text, is_boiler))
            lexical_hits.sort(key=lambda item: item[0], reverse=True)
            for lex, cid, doc_text, meta_dict, norm_text, is_boiler in lexical_hits[: max(top_k * 6, 36)]:
                if cid not in candidates:
                    candidates[cid] = (doc_text, meta_dict, 0.0, 1.0, norm_text, is_boiler)

        scored: list[tuple[float, str, str, dict[str, Any], float]] = []
        for cid, (doc_text, meta_dict, dense_sim, dense_dist, norm_text, is_boiler) in candidates.items():
            lex = self._lexical_score(
                norm_query, meaningful_tokens, bigrams, trigrams, norm_text, meta_dict
            )
            fused = dense_sim + lex * 0.72
            if is_boiler:
                fused -= 0.25

            doc_num_upper = str(meta_dict.get("document_number") or "").upper()
            doc_title_norm = _strip_accents_and_ocr(str(meta_dict.get("document_title") or ""))
            if any(tag in doc_num_upper for tag in ("ĐHBK", "ĐHĐN", "DHBK", "DHDN", "08/2021/TT-BGD", "56/2026/TT-BGD", "28/VBHN-BGD")):
                fused += 0.04
            if not is_postgrad_query and any(pg in doc_title_norm for pg in ("tien si", "thac si", "nghien cuu sinh")):
                fused -= 0.08

            final_score = max(0.0, min(1.0, fused))
            if final_score < min_score:
                continue
            scored.append((final_score, cid, doc_text, meta_dict, dense_dist))

        scored.sort(key=lambda item: item[0], reverse=True)

        # Apply per-document diversity cap (max 2 chunks per document) so one wrong doc cannot dominate top_k.
        selected: list[VectorSearchChunk] = []
        overflow: list[VectorSearchChunk] = []
        per_doc_counts: dict[str, int] = {}
        max_per_doc = 2 if top_k >= 4 else top_k

        for final_score, cid, doc_text, meta_dict, dense_dist in scored:
            doc_key = str(meta_dict.get("document_number") or meta_dict.get("document_id") or cid)
            chunk_obj = _build_chunk_obj(cid, final_score, dense_dist, doc_text, meta_dict)
            count_for_doc = per_doc_counts.get(doc_key, 0)
            if count_for_doc < max_per_doc:
                selected.append(chunk_obj)
                per_doc_counts[doc_key] = count_for_doc + 1
                if len(selected) >= top_k:
                    break
            else:
                overflow.append(chunk_obj)

        if len(selected) < top_k and overflow:
            needed = top_k - len(selected)
            selected.extend(overflow[:needed])

        return selected
