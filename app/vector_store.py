"""ChromaDB client, embedding setup, and semantic search service."""

from pathlib import Path
from typing import Any, Optional

import chromadb
from chromadb.utils import embedding_functions

from app.config import Settings, settings
from app.schemas import ChunkMetadata, VectorSearchChunk, VectorSearchFilters


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


class VectorStoreService:
    """Manages a persistent ChromaDB collection and semantic queries."""

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

    def search(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.0,
        filters: Optional[VectorSearchFilters] = None,
    ) -> list[VectorSearchChunk]:
        """Run semantic similarity search against ChromaDB."""
        total = self.count()
        if total == 0:
            return []

        n_results = min(top_k, total)
        where = build_chroma_where(filters)

        raw = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        ids = (raw.get("ids") or [[]])[0]
        documents = (raw.get("documents") or [[]])[0]
        metadatas = (raw.get("metadatas") or [[]])[0]
        distances = (raw.get("distances") or [[]])[0]

        results: list[VectorSearchChunk] = []
        for chunk_id, doc_text, meta, dist in zip(ids, documents, metadatas, distances):
            meta_dict = meta or {}
            distance_val = float(dist)
            similarity = round(max(0.0, 1.0 - distance_val), 4)
            if similarity < min_score:
                continue

            raw_content = str(meta_dict.get("content") or doc_text or "")
            results.append(
                VectorSearchChunk(
                    chunk_id=str(chunk_id),
                    score=similarity,
                    distance=round(distance_val, 4),
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
            )

        return results
