"""FastAPI entrypoint exposing ChromaDB semantic search endpoints."""

from contextlib import asynccontextmanager
import logging
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.schemas import HealthResponse, VectorSearchRequest, VectorSearchResponse
from app.vector_store import VectorStoreService

logger = logging.getLogger("it2_ai")


def create_app(vector_store: Optional[VectorStoreService] = None) -> FastAPI:
    """Build the FastAPI application with an optional injected VectorStoreService."""
    store = vector_store or VectorStoreService(app_settings=settings)

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        store.initialize()
        total = store.count()
        if total == 0:
            logger.warning(
                "ChromaDB collection '%s' is empty! Run `python scripts/ingest.py` first.",
                settings.CHROMA_COLLECTION_NAME,
            )
        else:
            logger.info(
                "Loaded ChromaDB collection '%s' with %d chunks (model=%s).",
                settings.CHROMA_COLLECTION_NAME,
                total,
                settings.EMBEDDING_MODEL,
            )
        application.state.vector_store = store
        yield

    application = FastAPI(
        title="DUT IT2 Vector Search API",
        description="FastAPI + ChromaDB semantic search service for DUT legal regulations.",
        version="1.0.0",
        lifespan=lifespan,
    )

    application.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @application.get("/health", response_model=HealthResponse, tags=["Health"])
    def health_check(request: Request) -> HealthResponse:
        active_store: VectorStoreService = request.app.state.vector_store
        return HealthResponse(
            status="ok",
            collection_name=settings.CHROMA_COLLECTION_NAME,
            total_chunks=active_store.count(),
            embedding_model=settings.EMBEDDING_MODEL,
        )

    @application.post("/api/vector/search", response_model=VectorSearchResponse, tags=["Vector Search"])
    @application.post("/api/search", response_model=VectorSearchResponse, tags=["Vector Search"], include_in_schema=True)
    def vector_search(payload: VectorSearchRequest, request: Request) -> VectorSearchResponse:
        active_store: VectorStoreService = request.app.state.vector_store
        try:
            results = active_store.search(
                query=payload.query,
                top_k=payload.top_k,
                min_score=payload.min_score,
                filters=payload.filters,
            )
        except Exception as exc:
            logger.exception("Vector search failed for query=%r", payload.query)
            raise HTTPException(status_code=500, detail=f"ChromaDB query error: {exc}") from exc

        context = "\n\n---\n\n".join(item.full_rag_text for item in results if item.full_rag_text)
        return VectorSearchResponse(
            query=payload.query,
            top_k=payload.top_k,
            total_found=len(results),
            embedding_model=settings.EMBEDDING_MODEL,
            context=context,
            results=results,
        )

    return application


app = create_app()
