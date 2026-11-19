"""Pydantic schemas for the FastAPI vector search endpoints."""

from typing import Optional
from pydantic import BaseModel, Field, field_validator


class VectorSearchFilters(BaseModel):
    """Optional metadata filters applied during ChromaDB query."""

    field: Optional[str] = Field(None, description="Lọc theo lĩnh vực, VD: 'Công tác sinh viên'")
    document_number: Optional[str] = Field(None, description="Lọc theo số hiệu VB, VD: '1414/QĐ-ĐHBK'")
    status: Optional[str] = Field(None, description="Lọc theo hiệu lực, VD: 'Còn hiệu lực'")


class VectorSearchRequest(BaseModel):
    """Request payload for POST /api/vector/search."""

    query: str = Field(..., min_length=1, max_length=1000, description="Câu truy vấn ngữ nghĩa")
    top_k: int = Field(default=5, ge=1, le=50, description="Số lượng chunks trả về tối đa")
    min_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Ngưỡng cosine similarity tối thiểu")
    filters: Optional[VectorSearchFilters] = Field(default=None, description="Bộ lọc metadata tùy chọn")
    check_guardrail: bool = Field(default=True, description="Bật kiểm tra Intent Router Guardrail trước khi truy xuất")

    @field_validator("query")
    @classmethod
    def query_must_not_be_blank(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("query không được để trống")
        return cleaned


class ChunkMetadata(BaseModel):
    """Legal document metadata attached to each retrieved chunk."""

    document_id: str = ""
    document_number: str = ""
    document_title: str = ""
    field: str = ""
    issuer: str = ""
    status: str = ""
    chapter_title: str = ""
    article_number: str = ""
    article_title: str = ""
    clause_number: str = ""
    token_count_estimate: int = 0


class VectorSearchChunk(BaseModel):
    """Single semantic search match returned from ChromaDB."""

    chunk_id: str
    score: float = Field(..., description="Cosine similarity (0..1, càng cao càng liên quan)")
    distance: float = Field(..., description="Cosine distance từ ChromaDB")
    content: str = Field(..., description="Nội dung điều khoản gốc (dùng làm excerpt)")
    full_rag_text: str = Field(..., description="Nội dung kèm Context Header (dùng đưa vào prompt LLM)")
    metadata: ChunkMetadata


class VectorSearchResponse(BaseModel):
    """Response payload for POST /api/vector/search."""

    query: str
    top_k: int
    total_found: int
    embedding_model: str
    context: str = Field(default="", description="Ngữ cảnh gộp sẵn từ các chunks tìm được cho RAG LLM")
    results: list[VectorSearchChunk]
    blocked: bool = Field(default=False, description="True nếu câu hỏi bị Guardrail chặn")
    guardrail_category: Optional[str] = Field(default=None, description="Loại vi phạm/ngoài luồng nếu bị chặn")
    guardrail_reason: Optional[str] = Field(default=None, description="Lý do phân loại của Guardrail")
    refusal_message: Optional[str] = Field(default=None, description="Thông điệp từ chối an toàn trả về cho người dùng")


class GuardrailRequest(BaseModel):
    """Request payload for POST /api/guardrail."""

    query: str = Field(..., min_length=1, max_length=1000, description="Câu hỏi cần kiểm tra trước khi chạy RAG")

    @field_validator("query")
    @classmethod
    def query_must_not_be_blank(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("query không được để trống")
        return cleaned


class GuardrailResponse(BaseModel):
    """Response payload for POST /api/guardrail."""

    query: str
    allowed: bool
    intent: str
    category: Optional[str] = None
    reason: str
    refusal_message: Optional[str] = None
    latency_ms: float


class HealthResponse(BaseModel):
    """Health check response for GET /health."""

    status: str
    collection_name: str
    total_chunks: int
    embedding_model: str
