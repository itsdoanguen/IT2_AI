"""Automated tests for the FastAPI + ChromaDB vector search service."""

from pathlib import Path
import sys
from typing import Any

from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
from fastapi.testclient import TestClient

SERVICE_ROOT = Path(__file__).resolve().parent.parent
if str(SERVICE_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVICE_ROOT))

from app.config import Settings
from app.main import create_app
from app.vector_store import VectorStoreService, build_chunk_metadata


class KeywordEmbeddingFunction(EmbeddingFunction[Documents]):
    """Lightweight deterministic embedding for offline unit tests."""

    KEYWORDS = (
        "học bổng",
        "khuyến khích",
        "tín chỉ",
        "cảnh báo",
        "rèn luyện",
        "xác nhận",
        "sinh viên",
        "nghĩa vụ",
    )

    def __init__(self) -> None:
        pass

    @staticmethod
    def name() -> str:
        return "test-keyword-embedder"

    def get_config(self) -> dict[str, Any]:
        return {}

    @classmethod
    def build_from_config(cls, config: dict[str, Any]) -> "KeywordEmbeddingFunction":
        return cls()

    def __call__(self, input: Documents) -> Embeddings:
        embeddings: Embeddings = []
        for text in input:
            lowered = text.lower()
            vec = [1.0 if kw in lowered else 0.0 for kw in self.KEYWORDS]
            # Avoid zero-norm vector in cosine HNSW when no keyword matches.
            vec.append(0.1)
            embeddings.append(vec)
        return embeddings


def _build_test_client(tmp_path: Path, seed_samples: bool = False) -> TestClient:
    test_settings = Settings(
        CHROMA_PERSIST_DIR=str(tmp_path / "chroma_test"),
        CHROMA_COLLECTION_NAME="test_dut_regulations",
        EMBEDDING_MODEL="test-keyword-embedder",
    )
    store = VectorStoreService(
        app_settings=test_settings,
        persist_dir=tmp_path / "chroma_test",
        embedding_function=KeywordEmbeddingFunction(),
    )
    store.initialize()

    if seed_samples:
        samples: list[dict[str, Any]] = [
            {
                "chunk_id": "001_dieu_4_1",
                "document_id": "001",
                "document_number": "1414/QĐ-ĐHBK",
                "document_title": "Quy định xét cấp Học bổng Khuyến khích học tập",
                "field": "Công tác sinh viên",
                "issuer": "Trường ĐHBK",
                "status": "Còn hiệu lực",
                "chapter_title": "Chương II",
                "article_number": "Điều 4",
                "article_title": "Tiêu chuẩn xét cấp học bổng",
                "clause_number": "1",
                "content": "Sinh viên đạt học lực Giỏi và điểm rèn luyện từ 80 trở lên.",
                "full_rag_text": "1414/QĐ-ĐHBK Điều 4: Sinh viên đạt học bổng khuyến khích và điểm rèn luyện từ 80.",
                "token_count_estimate": 35,
            },
            {
                "chunk_id": "002_dieu_12_1",
                "document_id": "002",
                "document_number": "2244/QĐ-ĐHBK",
                "document_title": "Quy chế đào tạo trình độ đại học hệ tín chỉ",
                "field": "Đào tạo",
                "issuer": "Trường ĐHBK",
                "status": "Còn hiệu lực",
                "chapter_title": "Chương III",
                "article_number": "Điều 12",
                "article_title": "Cảnh báo học tập",
                "clause_number": "1",
                "content": "Sinh viên không tích lũy đủ số tín chỉ tối thiểu sẽ bị cảnh báo học tập.",
                "full_rag_text": "2244/QĐ-ĐHBK Điều 12: Cảnh báo học tập khi thiếu tín chỉ.",
                "token_count_estimate": 30,
            },
            {
                "chunk_id": "003_dieu_2_1",
                "document_id": "003",
                "document_number": "1606/HD-ĐHBK",
                "document_title": "Hướng dẫn cấp giấy xác nhận sinh viên và tạm hoãn nghĩa vụ quân sự",
                "field": "Công tác sinh viên",
                "issuer": "Phòng CTSV",
                "status": "Còn hiệu lực",
                "chapter_title": "Mục I",
                "article_number": "Điều 2",
                "article_title": "Thủ tục xác nhận sinh viên",
                "clause_number": "1",
                "content": "Sinh viên đăng ký cấp giấy xác nhận nghĩa vụ quân sự trên cổng trực tuyến.",
                "full_rag_text": "1606/HD-ĐHBK Điều 2: Giấy xác nhận sinh viên và tạm hoãn nghĩa vụ quân sự.",
                "token_count_estimate": 28,
            },
        ]
        store.collection.upsert(
            ids=[s["chunk_id"] for s in samples],
            documents=[s["full_rag_text"] for s in samples],
            metadatas=[build_chunk_metadata(s) for s in samples],
        )

    app = create_app(vector_store=store)
    return TestClient(app)


def test_health_and_search_on_empty_collection(tmp_path: Path) -> None:
    with _build_test_client(tmp_path, seed_samples=False) as client:
        health_res = client.get("/health")
        assert health_res.status_code == 200
        assert health_res.json()["total_chunks"] == 0

        search_res = client.post("/api/vector/search", json={"query": "học bổng", "top_k": 5})
        assert search_res.status_code == 200
        body = search_res.json()
        assert body["total_found"] == 0
        assert body["results"] == []


def test_vector_search_returns_relevant_chunks_and_metadata(tmp_path: Path) -> None:
    with _build_test_client(tmp_path, seed_samples=True) as client:
        health_res = client.get("/health")
        assert health_res.status_code == 200
        assert health_res.json()["total_chunks"] == 3

        res = client.post(
            "/api/vector/search",
            json={"query": "tiêu chuẩn học bổng khuyến khích điểm rèn luyện", "top_k": 2},
        )
        assert res.status_code == 200
        body = res.json()
        assert body["total_found"] == 2
        top_hit = body["results"][0]
        assert top_hit["chunk_id"] == "001_dieu_4_1"
        assert top_hit["metadata"]["document_number"] == "1414/QĐ-ĐHBK"
        assert top_hit["metadata"]["article_number"] == "Điều 4"
        assert "rèn luyện" in top_hit["content"]
        assert top_hit["score"] >= body["results"][1]["score"]


def test_vector_search_with_metadata_filters(tmp_path: Path) -> None:
    with _build_test_client(tmp_path, seed_samples=True) as client:
        res = client.post(
            "/api/vector/search",
            json={
                "query": "học bổng tín chỉ",
                "top_k": 5,
                "filters": {"field": "Đào tạo"},
            },
        )
        assert res.status_code == 200
        results = res.json()["results"]
        assert len(results) == 1
        assert results[0]["chunk_id"] == "002_dieu_12_1"
        assert results[0]["metadata"]["field"] == "Đào tạo"


def test_vector_search_validation_errors(tmp_path: Path) -> None:
    with _build_test_client(tmp_path, seed_samples=True) as client:
        blank_res = client.post("/api/vector/search", json={"query": "   ", "top_k": 3})
        assert blank_res.status_code == 422

        bad_topk_low = client.post("/api/vector/search", json={"query": "học bổng", "top_k": 0})
        assert bad_topk_low.status_code == 422

        bad_topk_high = client.post("/api/vector/search", json={"query": "học bổng", "top_k": 100})
        assert bad_topk_high.status_code == 422


def test_spring_boot_alias_endpoint_and_context(tmp_path: Path) -> None:
    with _build_test_client(tmp_path, seed_samples=True) as client:
        res = client.post(
            "/api/search",
            json={"query": "học bổng khuyến khích", "topK": 2, "top_k": 2},
        )
        assert res.status_code == 200
        body = res.json()
        assert body["total_found"] == 2
        assert "1414/QĐ-ĐHBK" in body["context"]
