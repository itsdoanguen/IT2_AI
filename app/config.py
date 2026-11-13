"""Application configuration loaded from environment variables or .env."""

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Runtime settings for the FastAPI + ChromaDB vector search service."""

    APP_HOST: str = "0.0.0.0"
    APP_PORT: int = 8000
    CORS_ORIGINS: str = "http://localhost:3000,http://localhost:5173,http://localhost:8080"

    CHROMA_PERSIST_DIR: str = "./dut_rag_chroma_db"
    CHROMA_COLLECTION_NAME: str = "dut_legal_regulations"
    RAG_CHUNKS_PATH: str = "../dut_rag_dataset_processed/rag_chunks.jsonl"

    EMBEDDING_MODEL: str = "bkai-foundation-models/vietnamese-bi-encoder"
    EMBEDDING_DEVICE: str = "cpu"

    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    def resolve_path(self, raw_path: str) -> Path:
        """Resolve relative paths against the IT2_AI service root."""
        path = Path(raw_path)
        return path if path.is_absolute() else (BASE_DIR / path).resolve()

    @property
    def persist_dir_path(self) -> Path:
        return self.resolve_path(self.CHROMA_PERSIST_DIR)

    @property
    def rag_chunks_file_path(self) -> Path:
        return self.resolve_path(self.RAG_CHUNKS_PATH)

    @property
    def cors_origins_list(self) -> list[str]:
        origins = [item.strip() for item in self.CORS_ORIGINS.split(",") if item.strip()]
        return origins or ["*"]


settings = Settings()
