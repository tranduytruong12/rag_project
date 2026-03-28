"""
RAG Config — Settings module.

Loads all configuration from environment variables using Pydantic BaseSettings.
No hard-coded values. All secrets come from .env or the OS environment.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppEnv(str, Enum):
    """Deployment environment."""

    development = "development"
    staging = "staging"
    production = "production"


class LogLevel(str, Enum):
    """Allowed log levels."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class VectorStoreBackend(str, Enum):
    """Supported vector store backends."""

    chroma = "chroma"
    qdrant = "qdrant"
    pinecone = "pinecone"
    weaviate = "weaviate"


class LLMSettings(BaseSettings):
    """Settings for the LLM / generation backend."""

    model_config = SettingsConfigDict(env_prefix="OPENAI_", env_file=".env", extra="ignore")

    api_key: str = Field(default="", description="LLM API key")
    base_url: str = Field(default="https://api.openai.com/v1", description="LLM base URL")

    model_config = SettingsConfigDict(env_prefix="")  # override to use specific keys below


class Settings(BaseSettings):
    """
    Central application settings.

    All values are loaded from environment variables (or .env file).
    Nested config objects group related settings.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ------------------------------------------------------------------ App
    app_env: AppEnv = Field(default=AppEnv.development, description="Runtime environment")
    app_debug: bool = Field(default=False)
    log_level: LogLevel = Field(default=LogLevel.INFO)

    # ------------------------------------------------------------------ API
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000, ge=1, le=65535)

    # ------------------------------------------------------------------ LLM
    openai_api_key: str = Field(default="", description="OpenAI (or compatible) API key")
    openai_base_url: str = Field(default="https://api.openai.com/v1")
    llm_model_name: str = Field(default="gpt-4o-mini")
    llm_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    llm_max_tokens: int = Field(default=2048, gt=0)

    # ------------------------------------------------------------------ Embedding
    embedding_model_name: str = Field(default="text-embedding-3-small")
    embedding_dimension: int = Field(default=1536, gt=0)

    # ------------------------------------------------------------------ Vector Store
    vector_store_backend: VectorStoreBackend = Field(default=VectorStoreBackend.chroma)
    vector_store_collection: str = Field(default="rag_documents")
    chroma_persist_dir: Path = Field(default=Path("./data/chroma"))

    # qdrant (optional)
    qdrant_url: str = Field(default="http://localhost:6333")
    qdrant_api_key: str = Field(default="")

    # ------------------------------------------------------------------ Data paths
    data_raw_dir: Path = Field(default=Path("./data/raw"))
    data_processed_dir: Path = Field(default=Path("./data/processed"))

    # ------------------------------------------------------------------ Chunking
    chunk_size: int = Field(default=512, gt=0)
    chunk_overlap: int = Field(default=50, ge=0)

    # ------------------------------------------------------------------ Retrieval
    retrieval_top_k: int = Field(default=5, gt=0)

    # ------------------------------------------------------------------ Reranker
    reranker_enabled: bool = Field(default=False)
    reranker_model: str = Field(default="cross-encoder/ms-marco-MiniLM-L-6-v2")

    # ------------------------------------------------------------------ Evaluation
    eval_dataset_path: Path = Field(default=Path("./data/eval/dataset.json"))

    @field_validator("chunk_overlap")
    @classmethod
    def overlap_less_than_size(cls, v: int, info: object) -> int:
        """Ensure chunk_overlap < chunk_size."""
        # Check if chunk_size is already parsed before validating overlap
        if hasattr(info, "data") and "chunk_size" in info.data:
            if v >= info.data["chunk_size"]:
                raise ValueError("chunk_overlap must be less than chunk_size")
        return v


# --------------------------------------------------------------------------
# Singleton accessor
# --------------------------------------------------------------------------
_settings: Settings | None = None


def get_settings() -> Settings:
    """
    Return the global Settings singleton.

    Loads from environment / .env once, then caches.
    Use this everywhere instead of constructing Settings() directly.
    """
    global _settings  # noqa: PLW0603
    if _settings is None:
        _settings = Settings()
    return _settings
