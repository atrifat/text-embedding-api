# config.py
from functools import lru_cache
from typing import List

from pydantic import ConfigDict, Field
from pydantic_settings import BaseSettings


class AppSettings(BaseSettings):
    """
    Application settings loaded from environment variables.
    """

    cuda_cache_clear_enabled: bool = Field(
        True,
        json_schema_extra={"env": "CUDA_CACHE_CLEAR_ENABLED"},
        description="Enable CUDA cache clearing after each batch.",
    )
    default_model: str = Field(
        "text-embedding-3-large",
        json_schema_extra={"env": "DEFAULT_MODEL"},
        description="Default embedding model to use.",
    )
    warmup_enabled: bool = Field(
        True, json_schema_extra={"env": "WARMUP_ENABLED"}, description="Enable model warmup on startup."
    )
    app_port: int = Field(7860, json_schema_extra={"env": "APP_PORT"}, description="Port for the FastAPI application.")
    app_host: str = Field(
        "0.0.0.0", json_schema_extra={"env": "APP_HOST"}, description="Host for the FastAPI application."
    )
    embedding_batch_size: int = Field(
        8, json_schema_extra={"env": "EMBEDDING_BATCH_SIZE"}, description="Batch size for embedding generation."
    )
    embeddings_cache_enabled: bool = Field(
        True, json_schema_extra={"env": "EMBEDDINGS_CACHE_ENABLED"}, description="Enable in-memory embeddings cache."
    )
    report_cached_tokens: bool = Field(
        False,
        json_schema_extra={"env": "REPORT_CACHED_TOKENS"},
        description="Report token count for cached embeddings.",
    )
    embeddings_cache_maxsize: int = Field(
        2048, json_schema_extra={"env": "EMBEDDINGS_CACHE_MAXSIZE"}, description="Maximum size of the embeddings cache."
    )
    environment: str = Field(
        "development",
        json_schema_extra={"env": "ENVIRONMENT"},
        description="Application environment (e.g., 'production', 'development').",
    )
    allowed_origins: List[str] = Field(
        ["*"],
        json_schema_extra={"env": "ALLOWED_ORIGINS"},
        description=(
            "List of allowed origins for CORS. Use comma-separated values in .env "
            "(e.g., 'http://localhost:3000,https://example.com')."
        ),
    )

    model_config = ConfigDict(env_file=".env")


@lru_cache()
def get_app_settings():
    return AppSettings()
