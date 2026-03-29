"""Centralised configuration."""

from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Ollama (local LLM — no API key needed)
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "gemma3:4b-it-qat"   # match exactly what `ollama list` shows

    # ChromaDB
    chroma_host: str = "localhost"
    chroma_port: int = 8000

    # Embedding
    embed_model: str = "all-MiniLM-L6-v2"

    # RAG parameters
    max_context_tokens: int = 2000
    coarse_k: int = 20
    fine_k: int = 5
    chunk_size: int = 512
    chunk_overlap: int = 64

    # Evaluation thresholds
    drift_threshold: float = 0.15
    eval_recall_min: float = 0.5
    eval_ndcg_min: float = 0.6

    log_level: str = "INFO"


@lru_cache()
def get_settings() -> Settings:
    return Settings()