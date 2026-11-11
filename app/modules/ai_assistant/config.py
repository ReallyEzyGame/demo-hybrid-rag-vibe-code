"""Cấu hình M3. Giá trị mặc định lấy từ Technical Design v2.0; mục ghi (cần benchmark) chưa được chốt."""
from __future__ import annotations

from functools import lru_cache

from pydantic import AliasChoices, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class AiAssistantSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", populate_by_name=True)

    # --- Hạ tầng ---
    database_url: str | None = None  # chỉ dùng cho CLI nạp dữ liệu, dạng postgresql+asyncpg://...
    ollama_base_url: str = "http://localhost:11434"

    # --- Embedding (§4.2) ---
    embedding_model: str = "nomic-embed-text-v2"
    embedding_dim: int = 768

    # --- Hybrid retrieval (§5.1) ---
    vector_top_k: int = 25
    lexical_top_k: int = 25
    rrf_k: int = 60
    rrf_top_n: int = 15
    final_top_k: int = 5

    # --- Reranker (§5.2) ---
    reranker_url: str | None = Field(
        default=None, validation_alias=AliasChoices("KAGGLE_RERANKER_NGROK_URL", "RERANKER_URL")
    )
    reranker_timeout_s: float = 3.0
    min_rerank_score: float = 0.35  # (cần benchmark — Phong)
    # Khi reranker lỗi, dùng điểm gốc để vẫn chặn đoạn kém liên quan (cần benchmark):
    fallback_min_vector_similarity: float = 0.50
    fallback_min_lexical_score: float = 0.30

    # --- LLM (§7) ---
    gemini_api_key: SecretStr | None = Field(
        default=None, validation_alias=AliasChoices("GEMINI_API_KEY", "GOOGLE_API_KEY")
    )
    primary_llm_model: str = "gemini-3.1-flash-lite"
    primary_llm_timeout_s: float = 8.0
    fallback_llm_model: str = "medgemma"
    fallback_llm_timeout_s: float = 15.0
    kaggle_medgemma_ngrok_url: str | None = None
    llm_temperature: float = 0.1

    # --- API / an toàn ---
    max_message_chars: int = 2000
    rate_limit_per_minute: int = 15
    history_turns: int = 6
    max_selected_drugs: int = 10
    max_drug_candidates: int = 12
    stale_vitals_days: int = 30
    emergency_hotline: str = "115"


@lru_cache
def get_settings() -> AiAssistantSettings:
    return AiAssistantSettings()
