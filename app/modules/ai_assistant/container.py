"""Gom các thành phần dùng chung của M3 (khởi tạo một lần lúc khởi động app)."""
from __future__ import annotations

from dataclasses import dataclass

from .config import AiAssistantSettings, get_settings
from .integration import DrugDirectory, DrugEntityResolver, HealthContextProvider, InteractionChecker
from .rag.embeddings.ollama_nomic import EmbeddingsPort, NomicOllamaEmbeddingAdapter
from .rag.generation.lcel_chain import LcelGenerationEngine, build_default_engine
from .rag.guards.input_guard import InputGuard
from .rag.guards.output_guard import OutputGuard
from .rag.reranking.bge_client import BgeRerankerClient, RerankerPort
from .rag.retrieval import KnowledgeRetriever
from .rate_limit import RateLimiter, SlidingWindowRateLimiter
from .repositories.knowledge_repo import KnowledgeRepository
from .services.drug_mentions import DrugMentionExtractor
from .services.interaction_support import InteractionSupport
from .services.personal_context import PersonalContextService


@dataclass
class AiAssistantContainer:
    settings: AiAssistantSettings
    input_guard: InputGuard
    output_guard: OutputGuard
    retriever: KnowledgeRetriever
    engine: LcelGenerationEngine
    personal_context: PersonalContextService
    interactions: InteractionSupport
    chat_limiter: RateLimiter
    explain_limiter: RateLimiter


def build_container(
    *,
    health_context_provider: HealthContextProvider,
    interaction_checker: InteractionChecker,
    drug_resolver: DrugEntityResolver,
    drug_directory: DrugDirectory,
    settings: AiAssistantSettings | None = None,
    embeddings: EmbeddingsPort | None = None,
    reranker: RerankerPort | None = None,
    engine: LcelGenerationEngine | None = None,
    chat_limiter: RateLimiter | None = None,
    explain_limiter: RateLimiter | None = None,
) -> AiAssistantContainer:
    """Điểm nối duy nhất với M1/M2. Các tham số tùy chọn cho phép thay thế khi test (fake LLM, embeddings…)."""
    s = settings or get_settings()
    embeddings = embeddings or NomicOllamaEmbeddingAdapter(s.ollama_base_url, s.embedding_model)
    if reranker is None and s.reranker_url:
        reranker = BgeRerankerClient(s.reranker_url, s.reranker_timeout_s)
    repo = KnowledgeRepository(vector_k=s.vector_top_k, lexical_k=s.lexical_top_k, rrf_k=s.rrf_k, top_n=s.rrf_top_n)
    return AiAssistantContainer(
        settings=s,
        input_guard=InputGuard(),
        output_guard=OutputGuard(),
        retriever=KnowledgeRetriever(embeddings, repo, reranker, s),
        engine=engine or build_default_engine(s),
        personal_context=PersonalContextService(health_context_provider, s),
        interactions=InteractionSupport(
            interaction_checker, drug_directory, DrugMentionExtractor(drug_resolver, s.max_drug_candidates), s
        ),
        chat_limiter=chat_limiter or SlidingWindowRateLimiter(s.rate_limit_per_minute),
        explain_limiter=explain_limiter or SlidingWindowRateLimiter(s.rate_limit_per_minute),
    )
