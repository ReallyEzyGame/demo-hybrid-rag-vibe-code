"""Bộ điều phối truy xuất: Hybrid RRF → Rerank (timeout 3s) → ngưỡng tin cậy. Thiết kế §5."""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field, replace

from sqlalchemy.ext.asyncio import AsyncSession

from ..config import AiAssistantSettings
from ..domain import RetrievedChunk
from ..repositories.knowledge_repo import KnowledgeRepository
from .embeddings.ollama_nomic import EmbeddingsPort
from .reranking.bge_client import RerankerPort, RerankerUnavailable

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    chunks: list[RetrievedChunk] = field(default_factory=list)
    mode: str = "NONE"  # RERANKED | RRF_FALLBACK | LEXICAL_ONLY | NONE
    candidate_count: int = 0

    @property
    def sufficient(self) -> bool:
        return bool(self.chunks)


class KnowledgeRetriever:
    def __init__(
        self,
        embeddings: EmbeddingsPort,
        repo: KnowledgeRepository,
        reranker: RerankerPort | None,
        settings: AiAssistantSettings,
    ) -> None:
        self.embeddings, self.repo, self.reranker, self.s = embeddings, repo, reranker, settings

    async def retrieve(self, session: AsyncSession, query: str) -> RetrievalResult:
        # 1) Embedding + hybrid; nếu embedding lỗi → chỉ tìm từ khóa (không làm sập luồng chat)
        embed_failed = False
        qvec = None
        try:
            qvec = await self.embeddings.aembed_query(query)
        except Exception as exc:  # noqa: BLE001 - lỗi Ollama/mạng
            if isinstance(exc, asyncio.CancelledError):
                raise
            logger.warning("Embedding failed (%s) → lexical only", type(exc).__name__)
            embed_failed = True
        # SAVEPOINT: lỗi SQL ở bước truy xuất không làm hỏng transaction chứa tin nhắn của request.
        async with session.begin_nested():
            if embed_failed:
                candidates = await self.repo.lexical_search(session, query)
            else:
                candidates = await self.repo.hybrid_search(session, query, qvec)
        if not candidates:
            return RetrievalResult(mode="NONE")

        # 2) Rerank; lỗi/timeout → lấy Top-K từ RRF nhưng vẫn lọc bằng điểm gốc
        if self.reranker is not None and not embed_failed:
            try:
                scores = await asyncio.wait_for(
                    self.reranker.score(query, [c.content for c in candidates]), timeout=self.s.reranker_timeout_s + 0.5
                )
                scored = [replace(c, rerank_score=s) for c, s in zip(candidates, scores, strict=True)]
                scored.sort(key=lambda c: (-(c.rerank_score or 0.0), -c.rrf_score))
                kept = [c for c in scored if (c.rerank_score or 0.0) >= self.s.min_rerank_score][: self.s.final_top_k]
                return RetrievalResult(kept, "RERANKED", len(candidates))
            except (RerankerUnavailable, asyncio.TimeoutError):
                logger.warning("Reranker down/timeout → RRF fallback")

        mode = "LEXICAL_ONLY" if embed_failed else "RRF_FALLBACK"
        kept = [
            c
            for c in candidates
            if (c.vector_similarity is not None and c.vector_similarity >= self.s.fallback_min_vector_similarity)
            or (c.lexical_score is not None and c.lexical_score >= self.s.fallback_min_lexical_score)
        ][: self.s.final_top_k]
        return RetrievalResult(kept, mode, len(candidates))
