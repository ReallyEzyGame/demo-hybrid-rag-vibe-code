"""Truy vấn Hybrid (pgvector HNSW + trigram) hợp nhất bằng RRF ngay trong PostgreSQL — thiết kế §5.1."""
from __future__ import annotations

import json
from typing import Sequence
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from ..domain import RetrievedChunk

_ACTIVE_DOC = (
    "d.status = 'ACTIVE' AND (d.effective_from IS NULL OR d.effective_from <= CURRENT_DATE) "
    "AND (d.effective_until IS NULL OR d.effective_until >= CURRENT_DATE)"
)

# Nhánh vector chọn top-k từ knowledge_chunks TRƯỚC (để planner dùng được HNSW), lọc tài liệu hiệu lực ở bước sau.
_HYBRID_SQL = text(
    f"""
WITH vec AS (
    SELECT c.id, c.embedding <=> CAST(CAST(:qvec AS text) AS vector) AS dist
    FROM knowledge_chunks c
    WHERE c.embedding IS NOT NULL
    ORDER BY c.embedding <=> CAST(CAST(:qvec AS text) AS vector)
    LIMIT :vec_k
), vec_r AS (
    SELECT id, dist, ROW_NUMBER() OVER (ORDER BY dist) AS rnk FROM vec
), lex AS (
    SELECT c.id, word_similarity(:q, c.content) AS score
    FROM knowledge_chunks c
    WHERE :q <% c.content
    ORDER BY word_similarity(:q, c.content) DESC
    LIMIT :lex_k
), lex_r AS (
    SELECT id, score, ROW_NUMBER() OVER (ORDER BY score DESC) AS rnk FROM lex
), fused AS (
    SELECT COALESCE(v.id, l.id) AS id,
           COALESCE(1.0 / (:rrf_k + v.rnk), 0) + COALESCE(1.0 / (:rrf_k + l.rnk), 0) AS rrf,
           1 - v.dist AS vector_similarity,
           l.score AS lexical_score
    FROM vec_r v FULL OUTER JOIN lex_r l ON v.id = l.id
)
SELECT c.id AS chunk_id, d.id AS document_id, d.title, d.source_name, d.source_url, c.content, c.metadata AS chunk_metadata,
       f.rrf, f.vector_similarity, f.lexical_score
FROM fused f
JOIN knowledge_chunks c ON c.id = f.id
JOIN knowledge_documents d ON d.id = c.document_id
WHERE {_ACTIVE_DOC}
ORDER BY f.rrf DESC, c.id
LIMIT :top_n
"""
)

_LEXICAL_ONLY_SQL = text(
    f"""
SELECT c.id AS chunk_id, d.id AS document_id, d.title, d.source_name, d.source_url, c.content, c.metadata AS chunk_metadata,
       word_similarity(:q, c.content) AS lexical_score
FROM knowledge_chunks c JOIN knowledge_documents d ON d.id = c.document_id
WHERE :q <% c.content AND {_ACTIVE_DOC}
ORDER BY lexical_score DESC, c.id
LIMIT :top_n
"""
)


def to_pgvector_literal(vec: Sequence[float]) -> str:
    return "[" + ",".join(f"{float(x):.7f}" for x in vec) + "]"


def _row_to_chunk(row, *, rrf: float | None = None) -> RetrievedChunk:  # noqa: ANN001
    m = row._mapping
    meta = m.get("chunk_metadata") or {}
    if isinstance(meta, str):  # asyncpg trả jsonb dạng chuỗi khi đi qua text()
        meta = json.loads(meta)
    return RetrievedChunk(
        chunk_id=m["chunk_id"],
        document_id=m["document_id"],
        title=m["title"],
        source_name=m["source_name"],
        source_url=m["source_url"],
        content=m["content"],
        rrf_score=float(rrf if rrf is not None else m.get("rrf") or 0.0),
        vector_similarity=float(m["vector_similarity"]) if m.get("vector_similarity") is not None else None,
        lexical_score=float(m["lexical_score"]) if m.get("lexical_score") is not None else None,
        metadata=meta,
    )


class KnowledgeRepository:
    def __init__(self, *, vector_k: int = 25, lexical_k: int = 25, rrf_k: int = 60, top_n: int = 15) -> None:
        self.vector_k, self.lexical_k, self.rrf_k, self.top_n = vector_k, lexical_k, rrf_k, top_n

    async def hybrid_search(self, session: AsyncSession, query: str, query_vec: Sequence[float]) -> list[RetrievedChunk]:
        # Chỉ ảnh hưởng transaction hiện tại; giữ ef_search >= vec_k để HNSW trả đủ ứng viên.
        await session.execute(text(f"SET LOCAL hnsw.ef_search = {max(40, self.vector_k)}"))
        rows = await session.execute(
            _HYBRID_SQL,
            {
                "q": query,
                "qvec": to_pgvector_literal(query_vec),
                "vec_k": self.vector_k,
                "lex_k": self.lexical_k,
                "rrf_k": self.rrf_k,
                "top_n": self.top_n,
            },
        )
        return [_row_to_chunk(r) for r in rows]

    async def lexical_search(self, session: AsyncSession, query: str) -> list[RetrievedChunk]:
        """Dùng khi embedding service (Ollama) không khả dụng."""
        rows = await session.execute(_LEXICAL_ONLY_SQL, {"q": query, "top_n": self.top_n})
        out = []
        for i, r in enumerate(rows, start=1):
            out.append(_row_to_chunk(r, rrf=1.0 / (self.rrf_k + i)))
        return out

    async def chunk_exists(self, session: AsyncSession, chunk_id: UUID) -> bool:
        res = await session.execute(text("SELECT 1 FROM knowledge_chunks WHERE id = :i"), {"i": chunk_id})
        return res.first() is not None
