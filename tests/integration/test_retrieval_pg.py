"""Hybrid search (pgvector + trigram + RRF) chạy trên PostgreSQL thật."""

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker

from app.modules.ai_assistant.rag.retrieval import KnowledgeRetriever
from app.modules.ai_assistant.repositories.knowledge_repo import KnowledgeRepository
from tests.support import FakeReranker, HashingEmbeddings

from .conftest import seed_corpus

pytestmark = pytest.mark.integration
Q = "average blood sugar over three months prediabetes A1C"


def retriever(settings, *, rerank_down=False, embeddings=None):
    return KnowledgeRetriever(embeddings or HashingEmbeddings(), KnowledgeRepository(), FakeReranker(rerank_down), settings)


async def run(engine, r, q):
    async with async_sessionmaker(engine)() as s:
        return await r.retrieve(s, q)


async def test_hybrid_search_returns_both_signals_and_ranks_relevant_first(engine, settings):
    await seed_corpus(engine)
    async with async_sessionmaker(engine)() as s:
        cands = await KnowledgeRepository().hybrid_search(s, Q, HashingEmbeddings().embed_query(Q))
    assert cands and "A1C" in cands[0].title
    assert cands[0].vector_similarity is not None and cands[0].rrf_score > 0
    assert any(c.lexical_score for c in cands)  # nhánh trigram cũng đóng góp
    assert cands[0].source_url == "https://medlineplus.gov/a1c.html"


async def test_reranked_result_filters_below_threshold(engine, settings):
    await seed_corpus(engine)
    res = await run(engine, retriever(settings), Q)
    assert res.mode == "RERANKED" and res.chunks and all(c.rerank_score >= settings.min_rerank_score for c in res.chunks)
    assert len(res.chunks) <= settings.final_top_k and "A1C" in res.chunks[0].title


async def test_irrelevant_query_yields_nothing(engine, settings):
    await seed_corpus(engine)
    res = await run(engine, retriever(settings), "quantum chromodynamics lattice gauge")
    assert res.chunks == [] and not res.sufficient


async def test_reranker_down_uses_rrf_but_still_applies_floor(engine, settings):
    await seed_corpus(engine)
    ok = await run(engine, retriever(settings, rerank_down=True), Q)
    assert ok.mode == "RRF_FALLBACK" and ok.chunks and ok.chunks[0].rerank_score is None
    s2 = settings.model_copy(update={"fallback_min_vector_similarity": 0.99, "fallback_min_lexical_score": 0.99})
    none = await run(engine, retriever(s2, rerank_down=True), Q)
    assert none.mode == "RRF_FALLBACK" and none.chunks == []  # không có reranker thì không hạ chuẩn tin cậy


async def test_retired_and_expired_documents_are_never_retrieved(engine, settings):
    await seed_corpus(engine)
    r = retriever(settings)
    await _exec(engine, "UPDATE knowledge_documents SET status = 'RETIRED' WHERE title = 'A1C test'")
    res = await run(engine, r, Q)
    assert all(c.title != "A1C test" for c in res.chunks)
    await _exec(engine, "UPDATE knowledge_documents SET status = 'ACTIVE', effective_until = CURRENT_DATE - 1 WHERE title = 'A1C test'")
    res = await run(engine, r, Q)
    assert all(c.title != "A1C test" for c in res.chunks)
    await _exec(engine, "UPDATE knowledge_documents SET effective_until = CURRENT_DATE + 30 WHERE title = 'A1C test'")
    res = await run(engine, r, Q)
    assert res.chunks and res.chunks[0].title == "A1C test"


async def test_embedding_outage_degrades_to_lexical_only(engine, settings):
    await seed_corpus(engine)
    res = await run(engine, retriever(settings, embeddings=HashingEmbeddings(fail=True)), "acetaminophen paracetamol liver")
    assert res.mode == "LEXICAL_ONLY" and res.chunks and "Acetaminophen" in res.chunks[0].title


async def test_failed_sql_does_not_poison_outer_transaction(engine, settings):
    await seed_corpus(engine)

    class BadRepo(KnowledgeRepository):
        async def hybrid_search(self, session, query, qvec):  # noqa: ANN001, ANN201
            await session.execute(text("SELECT * FROM table_that_does_not_exist"))

    r = KnowledgeRetriever(HashingEmbeddings(), BadRepo(), None, settings)
    async with async_sessionmaker(engine)() as s:
        with pytest.raises(Exception):
            await r.retrieve(s, Q)
        assert (await s.execute(text("SELECT 1"))).scalar_one() == 1  # SAVEPOINT đã rollback, phiên vẫn dùng được


async def _exec(engine, sql):  # noqa: ANN001
    async with engine.begin() as c:
        await c.execute(text(sql))
