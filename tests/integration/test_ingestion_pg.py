import json
from uuid import uuid4

import pytest
from sqlalchemy import text

from app.modules.ai_assistant.rag.ingestion import cli
from app.modules.ai_assistant.rag.ingestion.chunker import CorpusRecord, HybridChunker
from app.modules.ai_assistant.rag.ingestion.cli import IngestReport, ingest_records, read_jsonl
from tests.support import HashingEmbeddings

from .conftest import CORPUS, seed_corpus

pytestmark = pytest.mark.integration


def rec(ext="doc-1", content="# T\nHello world.", title="Doc One", **kw):
    return CorpusRecord(external_id=ext, title=title, source_name="Src", content=content, **kw)


async def count(engine, table):  # noqa: ANN001
    async with engine.connect() as c:
        return (await c.execute(text(f"SELECT count(*) FROM {table}"))).scalar_one()


LONG = "\n\n".join(f"Paragraph {i}. " + ("Blood sugar glucose insulin metabolism. " * 12) for i in range(14))  # ≈ 1500+ token


async def test_ingest_stores_header_context_and_metadata(engine):
    await seed_corpus(engine)
    async with engine.connect() as c:
        row = (await c.execute(text("SELECT content, metadata, vector_dims(embedding) d FROM knowledge_chunks c JOIN knowledge_documents d ON d.id=c.document_id WHERE d.title='A1C test'"))).first()
    assert row.content.startswith("[Tài liệu: A1C test | Tên khác: HbA1c | Nhóm: diabetes]\nNội dung: ")
    assert row.d == 768 and row.metadata["external_id"] == "a1c" and row.metadata["language"] == "en"


async def test_reingest_is_idempotent_and_keeps_chunk_ids(engine):
    r = rec(content=LONG)
    rep = await ingest_records(iter([r]), engine, HashingEmbeddings(), HybridChunker(None))
    assert rep.failed == [] and rep.chunks >= 3
    async with engine.connect() as c:
        before = {x.chunk_index: x.id for x in await c.execute(text("SELECT chunk_index, id FROM knowledge_chunks"))}
    await ingest_records(iter([r]), engine, HashingEmbeddings(), HybridChunker(None))
    async with engine.connect() as c:
        after = {x.chunk_index: x.id for x in await c.execute(text("SELECT chunk_index, id FROM knowledge_chunks"))}
    assert before == after and await count(engine, "knowledge_documents") == 1


async def test_skip_existing(engine):
    await seed_corpus(engine)
    recs = [CorpusRecord(external_id=e, title=t, source_name=s, content=c) for e, t, s, _u, _a, _c, c in CORPUS]
    rep = await ingest_records(iter(recs), engine, HashingEmbeddings(), HybridChunker(None), skip_existing=True)
    assert rep.skipped == 3 and rep.documents == 0 and await count(engine, "knowledge_documents") == 3


async def test_shorter_document_drops_stale_chunks_but_keeps_cited_ones(engine):
    await ingest_records(iter([rec(content=LONG)]), engine, HashingEmbeddings(), HybridChunker(None))
    n = await count(engine, "knowledge_chunks")
    assert n >= 3
    uid, sid, mid = uuid4(), uuid4(), uuid4()
    async with engine.begin() as c:
        cited = (await c.execute(text("SELECT id FROM knowledge_chunks WHERE chunk_index = 2"))).scalar_one()
        await c.execute(text("INSERT INTO users (id) VALUES (:u)"), {"u": uid})
        await c.execute(text("INSERT INTO chat_sessions (id, user_id) VALUES (:s, :u)"), {"s": sid, "u": uid})
        await c.execute(text("INSERT INTO chat_messages (id, session_id, role, content) VALUES (:m, :s, 'ASSISTANT', 'x')"), {"m": mid, "s": sid})
        await c.execute(text("INSERT INTO message_citations (message_id, chunk_id, citation_type) VALUES (:m, :c, 'MEDICAL_DOCUMENT')"), {"m": mid, "c": cited})
    rep = await ingest_records(iter([rec(content="# T\nShort now.")]), engine, HashingEmbeddings(), HybridChunker(None))
    assert rep.failed == []
    async with engine.connect() as c:
        idx = sorted(r.chunk_index for r in await c.execute(text("SELECT chunk_index FROM knowledge_chunks")))
    assert idx == [0, 2]  # chunk 1 bị xóa, chunk 2 giữ lại vì đã được trích dẫn


async def test_wrong_embedding_dimension_fails_atomically(engine):
    rep = await ingest_records(iter([rec()]), engine, HashingEmbeddings(dim=8), HybridChunker(None))
    assert len(rep.failed) == 1 and "768" in rep.failed[0] and await count(engine, "knowledge_documents") == 0


async def test_one_bad_document_does_not_stop_the_batch(engine):
    class Flaky(HashingEmbeddings):
        async def aembed_documents(self, texts):  # noqa: ANN001, ANN201
            if any("Doc Bad" in t for t in texts):
                raise ConnectionError("ollama")
            return await super().aembed_documents(texts)

    rep = await ingest_records(iter([rec("a"), rec("b", title="Doc Bad"), rec("c", title="Doc C")]), engine, Flaky(), HybridChunker(None))
    assert rep.documents == 2 and len(rep.failed) == 1 and await count(engine, "knowledge_documents") == 2


async def test_dry_run_writes_nothing_and_invalid_lines_are_reported(engine, tmp_path):
    f = tmp_path / "c.jsonl"
    good = {"external_id": "g1", "title": "Good", "source_name": "S", "content": "# H\nbody text here"}
    f.write_text("\n".join([json.dumps(good), "{not json", json.dumps({"title": "no id"}), ""]), encoding="utf-8")
    rep = IngestReport()
    recs = list(read_jsonl(f, rep))
    assert len(recs) == 1 and len(rep.invalid) == 2 and rep.invalid[0].startswith("dòng 2")
    await ingest_records(iter(recs), engine, HashingEmbeddings(), HybridChunker(None), dry_run=True, report=rep)
    assert rep.documents == 1 and rep.chunks >= 1 and await count(engine, "knowledge_documents") == 0


def test_cli_requires_database_url(tmp_path, monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    f = tmp_path / "x.jsonl"
    f.write_text("")
    assert cli.main(["--input", str(f)]) == 2
