from __future__ import annotations

import os
from pathlib import Path
from uuid import UUID, uuid4

import asyncpg
import httpx
import pytest
import pytest_asyncio
from fastapi import FastAPI, Header
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.modules.ai_assistant import build_container, create_ai_assistant_router, register_exception_handlers
from app.modules.ai_assistant.config import AiAssistantSettings
from app.modules.ai_assistant.enums import CONSENT_SCOPE
from app.modules.ai_assistant.errors import auth_required
from app.modules.ai_assistant.rag.generation.lcel_chain import LcelGenerationEngine
from app.modules.ai_assistant.rag.ingestion.chunker import CorpusRecord, HybridChunker
from app.modules.ai_assistant.rag.ingestion.cli import ingest_records
from tests.support import (
    FakeDirectory, FakeHealthProvider, FakeInteractionChecker, FakePrincipal, FakeReranker, FakeResolver, HashingEmbeddings,
    ScriptedChatModel, kind_responder, sample_health,
)

DB_URL = os.getenv("M3_TEST_DATABASE_URL")
ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.integration

_schema_ready = False


def pytest_collection_modifyitems(config, items):  # noqa: ANN001
    if DB_URL:
        return
    skip = pytest.mark.skip(reason="Đặt M3_TEST_DATABASE_URL (postgresql+asyncpg://…) để chạy test tích hợp")
    for it in items:
        if "integration" in str(it.fspath):
            it.add_marker(skip)


async def _apply_schema() -> None:
    global _schema_ready
    if _schema_ready:
        return
    conn = await asyncpg.connect(DB_URL.replace("postgresql+asyncpg://", "postgresql://"))
    try:
        await conn.execute("DROP SCHEMA public CASCADE; CREATE SCHEMA public;")
        await conn.execute((ROOT / "tests/integration/schema_v3_base.sql").read_text())
        await conn.execute((ROOT / "migrations/m3_schema_delta.sql").read_text())
    finally:
        await conn.close()
    _schema_ready = True


@pytest_asyncio.fixture
async def engine():
    await _apply_schema()
    eng = create_async_engine(DB_URL)
    async with eng.begin() as c:
        await c.execute(text("TRUNCATE ai_feedback, message_citations, chat_messages, chat_sessions, knowledge_chunks, "
                             "knowledge_documents, user_consents, users RESTART IDENTITY CASCADE"))
    yield eng
    await eng.dispose()


@pytest.fixture
def sessionmaker_(engine):
    return async_sessionmaker(engine, expire_on_commit=False)


@pytest_asyncio.fixture
async def user_id(engine) -> UUID:
    uid = uuid4()
    async with engine.begin() as c:
        await c.execute(text("INSERT INTO users (id) VALUES (:i)"), {"i": uid})
        await c.execute(text("INSERT INTO user_consents (user_id, consent_type) VALUES (:i, :t)"), {"i": uid, "t": CONSENT_SCOPE})
    return uid


async def make_user(engine, *, consent: bool = True, revoked: bool = False) -> UUID:
    uid = uuid4()
    async with engine.begin() as c:
        await c.execute(text("INSERT INTO users (id) VALUES (:i)"), {"i": uid})
        if consent:
            await c.execute(
                text("INSERT INTO user_consents (user_id, consent_type, revoked_at) VALUES (:i, :t, CASE WHEN :r THEN now() END)"),
                {"i": uid, "t": CONSENT_SCOPE, "r": revoked},
            )
    return uid


CORPUS = [
    ("a1c", "A1C test", "MedlinePlus", "https://medlineplus.gov/a1c.html", ["HbA1c"], ["diabetes"],
     "# A1C test\nThe A1C test measures your average blood sugar level over the past three months. "
     "An A1C between 5.7 and 6.4 percent means prediabetes. An A1C of 6.5 percent or higher on two separate tests means diabetes."),
    ("paracetamol", "Acetaminophen", "MedlinePlus", "https://medlineplus.gov/druginfo/acetaminophen.html", ["Paracetamol"], ["pain relief"],
     "# Acetaminophen\nAcetaminophen paracetamol is used to relieve mild to moderate pain and to reduce fever. "
     "Adults should not take more than 4000 mg of acetaminophen in 24 hours. Overdose can seriously damage the liver."),
    ("warfarin", "Warfarin", "MedlinePlus", "https://medlineplus.gov/druginfo/warfarin.html", ["Coumadin"], ["blood thinner"],
     "# Warfarin\nWarfarin is an anticoagulant blood thinner that prevents blood clots. Taking warfarin with ibuprofen or aspirin "
     "increases the risk of serious bleeding."),
]


async def seed_corpus(engine) -> None:
    emb = HashingEmbeddings()
    recs = [
        CorpusRecord(external_id=e, title=t, source_name=s, source_url=u, aliases=al, categories=ca, content=c, language="en")
        for e, t, s, u, al, ca, c in CORPUS
    ]
    rep = await ingest_records(iter(recs), engine, emb, HybridChunker(None))
    assert rep.failed == [] and rep.documents == len(recs)


@pytest.fixture
def settings():
    return AiAssistantSettings(_env_file=None, min_rerank_score=0.2, fallback_min_vector_similarity=0.05, fallback_min_lexical_score=0.2)


class Harness:
    """Gom app + các fake để từng test chỉnh hành vi."""

    def __init__(self, engine, settings) -> None:  # noqa: ANN001
        self.settings = settings
        self.embeddings = HashingEmbeddings()
        self.reranker = FakeReranker()
        self.primary = ScriptedChatModel(responder=kind_responder)
        self.fallback = ScriptedChatModel(responder=kind_responder)
        self.health = FakeHealthProvider(sample_health())
        self.checker = FakeInteractionChecker()
        self.resolver = FakeResolver()
        sm = async_sessionmaker(engine, expire_on_commit=False)

        def get_principal(x_test_user: str | None = Header(default=None), x_test_guest: str | None = Header(default=None)):
            if x_test_guest:
                return FakePrincipal(uuid4(), is_guest=True)
            if not x_test_user:
                raise auth_required()
            return FakePrincipal(UUID(x_test_user))

        async def get_db():  # noqa: ANN202
            async with sm() as s:
                yield s

        self.container = build_container(
            health_context_provider=self.health, interaction_checker=self.checker, drug_resolver=self.resolver,
            drug_directory=FakeDirectory(), settings=settings, embeddings=self.embeddings, reranker=self.reranker,
            engine=LcelGenerationEngine.from_models(self.primary, self.fallback, settings),
        )
        self.app = FastAPI()
        register_exception_handlers(self.app)
        self.app.include_router(create_ai_assistant_router(get_principal=get_principal, get_db_session=get_db, container=self.container))
        self.client = httpx.AsyncClient(transport=httpx.ASGITransport(app=self.app, raise_app_exceptions=False), base_url="http://t")

    def h(self, uid: UUID) -> dict[str, str]:
        return {"X-Test-User": str(uid)}

    def set_engine(self, primary, fallback=None) -> None:  # noqa: ANN001
        self.container.engine = LcelGenerationEngine.from_models(primary, fallback, self.settings)
        self.primary, self.fallback = primary, fallback

    async def new_session(self, uid: UUID, mode: str = "PERSONAL") -> str:
        r = await self.client.post("/api/v1/ai/chat/sessions", json={"context_mode": mode}, headers=self.h(uid))
        assert r.status_code == 201, r.text
        return r.json()["id"]

    async def ask(self, uid: UUID, sid: str, content: str, **extra):  # noqa: ANN201
        return await self.client.post(f"/api/v1/ai/chat/sessions/{sid}/messages", json={"content": content, **extra}, headers=self.h(uid))


@pytest_asyncio.fixture
async def hx(engine, settings):
    h = Harness(engine, settings)
    yield h
    await h.client.aclose()
