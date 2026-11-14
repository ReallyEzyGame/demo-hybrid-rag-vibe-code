"""Nạp đúng 2 file JSON của nhóm (bản mẫu) rồi hỏi qua API."""
from pathlib import Path

import pytest
from sqlalchemy import text

from app.modules.ai_assistant.rag.ingestion.adapters import read_records
from app.modules.ai_assistant.rag.ingestion.chunker import HybridChunker
from app.modules.ai_assistant.rag.ingestion.cli import IngestReport, ingest_records
from tests.support import HashingEmbeddings

pytestmark = pytest.mark.integration
SAMPLES = Path(__file__).resolve().parents[2] / "data" / "sample"
BASE = "/api/v1/ai"


async def ingest_json_samples(engine, min_tokens=80):  # noqa: ANN001, ANN201
    rep = IngestReport()
    for name in ("nutrition_diet_guidelines_sample.json", "drug_food_interaction_sample.json"):
        await ingest_records(read_records(SAMPLES / name, rep), engine, HashingEmbeddings(), HybridChunker(None, min_chunk_tokens=min_tokens), report=rep)
    assert rep.failed == [] and rep.invalid == []
    return rep


async def test_both_files_land_in_db_with_severity_and_urls(engine):
    rep = await ingest_json_samples(engine)
    assert rep.documents == 4
    async with engine.connect() as c:
        rows = (await c.execute(text("SELECT d.title, d.source_url, d.document_type, d.external_id, count(*) n, max(k.metadata->>'severity') sev "
                                     "FROM knowledge_documents d JOIN knowledge_chunks k ON k.document_id = d.id GROUP BY 1,2,3,4 ORDER BY 4"))).all()
    by = {r.external_id: r for r in rows}
    assert by["1558"].source_url == "https://medlineplus.gov/alcohol.html" and by["1558"].n == 6
    assert by["dfi-1"].n == 1 and by["dfi-1"].sev == "RED" and by["dfi-1"].source_url is None and by["dfi-2"].sev == "GREEN"
    await ingest_json_samples(engine)  # nạp lại: không nhân đôi
    async with engine.connect() as c:
        assert (await c.execute(text("SELECT count(*) FROM knowledge_documents"))).scalar_one() == 4
        assert (await c.execute(text("SELECT count(*) FROM knowledge_chunks"))).scalar_one() == rep.chunks


async def test_red_drug_food_chunk_raises_warning_level_end_to_end(hx, engine, user_id):
    await ingest_json_samples(engine)
    sid = await hx.new_session(user_id, "GENERAL")
    r = await hx.ask(user_id, sid, "Dextromethorphan alcohol drinks liver damage warning")
    a = r.json()["assistant_message"]["answer"]
    assert r.status_code == 200 and a["answer_status"] == "ANSWERED"
    assert a["warning_level"] == "WARNING"  # LLM giả trả INFO; bằng chứng RED được trích dẫn → nâng cấp
    c = a["citations"][0]
    assert c["source_title"] == "FDA drug labeling — Dextromethorphan — rượu bia" and c["source_url"] is None
    assert "Tài liệu:" not in c["excerpt"] and "alcohol" in c["excerpt"].lower()


async def test_green_drug_food_row_stays_info(hx, engine, user_id):
    await ingest_json_samples(engine)
    sid = await hx.new_session(user_id, "GENERAL")
    a = (await hx.ask(user_id, sid, "Cefpodoxime tablets administered orally with food to enhance absorption")).json()["assistant_message"]["answer"]
    assert a["warning_level"] == "INFO" and "Cefpodoxim" in a["citations"][0]["source_title"]


async def test_nutrition_topic_is_cited_with_its_medlineplus_url(hx, engine, user_id):
    await ingest_json_samples(engine)
    sid = await hx.new_session(user_id, "GENERAL")
    a = (await hx.ask(user_id, sid, "What is a standard drink? grams of pure alcohol in beer wine")).json()["assistant_message"]["answer"]
    c = a["citations"][0]
    assert c["source_url"] == "https://medlineplus.gov/alcohol.html" and "14 grams" in c["excerpt"]
