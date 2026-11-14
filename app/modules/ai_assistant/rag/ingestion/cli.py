"""CLI nạp dữ liệu .jsonl đã làm sạch vào PostgreSQL pgvector (thiết kế §3.4).

    python -m app.modules.ai_assistant.rag.ingestion.cli \
        --input nutrition_diet_guidelines.json drug_food_interaction.json corpus.jsonl \
        --batch-size 64 --skip-existing --min-chunk-tokens 80
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from pydantic import ValidationError
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from ...config import AiAssistantSettings, get_settings
from ..embeddings.ollama_nomic import EmbeddingsPort, NomicOllamaEmbeddingAdapter
from ...repositories.knowledge_repo import to_pgvector_literal
from .adapters import FORMATS, read_records
from .chunker import Chunk, CorpusRecord, HybridChunker

logger = logging.getLogger("m3.ingestion")


@dataclass
class IngestReport:
    documents: int = 0
    chunks: int = 0
    skipped: int = 0
    invalid: list[str] = field(default_factory=list)  # "dòng N: lý do"
    failed: list[str] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"documents={self.documents} chunks={self.chunks} skipped={self.skipped} "
            f"invalid={len(self.invalid)} failed={len(self.failed)}"
        )


def read_jsonl(path: Path, report: IngestReport, limit: int | None = None) -> Iterator[CorpusRecord]:
    """Tương thích cũ: đọc .jsonl/.json và tự nhận diện định dạng."""
    return read_records(path, report, limit)


async def _document_exists(conn, rec: CorpusRecord) -> bool:  # noqa: ANN001
    row = await conn.execute(
        text("SELECT 1 FROM knowledge_documents WHERE source_name = :s AND external_id = :e"),
        {"s": rec.source_name, "e": rec.external_id},
    )
    return row.first() is not None


_UPSERT_DOC = text(
    """
INSERT INTO knowledge_documents
    (id, title, document_type, source_name, source_url, version, published_at, status, license_info, external_id)
VALUES (:id, :title, :dtype, :source_name, :source_url, :version, :published_at, 'ACTIVE', :license, :external_id)
ON CONFLICT (source_name, external_id) WHERE external_id IS NOT NULL DO UPDATE SET
    title = EXCLUDED.title, document_type = EXCLUDED.document_type, source_url = EXCLUDED.source_url,
    version = EXCLUDED.version, published_at = EXCLUDED.published_at, license_info = EXCLUDED.license_info
RETURNING id
"""
)
_UPSERT_CHUNK = text(
    """
INSERT INTO knowledge_chunks (id, document_id, chunk_index, content, embedding, metadata)
VALUES (:id, :doc, :idx, :content, CAST(CAST(:emb AS text) AS vector), CAST(:meta AS jsonb))
ON CONFLICT (document_id, chunk_index) DO UPDATE SET
    content = EXCLUDED.content, embedding = EXCLUDED.embedding, metadata = EXCLUDED.metadata
"""
)
# Chunk thừa sau khi tài liệu ngắn đi; giữ lại nếu đã được trích dẫn trong lịch sử chat (FK).
_DELETE_STALE = text(
    """
DELETE FROM knowledge_chunks kc WHERE kc.document_id = :doc AND kc.chunk_index >= :n
AND NOT EXISTS (SELECT 1 FROM message_citations mc WHERE mc.chunk_id = kc.id)
"""
)


async def _embed(chunks: list[Chunk], embeddings: EmbeddingsPort, batch_size: int, dim: int) -> list[list[float]]:
    vectors: list[list[float]] = []
    for i in range(0, len(chunks), batch_size):
        batch = await embeddings.aembed_documents([c.content for c in chunks[i : i + batch_size]])
        vectors.extend(batch)
    if len(vectors) != len(chunks):
        raise RuntimeError("embedding count mismatch")
    bad = [len(v) for v in vectors if len(v) != dim]
    if bad:
        raise RuntimeError(f"embedding dimension {bad[0]} != expected {dim} (cột knowledge_chunks.embedding là vector({dim}))")
    return vectors


async def ingest_records(
    records: Iterator[CorpusRecord],
    engine: AsyncEngine,
    embeddings: EmbeddingsPort,
    chunker: HybridChunker,
    *,
    batch_size: int = 64,
    skip_existing: bool = False,
    dry_run: bool = False,
    dim: int = 768,
    report: IngestReport | None = None,
) -> IngestReport:
    report = report or IngestReport()
    for rec in records:
        try:
            if skip_existing:
                async with engine.connect() as conn:
                    if await _document_exists(conn, rec):
                        report.skipped += 1
                        continue
            chunks = await asyncio.to_thread(chunker.chunk_document, rec)
            if not chunks:
                report.invalid.append(f"{rec.external_id}: nội dung rỗng sau khi chunk")
                continue
            if dry_run:
                report.documents += 1
                report.chunks += len(chunks)
                continue
            vectors = await _embed(chunks, embeddings, batch_size, dim)
            async with engine.begin() as conn:  # một transaction cho mỗi tài liệu: hoặc đủ, hoặc không có gì
                doc_id = (
                    await conn.execute(
                        _UPSERT_DOC,
                        {
                            "id": uuid.uuid4(), "title": rec.title[:500], "dtype": rec.document_type,
                            "source_name": rec.source_name[:255], "source_url": rec.source_url, "version": rec.version,
                            "published_at": rec.published_at, "license": rec.license_info, "external_id": rec.external_id,
                        },
                    )
                ).scalar_one()
                await conn.execute(
                    _UPSERT_CHUNK,
                    [
                        {
                            "id": uuid.uuid4(), "doc": doc_id, "idx": c.index, "content": c.content,
                            "emb": to_pgvector_literal(v), "meta": json.dumps(c.metadata, ensure_ascii=False, default=str),
                        }
                        for c, v in zip(chunks, vectors, strict=True)
                    ],
                )
                await conn.execute(_DELETE_STALE, {"doc": doc_id, "n": len(chunks)})
            report.documents += 1
            report.chunks += len(chunks)
        except Exception as exc:  # noqa: BLE001 - một tài liệu lỗi không dừng cả lô
            logger.exception("Ingest failed for %s", rec.external_id)
            report.failed.append(f"{rec.external_id}: {type(exc).__name__}: {exc}")
    return report


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Nạp corpus .jsonl vào knowledge_documents / knowledge_chunks (pgvector)")
    p.add_argument("--input", required=True, nargs="+", type=Path, help="Một hoặc nhiều file .jsonl/.json")
    p.add_argument("--format", choices=FORMATS, default="auto", help="Định dạng bản ghi (mặc định tự nhận diện theo từng bản ghi)")
    p.add_argument("--min-chunk-tokens", type=int, default=0, help="Gộp các mục ngắn hơn ngưỡng này (khuyến nghị 80 cho MedlinePlus)")
    p.add_argument("--batch-size", type=int, default=64, help="Số chunk embed mỗi lần gọi Ollama")
    p.add_argument("--skip-existing", action="store_true", help="Bỏ qua tài liệu đã có (source_name + external_id)")
    p.add_argument("--dry-run", action="store_true", help="Chỉ chunk và thống kê, không embed, không ghi DB")
    p.add_argument("--limit", type=int, default=None, help="Chỉ xử lý N tài liệu đầu (để thử)")
    p.add_argument("--database-url", default=None, help="postgresql+asyncpg://… (mặc định lấy DATABASE_URL)")
    p.add_argument("--no-semantic", action="store_true", help="Bỏ tầng SemanticChunker (chỉ header + hard cap)")
    return p


async def _amain(args: argparse.Namespace, settings: AiAssistantSettings) -> int:
    url = args.database_url or settings.database_url
    if not url and not args.dry_run:
        print("Thiếu --database-url hoặc biến môi trường DATABASE_URL", file=sys.stderr)
        return 2
    embeddings = NomicOllamaEmbeddingAdapter(settings.ollama_base_url, settings.embedding_model)
    chunker = HybridChunker(None if args.no_semantic else embeddings, min_chunk_tokens=args.min_chunk_tokens)
    engine = create_async_engine(url) if url else None
    report = IngestReport()
    try:
        for path in args.input:
            await ingest_records(
                read_records(path, report, args.limit, args.format), engine, embeddings, chunker,  # type: ignore[arg-type]
                batch_size=args.batch_size, skip_existing=args.skip_existing, dry_run=args.dry_run,
                dim=settings.embedding_dim, report=report,
            )
    finally:
        if engine:
            await engine.dispose()
    print(report.summary())
    for line in (report.invalid + report.failed)[:20]:
        print("  -", line, file=sys.stderr)
    return 1 if report.failed else 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    return asyncio.run(_amain(build_parser().parse_args(argv), get_settings()))


if __name__ == "__main__":
    raise SystemExit(main())
