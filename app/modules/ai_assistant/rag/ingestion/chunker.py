"""Hybrid Chunking 2 tầng (thiết kế §4.1): MarkdownHeader → SemanticChunker (>400 token) → chặn trần 500 token → gắn header ngữ cảnh."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Callable, Literal, Sequence

from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field

SEMANTIC_TRIGGER_TOKENS = 400
HARD_CAP_TOKENS = 500
RECURSIVE_CHUNK_SIZE = 450
RECURSIVE_OVERLAP = 50
HEADER_SEP = "\nNội dung: "


def estimate_tokens(text: str) -> int:
    """Ước lượng ~4 ký tự/token (không cần tokenizer nặng). Dùng thống nhất cho mọi ngưỡng; có thể thay bằng tokenizer thật."""
    return math.ceil(len(text) / 4)


class CorpusRecord(BaseModel):
    """Một dòng .jsonl theo 'Golden Format' (thiết kế §3.3)."""

    external_id: str
    title: str
    document_type: str = "TOPIC_SUMMARY"
    source_name: str
    source_url: str | None = None
    published_at: date | None = None
    language: str | None = None
    aliases: list[str] = Field(default_factory=list)
    categories: list[str] = Field(default_factory=list)
    content: str
    version: str | None = None
    license_info: str | None = None
    # "single": bản ghi ngắn dạng sự kiện (vd. 1 dòng thuốc–thực phẩm) → đúng 1 chunk, không tách theo header.
    chunk_strategy: Literal["sections", "single"] = "sections"
    extra: dict[str, Any] = Field(default_factory=dict)  # ghi vào metadata của chunk (vd. severity)


@dataclass
class Chunk:
    index: int
    body: str
    content: str  # = header ngữ cảnh + body; đây là text được embed và lưu
    metadata: dict[str, Any] = field(default_factory=dict)


def context_header(rec: CorpusRecord) -> str:
    parts = [f"Tài liệu: {rec.title}"]
    if rec.aliases:
        parts.append("Tên khác: " + ", ".join(rec.aliases))
    if rec.categories:
        parts.append("Nhóm: " + ", ".join(rec.categories))
    return "[" + " | ".join(parts) + "]"


class HybridChunker:
    def __init__(
        self,
        semantic_embeddings: Any | None = None,
        *,
        token_counter: Callable[[str], int] = estimate_tokens,
        semantic_trigger: int = SEMANTIC_TRIGGER_TOKENS,
        hard_cap: int = HARD_CAP_TOKENS,
        min_chunk_tokens: int = 0,
    ) -> None:
        """semantic_embeddings: đối tượng có `embed_documents(list[str])` (vd. NomicOllamaEmbeddingAdapter). None → bỏ tầng ngữ nghĩa."""
        self.count, self.trigger, self.cap = token_counter, semantic_trigger, hard_cap
        self.min_tokens = min_chunk_tokens  # >0: gộp các mục quá ngắn liền kề (tổng vẫn ≤ hard cap)
        self._md = MarkdownHeaderTextSplitter(
            headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")], strip_headers=False
        )
        self._recursive = RecursiveCharacterTextSplitter(
            chunk_size=RECURSIVE_CHUNK_SIZE, chunk_overlap=RECURSIVE_OVERLAP, length_function=token_counter
        )
        self._semantic = None
        if semantic_embeddings is not None:
            from langchain_experimental.text_splitter import SemanticChunker  # import trễ

            self._semantic = SemanticChunker(
                semantic_embeddings, breakpoint_threshold_type="percentile", breakpoint_threshold_amount=85
            )

    def split_body(self, content: str) -> list[tuple[str, list[str]]]:
        """Trả về [(đoạn, heading_path)]. Chưa gắn header ngữ cảnh."""
        out: list[tuple[str, list[str]]] = []
        for section in self._md.split_text(content):
            text = section.page_content.strip()
            if not text:
                continue
            path = [section.metadata[k] for k in ("h1", "h2", "h3") if k in section.metadata]
            pieces: Sequence[str] = [text]
            if self.count(text) > self.trigger and self._semantic is not None:
                pieces = [p.strip() for p in self._semantic.split_text(text) if p.strip()] or [text]
            for piece in pieces:
                if self.count(piece) > self.cap:  # Hard Cap Guard
                    out.extend((p.strip(), path) for p in self._recursive.split_text(piece) if p.strip())
                else:
                    out.append((piece, path))
        return self._merge_small(out)

    def _merge_small(self, parts: list[tuple[str, list[str]]]) -> list[tuple[str, list[str]]]:
        if self.min_tokens <= 0 or len(parts) < 2:
            return parts
        merged = [parts[0]]
        for text, path in parts[1:]:
            cur, cur_path = merged[-1]
            small = self.count(cur) < self.min_tokens or self.count(text) < self.min_tokens
            if small and self.count(cur) + self.count(text) <= self.cap:
                merged[-1] = (f"{cur}\n\n{text}", cur_path)
            else:
                merged.append((text, path))
        return merged

    def chunk_document(self, rec: CorpusRecord) -> list[Chunk]:
        header = context_header(rec)
        body_text = rec.content.strip()
        if rec.chunk_strategy == "single" and body_text and self.count(body_text) <= self.cap:
            parts = [(body_text, [])]
        else:
            parts = self.split_body(rec.content)
        chunks: list[Chunk] = []
        for i, (body, path) in enumerate(parts):
            base = {
                "external_id": rec.external_id,
                "source_name": rec.source_name,
                "language": rec.language,
                "heading_path": path,
                "aliases": rec.aliases,
                "categories": rec.categories,
                "token_estimate": self.count(body),
            }
            chunks.append(Chunk(index=i, body=body, content=f"{header}{HEADER_SEP}{body}", metadata={**rec.extra, **base}))
        return chunks
