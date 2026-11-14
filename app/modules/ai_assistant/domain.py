"""Kiểu dữ liệu nội bộ chạy xuyên pipeline (không phải DTO của API)."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from uuid import UUID

from .enums import AnswerStatus, CitationType, WarningLevel
from .integration import InteractionReport

EXCERPT_MAX = 300


def make_excerpt(text: str, limit: int = EXCERPT_MAX) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


_UNIT_SPLIT = re.compile(r"\n+|(?<=[.!?])\s+")
_WORD = re.compile(r"\w{3,}", re.UNICODE)
_STOP = frozenset("the and for with what that this are you your not can how does from about have has was were will".split())


def make_query_excerpt(text: str, query: str | None, limit: int = EXCERPT_MAX) -> str:
    """Trích đoạn BẮT ĐẦU từ câu khớp câu hỏi nhất (thay vì luôn lấy đầu chunk), để đoạn hiển thị thực sự chứa bằng chứng."""
    units = [u.strip() for u in _UNIT_SPLIT.split(text) if u.strip()]
    q = {w for w in _WORD.findall((query or "").lower()) if w not in _STOP}
    if not units or not q:
        return make_excerpt(text, limit)
    scores = [len(q & set(_WORD.findall(u.lower()))) for u in units]
    best = max(range(len(units)), key=lambda i: (scores[i], -i))  # hòa điểm → câu đứng trước
    if scores[best] == 0:
        return make_excerpt(text, limit)
    start = best - 1 if best > 0 and units[best - 1].startswith("#") else best  # giữ tiêu đề mục ngay trước câu
    return make_excerpt(" ".join(units[start:]), limit)


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: UUID
    document_id: UUID
    title: str
    source_name: str
    source_url: str | None
    content: str
    rrf_score: float
    vector_similarity: float | None = None
    lexical_score: float | None = None
    rerank_score: float | None = None
    metadata: dict = field(default_factory=dict)


@dataclass
class EvidenceItem:
    """Một mục bằng chứng đánh số [citation_id] trong prompt và trong câu trả lời."""

    citation_id: int
    citation_type: CitationType
    source_title: str
    text: str
    excerpt: str
    source_url: str | None = None
    document_id: UUID | None = None
    chunk_id: UUID | None = None
    drug_id: UUID | None = None
    severity_hint: str | None = None  # vd. 'RED' từ dữ liệu thuốc–thực phẩm


@dataclass
class PersonalContext:
    sections_used: list[str] = field(default_factory=list)
    prompt_text: str = ""
    active_drug_ids: list[UUID] = field(default_factory=list)
    active_drug_names: dict[UUID, str] = field(default_factory=dict)
    unavailable: bool = False

    @property
    def is_empty(self) -> bool:
        return not self.sections_used


@dataclass
class InteractionContext:
    """Kết quả Rule Engine M2 đã chuẩn bị cho prompt. `unavailable=True` khi M2 lỗi."""

    report: InteractionReport | None = None
    prompt_text: str = ""
    evidence: list[EvidenceItem] = field(default_factory=list)
    unavailable: bool = False
    asked: bool = True  # câu hỏi có thực sự liên quan tương tác thuốc
    labels: dict[str, str] = field(default_factory=dict)  # selection_id → tên thuốc hiển thị

    @property
    def statuses(self) -> set[str]:
        if not self.report:
            return set()
        return {p.status for p in self.report.pairs}

    @property
    def has_unresolved_gap(self) -> bool:
        """True nếu có cặp chưa kết luận được (không phải NO_KNOWN... hay INTERACTION_FOUND)."""
        if self.unavailable or not self.report:
            return True
        return any(p.status in ("INSUFFICIENT_DATA", "UNRESOLVED_DRUG") for p in self.report.pairs)


@dataclass
class AnswerOutcome:
    content: str
    answer_status: AnswerStatus
    warning_level: WarningLevel
    citations: list[EvidenceItem] = field(default_factory=list)
    personal_context_used: list[str] = field(default_factory=list)
    suggested_followups: list[str] = field(default_factory=list)
    model_used: str | None = None
    violations: list[str] = field(default_factory=list)
