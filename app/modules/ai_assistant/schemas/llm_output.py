"""Pydantic schema cho Structured Output của LLM (PydanticOutputParser)."""
from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

from ..enums import AnswerStatus, WarningLevel


class LLMGeneratedAnswer(BaseModel):
    content: str = Field(description="Câu trả lời tiếng Việt, mỗi khẳng định y khoa có ký hiệu [n] tương ứng bằng chứng.")
    answer_status: AnswerStatus = Field(
        description="ANSWERED | PARTIAL | INSUFFICIENT_EVIDENCE | CONFLICTING_EVIDENCE | NEEDS_CLINICIAN | EMERGENCY_REDIRECT | OUT_OF_SCOPE"
    )
    warning_level: WarningLevel = Field(default=WarningLevel.INFO, description="INFO | WARNING | CRITICAL")
    cited_ids: list[int] = Field(default_factory=list, description="Danh sách số [n] đã trích dẫn trong content.")
    suggested_followups: list[str] = Field(default_factory=list, description="Tối đa 3 câu hỏi gợi ý tiếp theo.")

    @field_validator("content")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("content rỗng")
        return v

    @field_validator("cited_ids", mode="before")
    @classmethod
    def _ints(cls, v):  # noqa: ANN001, ANN206
        return [int(x) for x in (v or [])]


class LLMExplainAnswer(BaseModel):
    explanation: str = Field(description="Giải thích 2–4 câu, tiếng Việt, dễ hiểu cho người bệnh.")
    clinical_management: str | None = Field(default=None, description="Hướng xử trí lâm sàng ngắn gọn, nếu có trong bằng chứng.")
    cited_ids: list[int] = Field(default_factory=list)

    @field_validator("explanation")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("explanation rỗng")
        return v
