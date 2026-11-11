"""DTO request/response của 7 endpoint M3 (VitaP_API_Docs.md §9.3). Request cấm trường lạ (vd. user_id)."""
from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from ..enums import AnswerStatus, CitationType, ContextMode, FeedbackRating, InteractionStatus, WarningLevel

MAX_CONTENT = 2000


class _Req(BaseModel):
    model_config = ConfigDict(extra="forbid")


class _Res(BaseModel):
    model_config = ConfigDict(from_attributes=True)


# ---- Sessions
class SessionCreateRequest(_Req):
    title: Annotated[str, StringConstraints(strip_whitespace=True, max_length=255)] | None = None
    context_mode: ContextMode = ContextMode.PERSONAL


class SessionOut(_Res):
    id: UUID
    title: str | None
    context_mode: ContextMode
    created_at: datetime
    updated_at: datetime


class Pagination(BaseModel):
    page: int
    page_size: int
    total: int


class SessionListOut(BaseModel):
    items: list[SessionOut]
    pagination: Pagination


# ---- Messages
class MessageCreateRequest(_Req):
    content: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=MAX_CONTENT)]
    context_mode_override: ContextMode | None = None


class CitationOut(BaseModel):
    citation_id: int
    citation_type: CitationType
    document_id: UUID | None = None
    chunk_id: UUID | None = None
    drug_id: UUID | None = None
    source_title: str | None = None
    source_url: str | None = None
    excerpt: str | None = None


class FeedbackSummary(BaseModel):
    rating: FeedbackRating
    comment: str | None = None
    updated_at: datetime | None = None


class AnswerOut(BaseModel):
    content: str
    answer_status: AnswerStatus
    warning_level: WarningLevel
    citations: list[CitationOut] = Field(default_factory=list)
    personal_context_used: list[str] = Field(default_factory=list)
    suggested_followups: list[str] = Field(default_factory=list)
    feedback: FeedbackSummary | None = None


class UserMessageOut(BaseModel):
    id: UUID
    session_id: UUID
    role: Literal["USER"] = "USER"
    content: str
    created_at: datetime


class AssistantMessageOut(BaseModel):
    id: UUID
    session_id: UUID
    role: Literal["ASSISTANT"] = "ASSISTANT"
    created_at: datetime
    model_used: str | None = None
    answer: AnswerOut


class SendMessageOut(BaseModel):
    user_message: UserMessageOut
    assistant_message: AssistantMessageOut


class MessageListOut(BaseModel):
    items: list[Annotated[UserMessageOut | AssistantMessageOut, Field(discriminator="role")]]
    pagination: Pagination


# ---- Feedback
class FeedbackRequest(_Req):
    rating: FeedbackRating
    comment: Annotated[str, StringConstraints(strip_whitespace=True, max_length=500)] | None = None


class FeedbackOut(BaseModel):
    message_id: UUID
    rating: FeedbackRating
    comment: str | None = None
    updated_at: datetime


# ---- Explain interaction
class ExplainInteractionRequest(_Req):
    drug_a_id: UUID
    drug_b_id: UUID
    include_personal_context: bool = False


class ExplainInteractionOut(BaseModel):
    interaction_status: InteractionStatus | str
    severity: str | None = None
    warning_level: WarningLevel
    explanation: str
    clinical_management: str | None = None
    citations: list[CitationOut] = Field(default_factory=list)
    model_used: str
