"""Orchestrator luồng chat: Input Guardrail → RAG (+M1 context, +M2 rule engine) → LLM → Output Guardrail → lưu DB."""
from __future__ import annotations

import logging
from datetime import timedelta
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from ..container import AiAssistantContainer
from ..domain import AnswerOutcome, EvidenceItem, InteractionContext, PersonalContext, RetrievedChunk, make_query_excerpt
from ..enums import AnswerStatus, CitationType, ContextMode, MessageRole, WarningLevel, warning_from_db
from ..errors import AppError, dependency_unavailable, not_found
from ..integration import Principal
from ..models import ChatMessage, utcnow
from ..rag.generation.lcel_chain import GenerationUnavailable
from ..rag.guards.input_guard import Hint
from ..rag.guards.messages import AI_BUSY_MESSAGE, INSUFFICIENT_EVIDENCE_MESSAGE, PERSONAL_UNAVAILABLE_NOTE
from ..repositories.chat_repo import ChatRepository, MessageView
from ..schemas.chat_dto import (
    AnswerOut,
    AssistantMessageOut,
    CitationOut,
    FeedbackSummary,
    MessageListOut,
    Pagination,
    SendMessageOut,
    SessionListOut,
    SessionOut,
    UserMessageOut,
)

logger = logging.getLogger(__name__)
_HEADER_SPLIT = "\nNội dung: "
_SHORT_FOLLOWUP_WORDS = 6
_HISTORY_CHARS = 500


def strip_context_header(content: str) -> str:
    """Bỏ dòng '[Tài liệu: … | Tên khác: …]' mà chunker gắn đầu đoạn, chỉ giữ nội dung để trích dẫn."""
    if content.startswith("[Tài liệu:") and _HEADER_SPLIT in content:
        return content.split(_HEADER_SPLIT, 1)[1]
    return content


def retrieval_query(content: str, history: list[ChatMessage]) -> str:
    """Câu hỏi nối tiếp rất ngắn ('còn liều dùng thì sao?') được ghép với câu hỏi trước để truy xuất đúng chủ đề."""
    if len(content.split()) <= _SHORT_FOLLOWUP_WORDS:
        for m in reversed(history):
            if m.role == MessageRole.USER.value:
                return f"{m.content} {content}"[:1000]
    return content


def _history_text(history: list[ChatMessage]) -> str:
    if not history:
        return "(Chưa có)"
    return "\n".join(
        f"{'Người dùng' if m.role == MessageRole.USER.value else 'Trợ lý'}: {m.content[:_HISTORY_CHARS]}" for m in history
    )


def chunk_evidence(chunks: list[RetrievedChunk], start: int, query: str | None = None) -> list[EvidenceItem]:
    out = []
    for i, c in enumerate(chunks, start=start):
        body = strip_context_header(c.content)
        out.append(
            EvidenceItem(
                i, CitationType.MEDICAL_DOCUMENT, f"{c.source_name} — {c.title}", c.content, make_query_excerpt(body, query),
                source_url=c.source_url, document_id=c.document_id, chunk_id=c.chunk_id,
                severity_hint=(c.metadata or {}).get("severity"),
            )
        )
    return out


def render_evidence(items: list[EvidenceItem]) -> str:
    if not items:
        return "(Không có)"
    parts = []
    for e in items:
        url = f" ({e.source_url})" if e.source_url else ""
        parts.append(f"[{e.citation_id}] Nguồn: {e.source_title}{url}\nNội dung: {e.text}")
    return "\n\n".join(parts)


def view_to_dto(v: MessageView):
    m = v.message
    if m.role == MessageRole.USER.value:
        return UserMessageOut(id=m.id, session_id=m.session_id, content=m.content, created_at=m.created_at)
    fb = (
        FeedbackSummary(rating=v.feedback.rating, comment=v.feedback.comment, updated_at=v.feedback.updated_at)
        if v.feedback
        else None
    )
    answer = AnswerOut(
        content=m.content,
        answer_status=m.answer_status or AnswerStatus.INSUFFICIENT_EVIDENCE.value,
        warning_level=warning_from_db(m.warning_level),
        citations=[
            CitationOut(
                citation_id=c.citation_index, citation_type=c.citation_type, document_id=c.document_id, chunk_id=c.chunk_id,
                drug_id=c.drug_id, source_title=c.source_title, source_url=c.source_url, excerpt=c.excerpt,
            )
            for c in v.citations
        ],
        personal_context_used=list(m.context_used or []),
        suggested_followups=list(m.suggested_followups or []),
        feedback=fb,
    )
    return AssistantMessageOut(id=m.id, session_id=m.session_id, created_at=m.created_at, model_used=m.model_name, answer=answer)


class ChatService:
    def __init__(self, db: AsyncSession, container: AiAssistantContainer) -> None:
        self.db, self.c = db, container
        self.repo = ChatRepository(db)

    # ------------------------------------------------------------ sessions
    async def create_session(self, principal: Principal, title: str | None, mode: ContextMode) -> SessionOut:
        chat = await self.repo.create_session(principal.user_id, title, mode)
        await self.db.commit()
        return SessionOut.model_validate(chat)

    async def list_sessions(self, principal: Principal, page: int, page_size: int) -> SessionListOut:
        items, total = await self.repo.list_sessions(principal.user_id, page, page_size)
        return SessionListOut(items=[SessionOut.model_validate(i) for i in items], pagination=Pagination(page=page, page_size=page_size, total=total))

    async def delete_session(self, principal: Principal, session_id: UUID) -> None:
        if not await self.repo.delete_session(principal.user_id, session_id):
            raise not_found("Phiên trò chuyện")
        await self.db.commit()

    async def list_messages(self, principal: Principal, session_id: UUID, page: int, page_size: int) -> MessageListOut:
        if await self.repo.get_session(principal.user_id, session_id) is None:
            raise not_found("Phiên trò chuyện")
        views, total = await self.repo.list_message_views(principal.user_id, session_id, page, page_size)
        return MessageListOut(items=[view_to_dto(v) for v in views], pagination=Pagination(page=page, page_size=page_size, total=total))

    # ------------------------------------------------------------ send message
    async def send_message(
        self, principal: Principal, session_id: UUID, content: str, override: ContextMode | None
    ) -> SendMessageOut:
        chat = await self.repo.get_session(principal.user_id, session_id)
        if chat is None:
            raise not_found("Phiên trò chuyện")
        mode = override or ContextMode(chat.context_mode)
        history = await self.repo.recent_messages(chat.id, self.c.settings.history_turns * 2)

        t_user = utcnow()
        user_msg = await self.repo.add_user_message(chat.id, content, at=t_user)
        outcome = await self._answer(principal, mode, content, history)
        t_asst = max(utcnow(), t_user + timedelta(microseconds=1))
        asst_msg = await self.repo.add_assistant_message(chat.id, outcome, at=t_asst)
        await self.repo.touch_session(
            chat, title=(content[:50].strip() if not chat.title else None), mode=override, at=t_asst
        )
        await self.db.commit()

        views = await self.repo.attach_details(principal.user_id, [asst_msg])
        return SendMessageOut(
            user_message=view_to_dto(MessageView(user_msg)),  # type: ignore[arg-type]
            assistant_message=view_to_dto(views[0]),  # type: ignore[arg-type]
        )

    async def _answer(self, principal: Principal, mode: ContextMode, content: str, history: list[ChatMessage]) -> AnswerOutcome:
        c = self.c
        # ---- LỚP 1
        decision = c.input_guard.evaluate(content)
        if decision.short_circuit:
            return AnswerOutcome(decision.message, decision.answer_status, decision.warning_level)  # type: ignore[arg-type]

        # ---- Ngữ cảnh cá nhân (M1) + Rule Engine (M2) + Truy xuất tri thức
        personal = await c.personal_context.build(principal, mode, content)
        interaction = await c.interactions.build_for_chat(principal, content, personal, mode)
        rquery = retrieval_query(content, history)
        try:
            retrieval = await c.retriever.retrieve(self.db, rquery)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Knowledge retrieval failed")
            raise dependency_unavailable("Kho tri thức tạm thời không khả dụng") from exc

        interaction_items = interaction.evidence if interaction else []
        docs = chunk_evidence(retrieval.chunks, start=len(interaction_items) + 1, query=rquery)
        evidence = interaction_items + docs

        # ---- Cổng đủ bằng chứng: không có tài liệu và không có kết quả Rule Engine → KHÔNG gọi LLM
        engine_has_result = bool(interaction and interaction.report is not None and not interaction.unavailable)
        if not docs and not engine_has_result:
            return AnswerOutcome(INSUFFICIENT_EVIDENCE_MESSAGE, AnswerStatus.INSUFFICIENT_EVIDENCE, WarningLevel.INFO)

        # ---- LỚP 2: LLM (Gemini → MedGemma)
        hints = []
        if decision.hint is Hint.DIAGNOSIS_REQUEST:
            hints.append("Người dùng đang hỏi bị bệnh gì: KHÔNG kết luận bệnh, đặt answer_status = NEEDS_CLINICIAN, hướng dẫn đi khám.")
        if mode is ContextMode.GENERAL:
            hints.append("Chế độ ẩn danh: không có hồ sơ cá nhân, không suy đoán về người dùng.")
        if personal.unavailable:
            hints.append("Hồ sơ cá nhân hiện không truy cập được: không suy đoán dị ứng hay thuốc đang dùng.")
        variables = {
            "history": _history_text(history),
            "personal_context": personal.prompt_text or "(Không có)",
            "interaction_result": interaction.prompt_text if interaction else "(Câu hỏi không liên quan tương tác thuốc)",
            "evidence": render_evidence(evidence),
            "hints": "\n".join(hints) or "(Không có)",
            "question": content,
        }
        try:
            result = await c.engine.generate("chat", variables)
        except GenerationUnavailable:
            return AnswerOutcome(AI_BUSY_MESSAGE, AnswerStatus.INSUFFICIENT_EVIDENCE, WarningLevel.INFO)

        # ---- LỚP 3
        guarded = c.output_guard.validate(
            result.parsed, evidence, interaction, diagnosis_hint=decision.hint is Hint.DIAGNOSIS_REQUEST
        )
        text = guarded.content
        answered = guarded.answer_status in (AnswerStatus.ANSWERED, AnswerStatus.PARTIAL, AnswerStatus.CONFLICTING_EVIDENCE, AnswerStatus.NEEDS_CLINICIAN)
        if personal.unavailable and answered:
            text = f"{text}\n\n{PERSONAL_UNAVAILABLE_NOTE}"
        used = personal.sections_used if answered else []
        if guarded.violations:
            logger.info("Output guard violations: %s", guarded.violations)
        return AnswerOutcome(
            text, guarded.answer_status, guarded.warning_level, guarded.citations, used, guarded.followups, result.model_used, guarded.violations
        )
