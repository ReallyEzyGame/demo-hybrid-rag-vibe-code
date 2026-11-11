"""CRUD chat_sessions / chat_messages / message_citations / ai_feedback — mọi truy vấn lọc theo user_id của principal."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Sequence
from uuid import UUID, uuid4

from sqlalchemy import delete, func, select, text, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from ..domain import AnswerOutcome
from ..enums import CONSENT_SCOPE, ContextMode, MessageRole, warning_to_db
from ..models import AiFeedback, ChatMessage, ChatSession, MessageCitation, utcnow


@dataclass
class CitationRow:
    citation_index: int
    citation_type: str
    document_id: UUID | None
    chunk_id: UUID | None
    drug_id: UUID | None
    source_title: str | None
    excerpt: str | None
    source_url: str | None


@dataclass
class MessageView:
    message: ChatMessage
    citations: list[CitationRow] = field(default_factory=list)
    feedback: AiFeedback | None = None


class ChatRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.db = session

    # ------------------------------------------------------------ consent
    async def has_active_ai_consent(self, user_id: UUID) -> bool:
        """Bản ghi consent mới nhất của user phải đang hiệu lực (granted, chưa revoke)."""
        row = (
            await self.db.execute(
                text(
                    "SELECT is_granted, revoked_at FROM user_consents "
                    "WHERE user_id = :u AND consent_type = :t ORDER BY granted_at DESC LIMIT 1"
                ),
                {"u": user_id, "t": CONSENT_SCOPE},
            )
        ).first()
        return bool(row and row.is_granted and row.revoked_at is None)

    # ------------------------------------------------------------ sessions
    async def create_session(self, user_id: UUID, title: str | None, mode: ContextMode) -> ChatSession:
        now = utcnow()
        chat = ChatSession(id=uuid4(), user_id=user_id, title=title or None, context_mode=mode.value, created_at=now, updated_at=now)
        self.db.add(chat)
        await self.db.flush()
        return chat

    async def get_session(self, user_id: UUID, session_id: UUID) -> ChatSession | None:
        return (
            await self.db.execute(select(ChatSession).where(ChatSession.id == session_id, ChatSession.user_id == user_id))
        ).scalar_one_or_none()

    async def list_sessions(self, user_id: UUID, page: int, page_size: int) -> tuple[list[ChatSession], int]:
        total = (await self.db.execute(select(func.count()).select_from(ChatSession).where(ChatSession.user_id == user_id))).scalar_one()
        rows = (
            await self.db.execute(
                select(ChatSession)
                .where(ChatSession.user_id == user_id)
                .order_by(ChatSession.updated_at.desc(), ChatSession.id.desc())
                .offset((page - 1) * page_size)
                .limit(page_size)
            )
        ).scalars()
        return list(rows), total

    async def delete_session(self, user_id: UUID, session_id: UUID) -> bool:
        """Xóa thủ công theo thứ tự con→cha để không phụ thuộc cấu hình ON DELETE CASCADE của migration."""
        chat = await self.get_session(user_id, session_id)
        if chat is None:
            return False
        msg_ids = select(ChatMessage.id).where(ChatMessage.session_id == session_id)
        await self.db.execute(delete(AiFeedback).where(AiFeedback.message_id.in_(msg_ids)))
        await self.db.execute(delete(MessageCitation).where(MessageCitation.message_id.in_(msg_ids)))
        await self.db.execute(delete(ChatMessage).where(ChatMessage.session_id == session_id))
        await self.db.execute(delete(ChatSession).where(ChatSession.id == session_id, ChatSession.user_id == user_id))
        return True

    # ------------------------------------------------------------ messages
    async def recent_messages(self, session_id: UUID, limit: int) -> list[ChatMessage]:
        rows = (
            await self.db.execute(
                select(ChatMessage)
                .where(ChatMessage.session_id == session_id, ChatMessage.role.in_([MessageRole.USER.value, MessageRole.ASSISTANT.value]))
                .order_by(ChatMessage.created_at.desc(), ChatMessage.id.desc())
                .limit(limit)
            )
        ).scalars()
        return list(reversed(list(rows)))

    async def add_user_message(self, session_id: UUID, content: str, at: datetime | None = None) -> ChatMessage:
        msg = ChatMessage(id=uuid4(), session_id=session_id, role=MessageRole.USER.value, content=content, created_at=at or utcnow())
        self.db.add(msg)
        await self.db.flush()
        return msg

    async def add_assistant_message(self, session_id: UUID, outcome: AnswerOutcome, at: datetime | None = None) -> ChatMessage:
        msg = ChatMessage(
            id=uuid4(),
            session_id=session_id,
            role=MessageRole.ASSISTANT.value,
            content=outcome.content,
            answer_status=outcome.answer_status.value,
            warning_level=warning_to_db(outcome.warning_level),
            context_used=list(outcome.personal_context_used),
            suggested_followups=list(outcome.suggested_followups),
            model_name=outcome.model_used,
            created_at=at or utcnow(),
        )
        self.db.add(msg)
        await self.db.flush()
        for ev in outcome.citations:
            self.db.add(
                MessageCitation(
                    id=uuid4(),
                    message_id=msg.id,
                    citation_index=ev.citation_id,
                    document_id=ev.document_id,
                    chunk_id=ev.chunk_id,
                    drug_id=ev.drug_id,
                    citation_type=ev.citation_type.value,
                    source_title=(ev.source_title or "")[:255] or None,
                    excerpt=ev.excerpt,
                )
            )
        await self.db.flush()
        return msg

    async def touch_session(self, chat: ChatSession, *, title: str | None = None, mode: ContextMode | None = None, at: datetime | None = None) -> None:
        values: dict = {"updated_at": at or utcnow()}
        if title is not None:
            values["title"] = title[:255]
        if mode is not None:
            values["context_mode"] = mode.value
        await self.db.execute(update(ChatSession).where(ChatSession.id == chat.id, ChatSession.user_id == chat.user_id).values(**values))
        for k, v in values.items():
            setattr(chat, k, v)

    async def count_messages(self, session_id: UUID) -> int:
        return (await self.db.execute(select(func.count()).select_from(ChatMessage).where(ChatMessage.session_id == session_id))).scalar_one()

    async def list_message_views(self, user_id: UUID, session_id: UUID, page: int, page_size: int) -> tuple[list[MessageView], int]:
        """Lịch sử cũ→mới kèm citations và feedback CỦA CHÍNH user (đã kiểm tra sở hữu qua session)."""
        total = await self.count_messages(session_id)
        msgs = list(
            (
                await self.db.execute(
                    select(ChatMessage)
                    .where(ChatMessage.session_id == session_id)
                    .order_by(ChatMessage.created_at.asc(), ChatMessage.id.asc())
                    .offset((page - 1) * page_size)
                    .limit(page_size)
                )
            ).scalars()
        )
        views = await self.attach_details(user_id, msgs)
        return views, total

    async def attach_details(self, user_id: UUID, msgs: Sequence[ChatMessage]) -> list[MessageView]:
        ids = [m.id for m in msgs if m.role == MessageRole.ASSISTANT.value]
        cites: dict[UUID, list[CitationRow]] = {i: [] for i in ids}
        fbs: dict[UUID, AiFeedback] = {}
        if ids:
            rows = await self.db.execute(
                text(
                    "SELECT mc.message_id, mc.citation_index, mc.citation_type, mc.document_id, mc.chunk_id, mc.drug_id, "
                    "mc.source_title, mc.excerpt, kd.source_url "
                    "FROM message_citations mc LEFT JOIN knowledge_documents kd ON kd.id = mc.document_id "
                    "WHERE mc.message_id = ANY(:ids) ORDER BY mc.message_id, mc.citation_index"
                ),
                {"ids": ids},
            )
            for r in rows:
                cites[r.message_id].append(
                    CitationRow(r.citation_index, r.citation_type, r.document_id, r.chunk_id, r.drug_id, r.source_title, r.excerpt, r.source_url)
                )
            for fb in (await self.db.execute(select(AiFeedback).where(AiFeedback.message_id.in_(ids), AiFeedback.user_id == user_id))).scalars():
                fbs[fb.message_id] = fb
        return [MessageView(m, cites.get(m.id, []), fbs.get(m.id)) for m in msgs]

    # ------------------------------------------------------------ feedback
    async def get_owned_message(self, user_id: UUID, message_id: UUID) -> ChatMessage | None:
        return (
            await self.db.execute(
                select(ChatMessage)
                .join(ChatSession, ChatSession.id == ChatMessage.session_id)
                .where(ChatMessage.id == message_id, ChatSession.user_id == user_id)
            )
        ).scalar_one_or_none()

    async def upsert_feedback(self, user_id: UUID, message_id: UUID, rating: str, comment: str | None) -> AiFeedback:
        now = utcnow()
        stmt = (
            pg_insert(AiFeedback)
            .values(id=uuid4(), message_id=message_id, user_id=user_id, rating=rating, comment=comment, created_at=now, updated_at=now)
            .on_conflict_do_update(
                constraint="uq_ai_feedback_message_user", set_={"rating": rating, "comment": comment, "updated_at": now}
            )
            .returning(AiFeedback)
        )
        row = (await self.db.execute(stmt.execution_options(populate_existing=True))).scalar_one()
        return row
