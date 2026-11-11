"""7 endpoint của Module 3 (VitaP_API_Docs.md §9). Điểm nối với Auth/DB do app chính truyền vào."""
from typing import Annotated, Awaitable, Callable
from uuid import UUID

from fastapi import APIRouter, Depends, Query, Response
from sqlalchemy.ext.asyncio import AsyncSession

from .container import AiAssistantContainer
from .errors import consent_required, invalid_message_role, login_required, not_found, rate_limited
from .enums import MessageRole
from .integration import Principal
from .repositories.chat_repo import ChatRepository
from .schemas.chat_dto import (
    ExplainInteractionOut,
    ExplainInteractionRequest,
    FeedbackOut,
    FeedbackRequest,
    MessageCreateRequest,
    MessageListOut,
    SendMessageOut,
    SessionCreateRequest,
    SessionListOut,
    SessionOut,
)
from .services.chat_service import ChatService
from .services.explain_service import ExplainService


def create_ai_assistant_router(
    *,
    get_principal: Callable[..., Principal | Awaitable[Principal]],
    get_db_session: Callable[..., AsyncSession],
    container: AiAssistantContainer,
    prefix: str = "/api/v1/ai",
) -> APIRouter:
    """
    get_principal : FastAPI dependency của Auth. Thiếu/sai token → 401; Guest hợp lệ trả principal.is_guest=True.
    get_db_session: FastAPI dependency trả AsyncSession (một session/request).
    """
    router = APIRouter(prefix=prefix, tags=["M3 · AI Health Assistant"])

    async def ai_user(principal: Principal = Depends(get_principal), db: AsyncSession = Depends(get_db_session)) -> Principal:
        if principal.is_guest:
            raise login_required()  # Guest không được dùng M3
        if not await ChatRepository(db).has_active_ai_consent(principal.user_id):
            raise consent_required()
        return principal

    def _limit(limiter_name: str):  # noqa: ANN202
        async def dep(principal: Principal = Depends(ai_user)) -> Principal:
            limiter = getattr(container, limiter_name)
            ok, retry = limiter.hit(str(principal.user_id))
            if not ok:
                raise rate_limited(retry)
            return principal

        return dep

    User = Annotated[Principal, Depends(ai_user)]
    Db = Annotated[AsyncSession, Depends(get_db_session)]
    chat_limited = Depends(_limit("chat_limiter"))
    explain_limited = Depends(_limit("explain_limiter"))

    def svc(db: AsyncSession) -> ChatService:
        return ChatService(db, container)

    @router.post("/chat/sessions", status_code=201, response_model=SessionOut)
    async def create_session(body: SessionCreateRequest, principal: User, db: Db):
        return await svc(db).create_session(principal, body.title, body.context_mode)

    @router.get("/chat/sessions", response_model=SessionListOut)
    async def list_sessions(
        principal: User, db: Db, page: int = Query(1, ge=1), page_size: int = Query(20, ge=1, le=100)
    ):
        return await svc(db).list_sessions(principal, page, page_size)

    @router.get("/chat/sessions/{session_id}/messages", response_model=MessageListOut)
    async def list_messages(
        session_id: UUID, principal: User, db: Db, page: int = Query(1, ge=1), page_size: int = Query(50, ge=1, le=100)
    ):
        return await svc(db).list_messages(principal, session_id, page, page_size)

    @router.post("/chat/sessions/{session_id}/messages", response_model=SendMessageOut)
    async def send_message(session_id: UUID, body: MessageCreateRequest, db: Db, principal: Principal = chat_limited):
        return await svc(db).send_message(principal, session_id, body.content, body.context_mode_override)

    @router.delete("/chat/sessions/{session_id}", status_code=204)
    async def delete_session(session_id: UUID, principal: User, db: Db):
        await svc(db).delete_session(principal, session_id)
        return Response(status_code=204)

    @router.post("/explain/interaction", response_model=ExplainInteractionOut)
    async def explain_interaction(body: ExplainInteractionRequest, db: Db, principal: Principal = explain_limited):
        return await ExplainService(db, container).explain(principal, body)

    @router.post("/chat/messages/{message_id}/feedback", response_model=FeedbackOut)
    async def feedback(message_id: UUID, body: FeedbackRequest, principal: User, db: Db):
        repo = ChatRepository(db)
        msg = await repo.get_owned_message(principal.user_id, message_id)
        if msg is None:
            raise not_found("Tin nhắn")  # không tồn tại hoặc của người khác → 404
        if msg.role != MessageRole.ASSISTANT.value:
            raise invalid_message_role()
        row = await repo.upsert_feedback(principal.user_id, message_id, body.rating.value, body.comment)
        await db.commit()
        return FeedbackOut(message_id=row.message_id, rating=row.rating, comment=row.comment, updated_at=row.updated_at)

    return router
