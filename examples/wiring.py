"""Ví dụ nối M3 vào app FastAPI chính. Các adapter M1/M2 bên dưới là KHUNG: Trọng cài đặt bằng service thật."""
from __future__ import annotations

from typing import Sequence
from uuid import UUID

from fastapi import FastAPI

from app.modules.ai_assistant import build_container, create_ai_assistant_router, register_exception_handlers
from app.modules.ai_assistant.integration import (
    DrugResolution, DrugSelection, DrugSummary, InteractionReport, Principal, RawHealthContext,
)
from app.modules.ai_assistant.enums import PersonalContextSection


class HealthContextAdapter:  # bọc GetPersonalHealthContext của M1
    async def get_personal_health_context(self, principal: Principal, requested_sections: frozenset[PersonalContextSection]) -> RawHealthContext:
        raise NotImplementedError("map dữ liệu M1 → RawHealthContext; chỉ truy vấn các nhóm trong requested_sections")


class InteractionAdapter:  # bọc CheckDrugInteractions của M2 (cùng Rule Engine với POST /interactions/check)
    async def check_drug_interactions(self, principal: Principal, selections: Sequence[DrugSelection]) -> InteractionReport:
        raise NotImplementedError


class ResolverAdapter:  # bọc ResolveDrugEntity của M2
    async def resolve_drug_entity(self, extracted_name: str, extracted_strength: str | None = None) -> DrugResolution:
        raise NotImplementedError


class DirectoryAdapter:  # tên thuốc từ bảng drugs
    async def get_drug_summaries(self, drug_ids: Sequence[UUID]) -> dict[UUID, DrugSummary]:
        raise NotImplementedError


def create_app(get_principal, get_db_session) -> FastAPI:  # noqa: ANN001
    """get_principal / get_db_session: dependency có sẵn của app chính (Auth + AsyncSession)."""
    app = FastAPI()
    register_exception_handlers(app)  # bỏ dòng này nếu app đã có handler tạo cùng envelope lỗi
    container = build_container(
        health_context_provider=HealthContextAdapter(),
        interaction_checker=InteractionAdapter(),
        drug_resolver=ResolverAdapter(),
        drug_directory=DirectoryAdapter(),
    )  # đọc cấu hình từ biến môi trường: GEMINI_API_KEY, KAGGLE_MEDGEMMA_NGROK_URL, KAGGLE_RERANKER_NGROK_URL, OLLAMA_BASE_URL…
    app.include_router(create_ai_assistant_router(get_principal=get_principal, get_db_session=get_db_session, container=container))
    return app
