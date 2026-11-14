"""POST /ai/explain/interaction — giải thích nhanh kết quả Rule Engine M2 cho MỘT cặp thuốc, không tạo session, không lưu DB."""
from __future__ import annotations

import logging

from sqlalchemy.ext.asyncio import AsyncSession

from ..container import AiAssistantContainer
from ..domain import EvidenceItem, InteractionContext
from ..enums import ContextMode, PersonalContextSection as S, WarningLevel
from ..errors import AppError, dependency_unavailable
from ..integration import Principal
from ..rag.generation.lcel_chain import GenerationUnavailable
from ..rag.guards.output_guard import validate_explanation
from ..schemas.chat_dto import CitationOut, ExplainInteractionOut, ExplainInteractionRequest
from .chat_service import chunk_evidence, render_evidence

logger = logging.getLogger(__name__)
TEMPLATE_MODEL = "rule-template"
_SEV_VI = {"MINOR": "nhẹ", "MODERATE": "trung bình", "MAJOR": "nghiêm trọng", "CONTRAINDICATED": "chống chỉ định"}


def warning_for(status: str, severity: str | None) -> WarningLevel:
    if status == "INTERACTION_FOUND":
        return WarningLevel.CRITICAL if severity in ("MAJOR", "CONTRAINDICATED") else (
            WarningLevel.WARNING if severity == "MODERATE" else WarningLevel.INFO
        )
    if status == "NO_KNOWN_INTERACTION_IN_DATABASE":
        return WarningLevel.INFO
    return WarningLevel.WARNING  # chưa đủ dữ liệu / thuốc chưa xác định: cần thận trọng


def template_explanation(ctx: InteractionContext, a_label: str, b_label: str) -> str:
    pair = ctx.report.pairs[0] if ctx.report and ctx.report.pairs else None
    tail = " Bạn nên hỏi ý kiến bác sĩ hoặc dược sĩ trước khi dùng cùng lúc."
    if pair is None or pair.status in ("INSUFFICIENT_DATA", "UNRESOLVED_DRUG"):
        return f"Hiện chưa đủ dữ liệu để kết luận về tương tác giữa {a_label} và {b_label}." + tail
    if pair.status == "NO_KNOWN_INTERACTION_IN_DATABASE":
        return (
            f"Cơ sở dữ liệu hiện có chưa ghi nhận tương tác đã biết giữa {a_label} và {b_label}. "
            "Điều này chưa có nghĩa là hai thuốc an toàn khi dùng cùng nhau." + tail
        )
    sev = _SEV_VI.get(pair.highest_severity or "", "chưa xác định")
    first = pair.findings[0].description.strip() if pair.findings else ""
    cite = " [1]" if ctx.evidence else ""
    return f"Hệ thống ghi nhận tương tác mức {sev} giữa {a_label} và {b_label}.{cite} {first}".strip() + tail


class ExplainService:
    def __init__(self, db: AsyncSession, container: AiAssistantContainer) -> None:
        self.db, self.c = db, container

    async def explain(self, principal: Principal, req: ExplainInteractionRequest) -> ExplainInteractionOut:
        if req.drug_a_id == req.drug_b_id:  # M2 cũng từ chối: không tự kiểm tra một thuốc
            raise AppError(422, "VALIDATION_ERROR", "Hai thuốc phải khác nhau", [{"field": "drug_b_id", "reason": "DUPLICATE_DRUG"}])
        c = self.c
        try:
            ctx = await c.interactions.build_for_pair(principal, req.drug_a_id, req.drug_b_id)
        except Exception as exc:  # noqa: BLE001 - Rule Engine lỗi: không trả kết quả âm tính giả
            raise dependency_unavailable("Không thể kiểm tra tương tác thuốc lúc này") from exc
        assert ctx.report is not None
        pair = ctx.report.pairs[0]
        a_label = ctx.labels.get(pair.selection_a, pair.selection_a)
        b_label = ctx.labels.get(pair.selection_b, pair.selection_b)
        severity, status = pair.highest_severity, pair.status
        management = next((f.recommendation for f in pair.findings if f.recommendation), None)

        personal = None
        if req.include_personal_context:
            personal = await c.personal_context.build(
                principal, ContextMode.PERSONAL, "", force_sections=frozenset({S.ALLERGIES, S.ACTIVE_MEDICATIONS, S.HEALTH_CONDITIONS})
            )

        docs: list[EvidenceItem] = []
        try:  # RAG bổ trợ: lỗi không làm hỏng endpoint
            rq = f"{a_label} {b_label} drug interaction"
            res = await c.retriever.retrieve(self.db, rq)
            docs = chunk_evidence(res.chunks[:2], start=len(ctx.evidence) + 1, query=rq)
        except Exception:  # noqa: BLE001
            logger.warning("Explain: RAG retrieval skipped")
        evidence = ctx.evidence + docs

        model_used = TEMPLATE_MODEL
        text, citations = "", list(ctx.evidence)
        try:
            result = await c.engine.generate(
                "explain",
                {
                    "personal_context": (personal.prompt_text if personal and personal.prompt_text else "(Không có)"),
                    "interaction_result": ctx.prompt_text,
                    "evidence": render_evidence(evidence),
                },
            )
            g = validate_explanation(
                result.parsed.explanation, result.parsed.clinical_management, result.parsed.cited_ids, evidence,
                require_citation=bool(ctx.evidence),
            )
            if not g.use_template:
                text, citations, model_used = g.explanation, g.citations, result.model_used
                management = g.clinical_management or management
            else:
                logger.info("Explain output rejected: %s", g.violations)
        except GenerationUnavailable:
            logger.warning("Explain: LLM unavailable → template")
        if model_used == TEMPLATE_MODEL:
            text = template_explanation(ctx, a_label, b_label)
            citations = list(ctx.evidence)

        return ExplainInteractionOut(
            interaction_status=status,
            severity=severity,
            warning_level=warning_for(status, severity),
            explanation=text,
            clinical_management=management,
            citations=[
                CitationOut(
                    citation_id=e.citation_id, citation_type=e.citation_type, document_id=e.document_id, chunk_id=e.chunk_id,
                    drug_id=e.drug_id, source_title=e.source_title, source_url=e.source_url, excerpt=e.excerpt,
                )
                for e in citations
            ],
            model_used=model_used,
        )
