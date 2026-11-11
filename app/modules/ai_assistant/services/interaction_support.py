"""Cầu nối M3 ↔ Rule Engine M2: chọn thuốc, gọi CheckDrugInteractions, đóng gói kết quả cho prompt. LLM không tự đoán tương tác."""
from __future__ import annotations

import logging
from uuid import UUID

from ..config import AiAssistantSettings
from ..domain import EvidenceItem, InteractionContext, PersonalContext, make_excerpt
from ..enums import CitationType, ContextMode
from ..integration import (
    DrugDirectory,
    DrugSelection,
    DrugSummary,
    InteractionChecker,
    InteractionReport,
    Principal,
)
from .drug_mentions import DrugMentionExtractor
from .personal_context import has_interaction_intent

logger = logging.getLogger(__name__)


def _label(drug_id: UUID, summaries: dict[UUID, DrugSummary]) -> str:
    s = summaries.get(drug_id)
    return s.name if s else f"thuốc {str(drug_id)[:8]}"


def _to_report(raw: InteractionReport | dict) -> InteractionReport:
    return raw if isinstance(raw, InteractionReport) else InteractionReport.model_validate(raw)


def build_context(report: InteractionReport, labels: dict[str, str]) -> InteractionContext:
    """Đánh số bằng chứng INTERACTION_RULE từ 1..k và dựng khối <drug_interaction_result>."""
    evidence: list[EvidenceItem] = []
    lines = [
        f"Tổng quan: overall_status={report.summary.overall_status}; số cặp={report.summary.pair_count}; "
        f"đã đánh giá đầy đủ={report.summary.evaluated_pair_count}; is_complete={str(report.summary.is_complete).lower()}"
    ]
    for p in report.pairs:
        a, b = labels.get(p.selection_a, p.selection_a), labels.get(p.selection_b, p.selection_b)
        head = f"- {a} + {b}: status={p.status}; mức cao nhất={p.highest_severity or 'không có'}"
        if p.missing_data:
            head += f"; thiếu dữ liệu={', '.join(p.missing_data)}"
        lines.append(head)
        for f in p.findings:
            n = len(evidence) + 1
            src = f.source.source_name if f.source and f.source.source_name else "Rule Engine VitaP (M2)"
            body = f"Tương tác {f.severity} giữa {a} và {b}: {f.description}"
            if f.mechanism:
                body += f" Cơ chế: {f.mechanism}"
            if f.recommendation:
                body += f" Khuyến nghị: {f.recommendation}"
            evidence.append(EvidenceItem(n, CitationType.INTERACTION_RULE, src, body, make_excerpt(f.description)))
            lines.append(f"  [{n}] {body}")
    if report.limitations:
        lines.append("Giới hạn của kiểm tra: " + " ".join(report.limitations))
    return InteractionContext(report=report, prompt_text="\n".join(lines), evidence=evidence, labels=dict(labels))


class InteractionSupport:
    def __init__(
        self,
        checker: InteractionChecker,
        directory: DrugDirectory,
        extractor: DrugMentionExtractor,
        settings: AiAssistantSettings,
    ) -> None:
        self.checker, self.directory, self.extractor, self.s = checker, directory, extractor, settings

    async def build_for_chat(
        self, principal: Principal, question: str, personal: PersonalContext, mode: ContextMode
    ) -> InteractionContext | None:
        mentions = await self.extractor.extract(question)
        intent = has_interaction_intent(question)
        if not mentions and not intent:
            return None
        ids: list[UUID] = [m.drug_id for m in mentions]
        if mode is ContextMode.PERSONAL:  # thuốc đang dùng (ACTIVE) chỉ khi người dùng không ẩn danh
            ids += [i for i in personal.active_drug_ids if i not in ids]
        ids = ids[: self.s.max_selected_drugs]
        if len(ids) < 2:
            return None  # không đủ cặp để kiểm tra; không đoán
        return await self._check(principal, ids)

    async def build_for_pair(self, principal: Principal, a: UUID, b: UUID) -> InteractionContext:
        return await self._check(principal, [a, b], fail_open=False)

    async def _check(self, principal: Principal, ids: list[UUID], fail_open: bool = True) -> InteractionContext:
        selections = [DrugSelection(str(i), i) for i in ids]
        try:
            report = _to_report(await self.checker.check_drug_interactions(principal, selections))
            summaries = await self.directory.get_drug_summaries(ids)
        except Exception:  # noqa: BLE001
            logger.exception("Rule Engine M2 unavailable")
            if not fail_open:
                raise
            # Không bao giờ để LLM tự suy đoán khi Rule Engine lỗi: báo rõ cho prompt và Output Guard.
            return InteractionContext(
                unavailable=True,
                prompt_text="Rule Engine tương tác thuốc hiện KHÔNG khả dụng. Không được nêu bất kỳ kết luận nào về tương tác.",
            )
        labels = {str(i): _label(i, summaries) for i in ids}
        return build_context(report, labels)
