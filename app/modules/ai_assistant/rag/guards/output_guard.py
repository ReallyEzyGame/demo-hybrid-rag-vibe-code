"""LỚP 3 — Output Guardrail (thiết kế §2.2): kiểm định citation, câu khẳng định an toàn, chỉ định điều trị."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from ...domain import EvidenceItem, InteractionContext
from ...enums import AnswerStatus, WarningLevel
from ...schemas.llm_output import LLMGeneratedAnswer
from .messages import CLINICIAN_FALLBACK_MESSAGE, EMERGENCY_MESSAGE, INSUFFICIENT_EVIDENCE_MESSAGE
from .text_utils import fold

_MARKER_GROUP = re.compile(r"\[(\s*\d{1,3}\s*(?:[,;]\s*\d{1,3}\s*)*)\]")

# Câu khẳng định "an toàn" bị cấm (áp trên văn bản đã bỏ dấu)
_SAFE_CLAIMS = re.compile("|".join(f"(?:{p})" for p in [
    r"hoan toan an toan", r"an toan tuyet doi", r"tuyet doi an toan",
    r"an toan khi (uong|dung|su dung|ket hop) (chung|cung|ket hop)",
    r"(uong|dung|su dung) (chung|cung nhau|cung luc) (duoc|thoai mai|khong sao|an toan)",
    r"khong sao ca", r"khong (co|ghi nhan duoc) bat ky tuong tac",
    r"khong (co|hay co|he co|gay) tuong tac", r"hoan toan khong tuong tac", r"khong tuong tac gi",
    r"co the ket hop (an toan|thoai mai)", r"ket hop (an toan|thoai mai)",
]))

# Chỉ định điều trị / chẩn đoán chắc chắn bị cấm
_DIRECTIVES = re.compile("|".join(f"(?:{p})" for p in [
    r"(nen|hay|can|phai|duoc phep|co the) (tu y |tu minh )?(ngung|dung|bo|cat) (han )?(uong )?thuoc",
    r"(nen|hay|can|phai|co the) (tang|giam|doi|gap doi) (them )?lieu",
    r"chan doan (cua ban )?(la|chinh la)",
    r"chac chan (ban )?(bi|mac)",
    r"ket luan (la )?ban (bi|mac)",
    r"ban (bi|mac) [\w ]{3,40} roi",
]))
_NEGATION = re.compile(r"(khong|dung|tranh|cam|chua nen|tuyet doi khong|chi nen khi|chi khi)\s*\w*\s*\w*\s*$")


def _has_unnegated_directive(folded: str) -> bool:
    for m in _DIRECTIVES.finditer(folded):
        window = folded[max(0, m.start() - 28): m.start()]
        # "không nên tự ý ngưng thuốc" -> được phép (có phủ định đứng trước)
        if _NEGATION.search(window):
            continue
        return True
    return False


def extract_marker_ids(text: str) -> list[int]:
    ids: list[int] = []
    for grp in _MARKER_GROUP.finditer(text):
        ids.extend(int(x) for x in re.findall(r"\d+", grp.group(1)))
    return ids


def _strip_invalid_markers(text: str, valid: set[int]) -> str:
    def repl(m: re.Match[str]) -> str:
        nums = [int(x) for x in re.findall(r"\d+", m.group(1))]
        keep = [n for n in nums if n in valid]
        return f"[{', '.join(str(n) for n in keep)}]" if keep else ""

    cleaned = _MARKER_GROUP.sub(repl, text)
    return re.sub(r"[ \t]+([.,;:])", r"\1", re.sub(r" {2,}", " ", cleaned)).strip()


@dataclass
class GuardedAnswer:
    content: str
    answer_status: AnswerStatus
    warning_level: WarningLevel
    citations: list[EvidenceItem] = field(default_factory=list)
    followups: list[str] = field(default_factory=list)
    violations: list[str] = field(default_factory=list)


def interaction_fallback_text(ctx: InteractionContext | None) -> str:
    """Câu trả lời xác định (không LLM) khi LLM vi phạm quy tắc tương tác thuốc."""
    if ctx is None or ctx.unavailable or not ctx.report:
        return (
            "Hiện hệ thống chưa kiểm tra được tương tác giữa các thuốc này nên mình không thể kết luận việc dùng "
            "chung có an toàn hay không. Bạn nên hỏi bác sĩ hoặc dược sĩ trước khi dùng cùng lúc."
        )
    parts = []
    for p in ctx.report.pairs:
        if p.status == "INTERACTION_FOUND":
            parts.append(f"Hệ thống ghi nhận tương tác mức {p.highest_severity or 'chưa rõ'} giữa một cặp thuốc bạn hỏi.")
        elif p.status == "NO_KNOWN_INTERACTION_IN_DATABASE":
            parts.append("Với một cặp thuốc, cơ sở dữ liệu hiện có chưa ghi nhận tương tác đã biết (điều này chưa có nghĩa là an toàn).")
        else:
            parts.append("Với một cặp thuốc, dữ liệu hiện có chưa đủ để kết luận.")
    parts.append("Hãy hỏi bác sĩ hoặc dược sĩ trước khi dùng chung các thuốc này.")
    return " ".join(parts)


class OutputGuard:
    """Kiểm định phản hồi LLM. Không bao giờ ném lỗi: vi phạm → hạ trạng thái hoặc thay bằng câu trả lời cố định."""

    def validate(
        self,
        answer: LLMGeneratedAnswer,
        evidence: list[EvidenceItem],
        interaction: InteractionContext | None = None,
        *,
        diagnosis_hint: bool = False,
    ) -> GuardedAnswer:
        violations: list[str] = []
        by_id = {e.citation_id: e for e in evidence}
        valid = set(by_id)
        status = answer.answer_status
        level = answer.warning_level
        content = answer.content.strip()

        # --- Chốt chặn nội dung cấm (ưu tiên trước citation) ---
        folded = fold(content)
        if _SAFE_CLAIMS.search(folded):
            violations.append("SAFE_CLAIM_BLOCKED")
            asked_interaction = interaction is not None and interaction.asked
            text = interaction_fallback_text(interaction if asked_interaction else None)
            return GuardedAnswer(text, AnswerStatus.NEEDS_CLINICIAN, WarningLevel.WARNING, [], [], violations)
        if _has_unnegated_directive(folded):
            violations.append("CLINICAL_DIRECTIVE_BLOCKED")
            return GuardedAnswer(CLINICIAN_FALLBACK_MESSAGE, AnswerStatus.NEEDS_CLINICIAN, WarningLevel.WARNING, [], [], violations)

        # --- Trạng thái đặc biệt do LLM tự phát hiện ---
        if status is AnswerStatus.EMERGENCY_REDIRECT:
            if "115" not in content:
                content = f"{content}\n\n{EMERGENCY_MESSAGE}".strip()
            return GuardedAnswer(content, status, WarningLevel.CRITICAL, [], [], violations)
        if status is AnswerStatus.INSUFFICIENT_EVIDENCE:
            return GuardedAnswer(content or INSUFFICIENT_EVIDENCE_MESSAGE, status, WarningLevel.INFO, [], [], violations)

        # --- Citation verification ---
        marker_ids = extract_marker_ids(content)
        invalid = {i for i in marker_ids if i not in valid}
        declared = [i for i in answer.cited_ids if i in valid]
        if invalid or len([i for i in answer.cited_ids if i not in valid]):
            violations.append("INVALID_CITATION_REMOVED")
        if invalid:
            content = _strip_invalid_markers(content, valid)
        used_ids = sorted({i for i in extract_marker_ids(content) if i in valid} | set(declared))

        needs_citation = status in (AnswerStatus.ANSWERED, AnswerStatus.PARTIAL, AnswerStatus.CONFLICTING_EVIDENCE) and bool(evidence)
        if needs_citation and not used_ids:
            violations.append("NO_VALID_CITATION")
            return GuardedAnswer(INSUFFICIENT_EVIDENCE_MESSAGE, AnswerStatus.INSUFFICIENT_EVIDENCE, WarningLevel.INFO, [], [], violations)
        if not evidence and status in (AnswerStatus.ANSWERED, AnswerStatus.PARTIAL, AnswerStatus.CONFLICTING_EVIDENCE):
            violations.append("ANSWER_WITHOUT_EVIDENCE")
            return GuardedAnswer(INSUFFICIENT_EVIDENCE_MESSAGE, AnswerStatus.INSUFFICIENT_EVIDENCE, WarningLevel.INFO, [], [], violations)

        # Một phần trích dẫn bị loại → chỉ còn trả lời được một phần
        if "INVALID_CITATION_REMOVED" in violations and status is AnswerStatus.ANSWERED:
            status = AnswerStatus.PARTIAL
            level = max(level, WarningLevel.WARNING, key=_rank)

        # Chẩn đoán: không bao giờ để ANSWERED
        if diagnosis_hint and status is AnswerStatus.ANSWERED:
            status = AnswerStatus.NEEDS_CLINICIAN
            violations.append("DIAGNOSIS_FORCED_NEEDS_CLINICIAN")
        if status is AnswerStatus.NEEDS_CLINICIAN or status is AnswerStatus.CONFLICTING_EVIDENCE or status is AnswerStatus.PARTIAL:
            level = max(level, WarningLevel.WARNING, key=_rank)

        # Có khoảng trống tương tác: không để mức INFO "êm" và không để ANSWERED nếu không nhắc giới hạn
        if interaction is not None and interaction.asked and interaction.has_unresolved_gap and status is AnswerStatus.ANSWERED:
            status = AnswerStatus.PARTIAL
            level = max(level, WarningLevel.WARNING, key=_rank)
            violations.append("INTERACTION_GAP_DOWNGRADED")
        if interaction is not None and "INTERACTION_FOUND" in interaction.statuses:
            sev = {p.highest_severity for p in (interaction.report.pairs if interaction.report else [])}
            if sev & {"MAJOR", "CONTRAINDICATED"}:
                level = WarningLevel.CRITICAL if status is not AnswerStatus.OUT_OF_SCOPE else level
            else:
                level = max(level, WarningLevel.WARNING, key=_rank)

        followups = [f.strip() for f in answer.suggested_followups if f and f.strip()][:3]
        return GuardedAnswer(content, status, level, [by_id[i] for i in used_ids], followups, violations)


def _rank(level: WarningLevel) -> int:
    return {WarningLevel.INFO: 0, WarningLevel.WARNING: 1, WarningLevel.CRITICAL: 2}[level]


@dataclass
class GuardedExplanation:
    explanation: str
    clinical_management: str | None
    citations: list[EvidenceItem]
    violations: list[str] = field(default_factory=list)
    use_template: bool = False


def validate_explanation(
    explanation: str,
    clinical_management: str | None,
    cited_ids: list[int],
    evidence: list[EvidenceItem],
    *,
    require_citation: bool,
) -> GuardedExplanation:
    """Kiểm định phản hồi của endpoint /ai/explain/interaction. use_template=True → caller dùng câu giải thích cố định."""
    violations: list[str] = []
    valid = {e.citation_id for e in evidence}
    by_id = {e.citation_id: e for e in evidence}
    for field_name, text in (("explanation", explanation), ("clinical_management", clinical_management or "")):
        folded = fold(text)
        if _SAFE_CLAIMS.search(folded):
            violations.append(f"SAFE_CLAIM_BLOCKED:{field_name}")
        if _has_unnegated_directive(folded):
            violations.append(f"CLINICAL_DIRECTIVE_BLOCKED:{field_name}")
    if violations:
        return GuardedExplanation("", None, [], violations, use_template=True)
    if any(i not in valid for i in extract_marker_ids(explanation)):
        violations.append("INVALID_CITATION_REMOVED")
        explanation = _strip_invalid_markers(explanation, valid)
    if clinical_management and any(i not in valid for i in extract_marker_ids(clinical_management)):
        clinical_management = _strip_invalid_markers(clinical_management, valid)
    used = sorted({i for i in extract_marker_ids(explanation) if i in valid} | {i for i in cited_ids if i in valid})
    if require_citation and not used:
        violations.append("NO_VALID_CITATION")
        return GuardedExplanation("", None, [], violations, use_template=True)
    return GuardedExplanation(explanation, clinical_management or None, [by_id[i] for i in used], violations)
