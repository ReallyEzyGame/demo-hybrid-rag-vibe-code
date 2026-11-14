from uuid import uuid4

from app.modules.ai_assistant.domain import EvidenceItem, InteractionContext
from app.modules.ai_assistant.enums import AnswerStatus, CitationType, WarningLevel
from app.modules.ai_assistant.integration import InteractionReport, PairResult, ReportSummary
from app.modules.ai_assistant.rag.guards.output_guard import OutputGuard, extract_marker_ids
from app.modules.ai_assistant.schemas.llm_output import LLMGeneratedAnswer

guard = OutputGuard()


def ev(n: int) -> EvidenceItem:
    return EvidenceItem(n, CitationType.MEDICAL_DOCUMENT, f"Nguồn {n}", "text", "excerpt", document_id=uuid4(), chunk_id=uuid4())


def ans(content, status=AnswerStatus.ANSWERED, level=WarningLevel.INFO, cited=None, follow=None):
    return LLMGeneratedAnswer(content=content, answer_status=status, warning_level=level, cited_ids=cited or [], suggested_followups=follow or [])


def ctx(status: str, severity=None) -> InteractionContext:
    pair = PairResult(selection_a="a", selection_b="b", status=status, highest_severity=severity)
    rep = InteractionReport(summary=ReportSummary(overall_status=status), pairs=[pair])
    return InteractionContext(report=rep, asked=True)


def test_marker_extraction_handles_groups():
    assert extract_marker_ids("A [1] và B [2, 3] cùng C [4][5]") == [1, 2, 3, 4, 5]


def test_valid_citations_pass_through():
    g = guard.validate(ans("A1C 6.2% là tiền tiểu đường [1].", cited=[1]), [ev(1), ev(2)])
    assert g.answer_status is AnswerStatus.ANSWERED
    assert [c.citation_id for c in g.citations] == [1]
    assert g.violations == []


def test_hallucinated_citation_id_is_removed_and_downgraded():
    g = guard.validate(ans("Thông tin X [1]. Thông tin Y [9].", cited=[1, 9]), [ev(1)])
    assert "[9]" not in g.content and "[1]" in g.content
    assert "INVALID_CITATION_REMOVED" in g.violations
    assert g.answer_status is AnswerStatus.PARTIAL
    assert g.warning_level is WarningLevel.WARNING


def test_only_fake_citations_becomes_insufficient_evidence():
    g = guard.validate(ans("Khẳng định y khoa [7].", cited=[7]), [ev(1)])
    assert g.answer_status is AnswerStatus.INSUFFICIENT_EVIDENCE
    assert g.citations == []


def test_answer_without_any_citation_is_rejected():
    g = guard.validate(ans("Bạn nên uống nhiều nước."), [ev(1)])
    assert g.answer_status is AnswerStatus.INSUFFICIENT_EVIDENCE
    assert "NO_VALID_CITATION" in g.violations


def test_answer_with_no_evidence_is_rejected():
    g = guard.validate(ans("Có đấy [1].", cited=[1]), [])
    assert g.answer_status is AnswerStatus.INSUFFICIENT_EVIDENCE


def test_safe_claim_blocked_even_with_valid_status():
    for bad in ["Hai thuốc này hoàn toàn an toàn khi dùng chung [1].", "Uống chung không sao [1].", "Không có tương tác giữa hai thuốc [1]."]:
        g = guard.validate(ans(bad, cited=[1]), [ev(1)], ctx("INSUFFICIENT_DATA"))
        assert g.answer_status is AnswerStatus.NEEDS_CLINICIAN, bad
        assert "SAFE_CLAIM_BLOCKED" in g.violations
        assert "an toàn khi dùng chung" not in g.content


def test_hedged_no_known_interaction_wording_is_allowed():
    text = "Hệ thống chưa ghi nhận tương tác đã biết trong cơ sở dữ liệu hiện có, nhưng điều này chưa có nghĩa là an toàn [1]."
    g = guard.validate(ans(text, cited=[1]), [ev(1)], ctx("NO_KNOWN_INTERACTION_IN_DATABASE"))
    assert g.answer_status is AnswerStatus.ANSWERED
    assert g.violations == []


def test_directive_to_stop_medication_is_blocked():
    g = guard.validate(ans("Bạn nên ngưng thuốc huyết áp vì chỉ số đã ổn [1].", cited=[1]), [ev(1)])
    assert g.answer_status is AnswerStatus.NEEDS_CLINICIAN
    assert "CLINICAL_DIRECTIVE_BLOCKED" in g.violations


def test_negated_directive_is_allowed():
    g = guard.validate(ans("Bạn không nên tự ý ngưng thuốc; hãy hỏi bác sĩ điều trị [1].", cited=[1]), [ev(1)])
    assert g.answer_status is AnswerStatus.ANSWERED


def test_firm_diagnosis_blocked():
    g = guard.validate(ans("Chẩn đoán của bạn là viêm dạ dày [1].", cited=[1]), [ev(1)])
    assert g.answer_status is AnswerStatus.NEEDS_CLINICIAN


def test_diagnosis_hint_never_stays_answered():
    g = guard.validate(ans("Đau bụng âm ỉ thường do các nguyên nhân phổ biến sau [1].", cited=[1]), [ev(1)], diagnosis_hint=True)
    assert g.answer_status is AnswerStatus.NEEDS_CLINICIAN
    assert g.warning_level is WarningLevel.WARNING


def test_major_interaction_forces_critical():
    g = guard.validate(ans("Có tương tác đáng kể giữa hai thuốc [1].", cited=[1]), [ev(1)], ctx("INTERACTION_FOUND", "MAJOR"))
    assert g.warning_level is WarningLevel.CRITICAL


def test_interaction_gap_downgrades_answered():
    g = guard.validate(ans("Chưa đủ dữ liệu cho một số thuốc [1].", cited=[1]), [ev(1)], ctx("UNRESOLVED_DRUG"))
    assert g.answer_status is AnswerStatus.PARTIAL


def test_llm_emergency_gets_critical_and_115():
    g = guard.validate(ans("Tình trạng này rất nguy hiểm.", AnswerStatus.EMERGENCY_REDIRECT), [])
    assert g.warning_level is WarningLevel.CRITICAL and "115" in g.content


def test_followups_capped_at_three():
    g = guard.validate(ans("Ok [1].", cited=[1], follow=["a", "b", "c", "d", " "]), [ev(1)])
    assert g.followups == ["a", "b", "c"]


def test_cited_red_evidence_escalates_warning_even_if_llm_says_info():
    red = ev(1)
    red.severity_hint = "RED"
    g = guard.validate(ans("Tránh rượu bia khi dùng thuốc này [1].", level=WarningLevel.INFO, cited=[1]), [red])
    assert g.warning_level is WarningLevel.WARNING and "RED_EVIDENCE_ESCALATED" in g.violations


def test_uncited_red_evidence_and_green_do_not_escalate():
    red, green = ev(1), ev(2)
    red.severity_hint, green.severity_hint = "RED", "GREEN"
    assert guard.validate(ans("Chỉ dùng nguồn xanh [2].", cited=[2]), [red, green]).warning_level is WarningLevel.INFO
