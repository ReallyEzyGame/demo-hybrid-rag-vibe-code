"""Đồ giả (fakes) dùng chung cho test: embedding băm từ, LLM theo kịch bản, provider M1/M2."""
from __future__ import annotations

import json
import re
import zlib
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Any, Callable, Sequence
from uuid import UUID, uuid4

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import PrivateAttr

from app.modules.ai_assistant.enums import PersonalContextSection
from app.modules.ai_assistant.integration import (
    AllergyData, ConditionData, DrugResolution, DrugSelection, DrugSummary, Finding, InteractionReport, MedicationData,
    PairResult, ProfileData, RawHealthContext, ReportSummary, SourceRef, VitalData,
)
from app.modules.ai_assistant.rag.reranking.bge_client import RerankerUnavailable

_WORD = re.compile(r"[a-z0-9]+")


def _vec(text: str, dim: int = 768) -> list[float]:
    v = [0.0] * dim
    for w in _WORD.findall(text.lower()):
        h = zlib.crc32(w.encode())
        v[h % dim] += 1.0 if (h >> 16) & 1 else -1.0
        v[(h >> 8) % dim] += 0.5
    norm = sum(x * x for x in v) ** 0.5 or 1.0
    return [x / norm for x in v]


class HashingEmbeddings:
    """Bag-of-words băm → 768 chiều. Câu chung nhiều từ thì cosine cao; đủ để kiểm thử truy xuất xác định."""

    def __init__(self, fail: bool = False, dim: int = 768) -> None:
        self.fail, self.dim = fail, dim

    def embed_query(self, q: str) -> list[float]:
        return _vec(q, self.dim)

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [_vec(t, self.dim) for t in texts]

    async def aembed_query(self, q: str) -> list[float]:
        if self.fail:
            raise ConnectionError("ollama down")
        return _vec(q, self.dim)

    async def aembed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [_vec(t, self.dim) for t in texts]


class FakeReranker:
    def __init__(self, down: bool = False) -> None:
        self.down, self.calls = down, 0

    async def score(self, query: str, documents: Sequence[str]) -> list[float]:
        self.calls += 1
        if self.down:
            raise RerankerUnavailable("down")
        q = set(_WORD.findall(query.lower()))
        return [min(1.0, len(q & set(_WORD.findall(d.lower()))) / max(1, len(q))) for d in documents]


class ScriptedChatModel(BaseChatModel):
    """LLM giả: `responder(prompt_text) -> str` (có thể ném lỗi). Ghi lại mọi prompt đã nhận."""

    responder: Any
    _calls: list[str] = PrivateAttr(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "scripted"

    @property
    def calls(self) -> list[str]:
        return self._calls

    def _gen(self, messages) -> ChatResult:  # noqa: ANN001
        text = "\n".join(str(m.content) for m in messages)
        self._calls.append(text)
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=self.responder(text)))])

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):  # noqa: ANN001, ANN201
        return self._gen(messages)

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):  # noqa: ANN001, ANN201
        return self._gen(messages)


_EVIDENCE_ID = re.compile(r"^\[(\d+)\] Nguồn:", re.M)


def evidence_ids(prompt: str) -> list[int]:
    block = prompt.split("<retrieved_evidence>", 1)[-1].split("</retrieved_evidence>", 1)[0]
    return [int(x) for x in _EVIDENCE_ID.findall(block)]


def default_chat_responder(prompt: str) -> str:
    ids = evidence_ids(prompt)
    first = ids[0] if ids else 1
    return json.dumps(
        {"content": f"Theo tài liệu tham khảo, đây là thông tin bạn cần [{first}].", "answer_status": "ANSWERED",
         "warning_level": "INFO", "cited_ids": [first], "suggested_followups": ["Cần lưu ý gì khi dùng thuốc?"]},
        ensure_ascii=False,
    )


def default_explain_responder(prompt: str) -> str:
    ids = evidence_ids(prompt)
    cite = f" [{ids[0]}]" if ids else ""
    return json.dumps(
        {"explanation": f"Hai thuốc này có thể làm tăng nguy cơ xuất huyết khi dùng cùng nhau{cite}. Hãy hỏi bác sĩ hoặc dược sĩ.",
         "clinical_management": "Theo dõi dấu hiệu chảy máu và hỏi ý kiến bác sĩ.", "cited_ids": ids[:1]},
        ensure_ascii=False,
    )


def kind_responder(prompt: str) -> str:
    return default_explain_responder(prompt) if "KẾT QUẢ KIỂM TRA TƯƠNG TÁC THUỐC" in prompt else default_chat_responder(prompt)


def boom(prompt: str) -> str:  # noqa: ARG001
    raise RuntimeError("429 Too Many Requests")


# ------------------------------------------------------------------ principal & M1/M2 providers
@dataclass(frozen=True)
class FakePrincipal:
    user_id: UUID
    is_guest: bool = False


class FakeHealthProvider:
    def __init__(self, raw: RawHealthContext | None = None, fail: bool = False) -> None:
        self.raw, self.fail, self.requests = raw or RawHealthContext(), fail, []

    async def get_personal_health_context(self, principal, requested_sections):  # noqa: ANN001, ANN201
        self.requests.append(frozenset(requested_sections))
        if self.fail:
            raise RuntimeError("M1 down")
        return self.raw


PARACETAMOL, IBUPROFEN, WARFARIN, ASPIRIN = (uuid4() for _ in range(4))
NAMES = {PARACETAMOL: "Paracetamol", IBUPROFEN: "Ibuprofen", WARFARIN: "Warfarin", ASPIRIN: "Aspirin"}


class FakeResolver:
    def __init__(self) -> None:
        self.table = {"paracetamol": PARACETAMOL, "ibuprofen": IBUPROFEN, "warfarin": WARFARIN, "aspirin": ASPIRIN}
        self.calls: list[str] = []

    async def resolve_drug_entity(self, extracted_name, extracted_strength=None):  # noqa: ANN001, ANN201
        self.calls.append(extracted_name)
        did = self.table.get(extracted_name.lower())
        return DrugResolution("MATCHED", did, 0.99) if did else DrugResolution("UNRESOLVED")


class FakeDirectory:
    async def get_drug_summaries(self, drug_ids):  # noqa: ANN001, ANN201
        return {i: DrugSummary(i, NAMES.get(i, "X"), ()) for i in drug_ids}


class FakeInteractionChecker:
    """Trả kết quả theo cặp (tập hợp 2 drug_id) → (status, severity); mặc định NO_KNOWN."""

    def __init__(self, rules: dict[frozenset, tuple[str, str | None]] | None = None, fail: bool = False) -> None:
        self.rules, self.fail, self.calls = rules or {}, fail, []

    async def check_drug_interactions(self, principal, selections: Sequence[DrugSelection]):  # noqa: ANN001, ANN201
        self.calls.append([s.drug_id for s in selections])
        if self.fail:
            raise RuntimeError("M2 down")
        pairs = []
        for i, a in enumerate(selections):
            for b in selections[i + 1:]:
                status, sev = self.rules.get(frozenset({a.drug_id, b.drug_id}), ("NO_KNOWN_INTERACTION_IN_DATABASE", None))
                findings = []
                if status == "INTERACTION_FOUND":
                    findings = [Finding(severity=sev or "MODERATE", description="Increased risk of bleeding when combined.",
                                        mechanism="Additive effect on platelet function.", recommendation="Monitor for bleeding signs.",
                                        source=SourceRef(source_name="DDInter 2.0"))]
                pairs.append(PairResult(selection_a=a.selection_id, selection_b=b.selection_id, status=status,
                                        highest_severity=sev, findings=findings))
        overall = "INTERACTION_FOUND" if any(p.status == "INTERACTION_FOUND" for p in pairs) else pairs[0].status
        return InteractionReport(check_id="chk-1", summary=ReportSummary(overall_status=overall, pair_count=len(pairs),
                                 evaluated_pair_count=len(pairs), is_complete=True), pairs=pairs,
                                 limitations=["Chỉ dựa trên dữ liệu tương tác đã được số hóa."])


def sample_health(now: datetime | None = None) -> RawHealthContext:
    now = now or datetime.now(timezone.utc)
    return RawHealthContext(
        profile=ProfileData(date_of_birth=date(1972, 3, 14), biological_sex="FEMALE", height_cm=158),
        conditions=[ConditionData("Tăng huyết áp", "CHRONIC"), ConditionData("Viêm xoang", "RESOLVED")],
        allergies=[AllergyData("Penicillin", severity="SEVERE", reaction="nổi mề đay")],
        medications=[MedicationData("Warfarin", WARFARIN, ("warfarin",), "5mg", "1 lần/ngày"),
                     MedicationData("Thuốc cũ", uuid4(), (), None, None, status="COMPLETED")],
        latest_vitals=[VitalData("BLOOD_PRESSURE", 130, now, 85), VitalData("WEIGHT", 62, now)],
    )
