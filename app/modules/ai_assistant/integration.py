"""Cổng tích hợp (ports) giữa M3 và M1/M2/Auth — VitaP_API_Docs.md §10.2.

M3 không import trực tiếp code M1/M2: nhóm Trọng cài đặt các Protocol dưới đây (adapter mỏng bọc service
thật) rồi truyền vào `build_container(...)`. Giả định: các hàm là `async`; nếu service M1/M2 là sync, bọc bằng
`anyio.to_thread.run_sync` trong adapter.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Literal, Protocol, Sequence, runtime_checkable
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from .enums import PersonalContextSection


# ---------------------------------------------------------------- Principal
@runtime_checkable
class Principal(Protocol):
    """Chủ thể đã xác thực do Auth cung cấp. M3 chỉ phục vụ User (is_guest=False)."""

    @property
    def user_id(self) -> UUID: ...

    @property
    def is_guest(self) -> bool: ...


# ---------------------------------------------------------------- M2: interaction report
class _Lenient(BaseModel):
    model_config = ConfigDict(extra="ignore")


class SourceRef(_Lenient):
    source_id: UUID | str | None = None
    source_name: str | None = None
    version: str | None = None
    source_url: str | None = None


class Finding(_Lenient):
    interaction_id: UUID | str | None = None
    ingredient_a_id: UUID | str | None = None
    ingredient_b_id: UUID | str | None = None
    severity: str
    description: str
    mechanism: str | None = None
    recommendation: str | None = None
    evidence_level: str | None = None
    source: SourceRef | None = None


class PairResult(_Lenient):
    selection_a: str
    selection_b: str
    status: str  # InteractionStatus
    highest_severity: str | None = None
    findings: list[Finding] = Field(default_factory=list)
    missing_data: list[str] = Field(default_factory=list)
    sources: list[SourceRef] = Field(default_factory=list)


class ReportSummary(_Lenient):
    overall_status: str
    pair_count: int = 0
    evaluated_pair_count: int = 0
    is_complete: bool = False


class InteractionReport(_Lenient):
    check_id: str | None = None
    summary: ReportSummary
    pairs: list[PairResult] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)


@dataclass(frozen=True)
class DrugSelection:
    selection_id: str
    drug_id: UUID


@dataclass(frozen=True)
class DrugResolution:
    resolution_status: str  # MATCHED | AMBIGUOUS | UNRESOLVED
    candidate_drug_id: UUID | None = None
    match_score: float | None = None


@dataclass(frozen=True)
class DrugSummary:
    drug_id: UUID
    name: str
    ingredient_names: tuple[str, ...] = ()


# ---------------------------------------------------------------- M1: personal health context
@dataclass(frozen=True)
class ProfileData:
    date_of_birth: date | None = None
    biological_sex: str | None = None
    height_cm: float | None = None


@dataclass(frozen=True)
class ConditionData:
    condition_name: str
    status: str = "ACTIVE"  # ACTIVE | CHRONIC | RESOLVED
    icd10_code: str | None = None
    verified: bool | None = None


@dataclass(frozen=True)
class AllergyData:
    allergen_name: str
    allergen_type: str | None = None
    severity: str | None = None
    reaction: str | None = None
    ingredient_id: UUID | None = None
    verified: bool | None = None


@dataclass(frozen=True)
class MedicationData:
    drug_name: str
    drug_id: UUID | None = None
    ingredient_names: tuple[str, ...] = ()
    dosage: str | None = None
    frequency: str | None = None
    status: str = "ACTIVE"


@dataclass(frozen=True)
class VitalData:
    observation_type: str  # BLOOD_PRESSURE | HEART_RATE | BODY_TEMPERATURE | WEIGHT | SPO2
    value: float
    measured_at: datetime
    secondary_value: float | None = None
    unit: str | None = None


@dataclass
class RawHealthContext:
    profile: ProfileData | None = None
    conditions: list[ConditionData] = field(default_factory=list)
    allergies: list[AllergyData] = field(default_factory=list)
    medications: list[MedicationData] = field(default_factory=list)
    latest_vitals: list[VitalData] = field(default_factory=list)  # mỗi loại chỉ số: bản ghi mới nhất


# ---------------------------------------------------------------- Protocols
class HealthContextProvider(Protocol):
    async def get_personal_health_context(
        self, principal: Principal, requested_sections: frozenset[PersonalContextSection]
    ) -> RawHealthContext:
        """`GetPersonalHealthContext`: chỉ trả đúng nhóm được yêu cầu, đã lọc theo principal."""
        ...


class InteractionChecker(Protocol):
    async def check_drug_interactions(
        self, principal: Principal, selections: Sequence[DrugSelection]
    ) -> InteractionReport | dict[str, Any]:
        """`CheckDrugInteractions`: cùng Rule Engine với POST /interactions/check (2…10 lựa chọn)."""
        ...


class DrugEntityResolver(Protocol):
    async def resolve_drug_entity(self, extracted_name: str, extracted_strength: str | None = None) -> DrugResolution:
        """`ResolveDrugEntity`: không dùng LLM đoán; chỉ MATCHED mới được M3 chấp nhận."""
        ...


class DrugDirectory(Protocol):
    async def get_drug_summaries(self, drug_ids: Sequence[UUID]) -> dict[UUID, DrugSummary]:
        """Tên thuốc/hoạt chất để diễn giải (interface bổ sung M3 cần từ M2, xem README)."""
        ...


Literal  # re-export guard for linters
