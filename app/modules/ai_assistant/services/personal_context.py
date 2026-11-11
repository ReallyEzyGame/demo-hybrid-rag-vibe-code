"""PersonalContextService — thiết kế §6.1. Chỉ lấy dữ liệu tối thiểu, bỏ PII, chỉ thuốc ACTIVE, chỉ bệnh ACTIVE/CHRONIC."""
from __future__ import annotations

import logging
import re
from datetime import date, datetime, timezone
from typing import Callable

from ..config import AiAssistantSettings
from ..domain import PersonalContext
from ..enums import ContextMode, PersonalContextSection as S
from ..integration import HealthContextProvider, Principal, RawHealthContext
from ..rag.guards.text_utils import fold

logger = logging.getLogger(__name__)

_FIRST_PERSON = re.compile(r"\b(toi|minh|em|con toi|bo toi|me toi|cua toi|cua minh|ban than|ho so)\b")
_DRUG_WORDS = re.compile(r"\b(thuoc|lieu|vien|uong|tiem|tac dung phu|di ung|chong chi dinh|ke don|don thuoc|dung chung|ket hop)\b")
_VITAL_WORDS = re.compile(r"\b(huyet ap|nhip tim|nhiet do|sot|can nang|bmi|spo2|oxy|chi so|suc khoe)\b")
_CONDITION_WORDS = re.compile(r"\b(benh|tieu duong|tang huyet|tim mach|mo mau|gout|hen|viem|ung thu|mang thai|thai ky|chan doan|trieu chung|ket qua|xet nghiem|a1c|duong huyet)\b")
_INTERACTION_WORDS = re.compile(r"\b(uong chung|dung chung|ket hop|cung luc|cung voi|uong them|dung them|tuong tac|dong thoi)\b")

VITAL_LABEL = {
    "BLOOD_PRESSURE": "Huyết áp",
    "HEART_RATE": "Nhịp tim",
    "BODY_TEMPERATURE": "Thân nhiệt",
    "WEIGHT": "Cân nặng",
    "SPO2": "SpO2",
}
VITAL_UNIT = {"BLOOD_PRESSURE": "mmHg", "HEART_RATE": "lần/phút", "BODY_TEMPERATURE": "°C", "WEIGHT": "kg", "SPO2": "%"}


def has_interaction_intent(question: str) -> bool:
    return bool(_INTERACTION_WORDS.search(fold(question)))


def select_sections(question: str) -> frozenset[S]:
    """Chọn đúng nhóm dữ liệu cần cho câu hỏi (tối thiểu hóa dữ liệu gửi LLM bên thứ ba).

    - Câu hỏi thuần kiến thức chung (không đại từ ngôi thứ nhất, không nhắc thuốc/chỉ số) → không lấy gì.
    - Có nhắc thuốc / tương tác → luôn kèm DỊ ỨNG + THUỐC ĐANG DÙNG (để cảnh báo an toàn).
    """
    q = fold(question)
    sections: set[S] = set()
    personal = bool(_FIRST_PERSON.search(q))
    drug = bool(_DRUG_WORDS.search(q))
    if drug or has_interaction_intent(question):
        sections |= {S.ALLERGIES, S.ACTIVE_MEDICATIONS}
    if _CONDITION_WORDS.search(q) and personal:
        sections |= {S.HEALTH_CONDITIONS, S.HEALTH_PROFILE}
    if _VITAL_WORDS.search(q) and personal:
        sections |= {S.RECENT_VITALS, S.HEALTH_PROFILE}
    if personal and not sections:
        sections.add(S.HEALTH_PROFILE)
    if S.HEALTH_CONDITIONS in sections or S.ACTIVE_MEDICATIONS in sections:
        sections.add(S.HEALTH_PROFILE)  # tuổi/giới giúp diễn giải liều, chống chỉ định
    return frozenset(sections)


def _age(dob: date | None, today: date) -> int | None:
    if not dob:
        return None
    return today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))


def _fmt_num(x: float) -> str:
    return f"{x:g}"


class PersonalContextService:
    def __init__(self, provider: HealthContextProvider, settings: AiAssistantSettings, clock: Callable[[], datetime] | None = None) -> None:
        self.provider, self.s = provider, settings
        self._clock = clock or (lambda: datetime.now(timezone.utc))

    async def build(
        self,
        principal: Principal,
        mode: ContextMode | str,
        question: str,
        *,
        force_sections: frozenset[S] = frozenset(),
    ) -> PersonalContext:
        if str(mode) != ContextMode.PERSONAL.value:
            return PersonalContext()  # GENERAL: tuyệt đối không truy xuất dữ liệu cá nhân
        sections = select_sections(question) | force_sections
        if not sections:
            return PersonalContext()
        try:
            raw = await self.provider.get_personal_health_context(principal, sections)
        except Exception:  # noqa: BLE001 - M1 lỗi không được làm sập chat
            logger.exception("PersonalContext provider failed")
            return PersonalContext(unavailable=True)
        return self.render(raw, sections)

    def render(self, raw: RawHealthContext, sections: frozenset[S]) -> PersonalContext:
        now = self._clock()
        lines: list[str] = []
        used: list[str] = []
        active_ids, active_names = [], {}

        if S.HEALTH_PROFILE in sections and raw.profile:
            p = raw.profile
            bits = []
            if p.biological_sex:
                bits.append({"MALE": "nam", "FEMALE": "nữ"}.get(p.biological_sex.upper(), p.biological_sex.lower()))
            age = _age(p.date_of_birth, now.date())
            if age is not None:
                bits.append(f"{age} tuổi")
            if p.height_cm:
                bits.append(f"cao {_fmt_num(p.height_cm)} cm")
            if bits:  # KHÔNG đưa họ tên, email, SĐT, địa chỉ
                lines.append("Thông tin cơ bản: " + ", ".join(bits) + ".")
                used.append(S.HEALTH_PROFILE.value)

        if S.HEALTH_CONDITIONS in sections:
            conds = [c for c in raw.conditions if c.status.upper() in ("ACTIVE", "CHRONIC")]
            if conds:
                items = [c.condition_name + (f" (ICD-10 {c.icd10_code})" if c.icd10_code else "") for c in conds]
                lines.append("Bệnh nền đang theo dõi (người dùng tự khai báo, chưa xác minh): " + "; ".join(items) + ".")
                used.append(S.HEALTH_CONDITIONS.value)

        if S.ALLERGIES in sections and raw.allergies:
            items = []
            for a in raw.allergies:
                extra = ", ".join(x for x in [f"mức {a.severity}" if a.severity else "", f"phản ứng: {a.reaction}" if a.reaction else ""] if x)
                items.append(a.allergen_name + (f" ({extra})" if extra else ""))
            lines.append("Dị ứng (người dùng tự khai báo): " + "; ".join(items) + ".")
            used.append(S.ALLERGIES.value)

        if S.ACTIVE_MEDICATIONS in sections:
            meds = [m for m in raw.medications if m.status.upper() == "ACTIVE"]  # không lấy đơn thuốc cũ
            if meds:
                items = []
                for m in meds:
                    detail = ", ".join(
                        x for x in [
                            f"hoạt chất: {', '.join(m.ingredient_names)}" if m.ingredient_names else "",
                            f"liều: {m.dosage}" if m.dosage else "",
                            f"tần suất: {m.frequency}" if m.frequency else "",
                        ] if x
                    )
                    items.append(m.drug_name + (f" ({detail})" if detail else ""))
                    if m.drug_id and m.drug_id not in active_names:
                        active_ids.append(m.drug_id)
                        active_names[m.drug_id] = m.drug_name
                lines.append("Thuốc đang dùng: " + "; ".join(items) + ".")
                used.append(S.ACTIVE_MEDICATIONS.value)

        if S.RECENT_VITALS in sections and raw.latest_vitals:
            parts = []
            height = raw.profile.height_cm if raw.profile else None
            for v in raw.latest_vitals:
                label = VITAL_LABEL.get(v.observation_type, v.observation_type)
                val = _fmt_num(v.value) + (f"/{_fmt_num(v.secondary_value)}" if v.secondary_value is not None else "")
                unit = v.unit or VITAL_UNIT.get(v.observation_type, "")
                when = v.measured_at if v.measured_at.tzinfo else v.measured_at.replace(tzinfo=timezone.utc)
                days = (now - when).days
                age_note = f", đã cũ {days} ngày" if days > self.s.stale_vitals_days else ""
                extra = ""
                if v.observation_type == "WEIGHT" and height:
                    extra = f", BMI {v.value / ((height / 100) ** 2):.1f}"
                parts.append(f"{label} {val} {unit} (đo {when.astimezone(timezone.utc):%Y-%m-%d}{age_note}{extra})".replace("  ", " "))
            lines.append("Sinh hiệu gần nhất: " + "; ".join(parts) + ".")
            used.append(S.RECENT_VITALS.value)

        return PersonalContext(used, "\n".join(lines), active_ids, active_names)
