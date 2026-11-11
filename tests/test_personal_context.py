from datetime import date, datetime, timedelta, timezone
from uuid import uuid4

from app.modules.ai_assistant.config import AiAssistantSettings
from app.modules.ai_assistant.enums import ContextMode, PersonalContextSection as S
from app.modules.ai_assistant.integration import ProfileData, RawHealthContext, VitalData
from app.modules.ai_assistant.services.personal_context import PersonalContextService, select_sections
from tests.support import FakeHealthProvider, FakePrincipal, sample_health

NOW = datetime(2026, 11, 10, 12, 0, tzinfo=timezone.utc)
P = FakePrincipal(uuid4())


def svc(provider):  # noqa: ANN001, ANN201
    return PersonalContextService(provider, AiAssistantSettings(_env_file=None), clock=lambda: NOW)


async def test_general_mode_never_calls_provider():
    prov = FakeHealthProvider(sample_health(NOW))
    ctx = await svc(prov).build(P, ContextMode.GENERAL, "Tôi đang uống thuốc gì đó")
    assert ctx.is_empty and prov.requests == []


async def test_pure_knowledge_question_requests_nothing():
    prov = FakeHealthProvider(sample_health(NOW))
    ctx = await svc(prov).build(P, ContextMode.PERSONAL, "Vitamin C có tác dụng gì")
    assert ctx.is_empty and prov.requests == []


async def test_renders_minimal_pii_free_active_only_text():
    raw = sample_health(NOW)
    ctx = await svc(FakeHealthProvider(raw)).build(P, ContextMode.PERSONAL, "Tôi đang uống thuốc và bị tăng huyết áp, huyết áp tôi 130/85", force_sections=frozenset(S))
    t = ctx.prompt_text
    assert "nữ, 54 tuổi, cao 158 cm" in t and "Tăng huyết áp" in t and "Penicillin" in t and "Warfarin" in t
    assert "Viêm xoang" not in t and "Thuốc cũ" not in t  # RESOLVED / không ACTIVE bị loại
    assert "130/85 mmHg" in t and "BMI 24.8" in t and "tự khai báo" in t
    assert set(ctx.sections_used) == {s.value for s in S}
    assert len(ctx.active_drug_ids) == 1 and "Warfarin" in ctx.active_drug_names.values()


def test_age_is_exact_across_birthday():
    raw = RawHealthContext(profile=ProfileData(date_of_birth=date(1972, 11, 11), biological_sex="MALE"))
    ctx = svc(None).render(raw, frozenset({S.HEALTH_PROFILE}))  # type: ignore[arg-type]
    assert "53 tuổi" in ctx.prompt_text and "nam" in ctx.prompt_text  # chưa tới sinh nhật 11/11


def test_stale_vitals_are_flagged():
    old = NOW - timedelta(days=45)
    raw = RawHealthContext(latest_vitals=[VitalData("HEART_RATE", 72, old)])
    t = svc(None).render(raw, frozenset({S.RECENT_VITALS})).prompt_text  # type: ignore[arg-type]
    assert "đã cũ 45 ngày" in t


async def test_provider_failure_marks_unavailable_instead_of_raising():
    ctx = await svc(FakeHealthProvider(fail=True)).build(P, ContextMode.PERSONAL, "Tôi đang uống thuốc")
    assert ctx.unavailable and ctx.is_empty


def test_drug_or_interaction_questions_always_include_allergies_and_meds():
    for q in ["Uống chung A và B được không", "Tôi uống thuốc này có tác dụng phụ gì"]:
        assert {S.ALLERGIES, S.ACTIVE_MEDICATIONS} <= select_sections(q)
    assert select_sections("Tiểu đường là gì") == frozenset()
