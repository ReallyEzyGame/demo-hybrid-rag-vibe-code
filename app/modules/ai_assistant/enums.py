"""Enum dùng chung của Module 3. Giá trị khớp API contract (VitaP_API_Docs.md §9) và schema v3."""
from __future__ import annotations

from enum import StrEnum

CONSENT_SCOPE = "HEALTH_DATA_AND_AI_PROCESSING"


class ContextMode(StrEnum):
    PERSONAL = "PERSONAL"
    GENERAL = "GENERAL"


class MessageRole(StrEnum):
    USER = "USER"
    ASSISTANT = "ASSISTANT"
    SYSTEM = "SYSTEM"


class AnswerStatus(StrEnum):
    ANSWERED = "ANSWERED"
    PARTIAL = "PARTIAL"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    CONFLICTING_EVIDENCE = "CONFLICTING_EVIDENCE"
    NEEDS_CLINICIAN = "NEEDS_CLINICIAN"
    EMERGENCY_REDIRECT = "EMERGENCY_REDIRECT"
    OUT_OF_SCOPE = "OUT_OF_SCOPE"


class WarningLevel(StrEnum):
    """Giá trị phía API (Frontend). Cột DB dùng bộ NORMAL/CAUTION/WARNING/CRITICAL (schema v3)."""

    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


_TO_DB = {WarningLevel.INFO: "NORMAL", WarningLevel.WARNING: "WARNING", WarningLevel.CRITICAL: "CRITICAL"}
_FROM_DB = {
    "NORMAL": WarningLevel.INFO,
    "INFO": WarningLevel.INFO,
    "CAUTION": WarningLevel.WARNING,
    "WARNING": WarningLevel.WARNING,
    "CRITICAL": WarningLevel.CRITICAL,
}


def warning_to_db(level: WarningLevel) -> str:
    return _TO_DB[level]


def warning_from_db(value: str | None) -> WarningLevel:
    return _FROM_DB.get((value or "").upper(), WarningLevel.INFO)


class CitationType(StrEnum):
    MEDICAL_DOCUMENT = "MEDICAL_DOCUMENT"
    DRUG_DATABASE = "DRUG_DATABASE"
    INTERACTION_RULE = "INTERACTION_RULE"
    PERSONAL_PROFILE = "PERSONAL_PROFILE"


class FeedbackRating(StrEnum):
    THUMBS_UP = "THUMBS_UP"
    THUMBS_DOWN = "THUMBS_DOWN"


class InteractionStatus(StrEnum):
    """Bốn trạng thái của Rule Engine M2 (VitaP_API_Docs.md §8.2)."""

    INTERACTION_FOUND = "INTERACTION_FOUND"
    UNRESOLVED_DRUG = "UNRESOLVED_DRUG"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"
    NO_KNOWN_INTERACTION_IN_DATABASE = "NO_KNOWN_INTERACTION_IN_DATABASE"


class Severity(StrEnum):
    MINOR = "MINOR"
    MODERATE = "MODERATE"
    MAJOR = "MAJOR"
    CONTRAINDICATED = "CONTRAINDICATED"


class PersonalContextSection(StrEnum):
    HEALTH_PROFILE = "HEALTH_PROFILE"
    HEALTH_CONDITIONS = "HEALTH_CONDITIONS"
    ALLERGIES = "ALLERGIES"
    ACTIVE_MEDICATIONS = "ACTIVE_MEDICATIONS"
    RECENT_VITALS = "RECENT_VITALS"
