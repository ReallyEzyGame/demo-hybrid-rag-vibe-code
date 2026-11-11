"""Query Understanding: nhận diện thuốc được nhắc trong câu hỏi qua ResolveDrugEntity của M2 (không dùng LLM đoán)."""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from uuid import UUID

from ..integration import DrugEntityResolver
from ..rag.guards.text_utils import fold

logger = logging.getLogger(__name__)

_TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9\-]{3,}")
_STOP = frozenset(
    "thuoc uong dung chung cung voi them toi minh duoc khong nhu the nao bao nhieu co sao la gi nay kia nhung "
    "ngay lan sang trua toi truoc sau bua can nen thi hay cho nguoi lon tre em benh bac si duoc si and the for "
    "with what does that this have from your about should take taking drug drugs medicine".split()
)


@dataclass(frozen=True)
class DrugMention:
    surface: str
    drug_id: UUID


def candidate_names(text: str, limit: int) -> list[str]:
    """Từ ASCII dài ≥4 ký tự (tên thuốc quốc tế/biệt dược thường là ASCII); từ tiếng Việt có dấu bị loại."""
    seen: dict[str, str] = {}
    for m in _TOKEN.finditer(text):
        tok = m.group(0)
        key = tok.lower()
        if fold(key) in _STOP or key in seen:
            continue
        seen[key] = tok
    return list(seen.values())[:limit]


class DrugMentionExtractor:
    def __init__(self, resolver: DrugEntityResolver, max_candidates: int = 12) -> None:
        self.resolver, self.max_candidates = resolver, max_candidates

    async def extract(self, text: str) -> list[DrugMention]:
        found: dict[UUID, DrugMention] = {}
        for name in candidate_names(text, self.max_candidates):
            try:
                res = await self.resolver.resolve_drug_entity(name, None)
            except Exception:  # noqa: BLE001 - lỗi M2 với một từ không làm hỏng cả câu hỏi
                logger.warning("resolve_drug_entity failed for a candidate")
                continue
            # Chỉ chấp nhận MATCHED. AMBIGUOUS/UNRESOLVED bị bỏ qua, không đoán.
            if res.resolution_status == "MATCHED" and res.candidate_drug_id and res.candidate_drug_id not in found:
                found[res.candidate_drug_id] = DrugMention(name, res.candidate_drug_id)
        return list(found.values())
