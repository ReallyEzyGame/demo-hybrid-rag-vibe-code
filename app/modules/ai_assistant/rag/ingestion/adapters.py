"""Chuyển các định dạng JSON nguồn của nhóm thành CorpusRecord (Golden Format) trước khi chunk.

Hỗ trợ (tự nhận diện theo từng bản ghi):
  * golden       — {external_id, title, source_name, content, …}                      (.jsonl, mỗi dòng một object)
  * medlineplus  — {topic_id, title, url, summary, also_called, groups, …}            (.json mảng — file nutrition/diet guidelines)
  * drug-food    — {ingredient_id, inn_name, food_category, severity, fda_evidence, …} (.json mảng — drug_food_interaction.json)
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable, Iterator

from pydantic import ValidationError

from .chunker import CorpusRecord

FORMATS = ("auto", "golden", "medlineplus", "drug-food")

# --------------------------------------------------------------------------- MedlinePlus topic
_ATTRIBUTION = re.compile(r"^(NIH|CDC|FDA|NLM|WHO|AHRQ|SAMHSA|HHS):\s+\S")
_HEADING_MAX_CHARS = 140


def _is_heading(line: str) -> bool:
    return line.endswith("?") and len(line) <= _HEADING_MAX_CHARS and not line.startswith("- ")


def summary_to_markdown(title: str, summary: str) -> tuple[str, str | None]:
    """Văn bản thuần của MedlinePlus → markdown.

    - Dòng đầu của một đoạn kết thúc bằng '?' (tiêu đề dạng câu hỏi, có thể dính liền nội dung bên dưới) → `## …`
      để chunker tách theo mục.
    - Các gạch đầu dòng bị cách nhau bởi dòng trống được gom lại thành một danh sách.
    - Dòng ghi nguồn cuối bài ('NIH: …') được tách ra và trả về riêng.
    Trả về (markdown, attribution | None).
    """
    paras: list[list[str]] = []
    for raw in re.split(r"\n\s*\n", summary.strip()):
        lines = [" ".join(l.split()) for l in raw.splitlines()]
        lines = [l for l in lines if l]
        if lines:
            paras.append(lines)
    attribution = None
    if paras and len(paras[-1]) == 1 and _ATTRIBUTION.match(paras[-1][0]):
        attribution = paras.pop()[0]
    blocks: list[str] = []
    for lines in paras:
        if _is_heading(lines[0]):
            blocks.append(f"## {lines[0]}")
            lines = lines[1:]
            if not lines:
                continue
        text = "\n".join(lines)
        if text.startswith("- ") and blocks and blocks[-1].startswith("- "):
            blocks[-1] += "\n" + text  # gom bullet liên tiếp thành một danh sách
        else:
            blocks.append(text)
    return f"# {title}\n\n" + "\n\n".join(blocks), attribution


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out = []
    for x in items:
        x = (x or "").strip()
        if x and x.lower() not in seen:
            seen.add(x.lower())
            out.append(x)
    return out


def medlineplus_to_record(obj: dict[str, Any]) -> CorpusRecord:
    for key in ("topic_id", "title", "summary"):
        if not str(obj.get(key) or "").strip():
            raise ValueError(f"thiếu trường '{key}'")
    content, attribution = summary_to_markdown(obj["title"].strip(), obj["summary"])
    mesh = [m.get("name") for m in obj.get("mesh_headings") or [] if isinstance(m, dict) and m.get("name")]
    return CorpusRecord(
        external_id=str(obj["topic_id"]),
        title=obj["title"].strip(),
        document_type="TOPIC_SUMMARY",
        source_name="MedlinePlus",
        source_url=obj.get("url"),
        language="en",
        aliases=_dedupe(list(obj.get("also_called") or [])),
        categories=_dedupe([obj.get("category") or ""] + list(obj.get("groups") or [])),
        content=content,
        license_info=attribution,
        extra={"mesh_headings": mesh, "meta_description": obj.get("meta_description")},
    )


# --------------------------------------------------------------------------- Drug–food interaction
_FOOD_VI = {"ALCOHOL": "rượu bia", "FOOD_TIMING": "thời điểm uống thuốc so với bữa ăn"}
_SEVERITY_VI = {
    "RED": "ĐỎ — nguy hiểm, cần tránh hoặc hạn chế tối đa",
    "GREEN": "XANH — lưu ý thông thường, chưa ghi nhận nguy cơ nghiêm trọng trong nhãn thuốc",
}
DRUG_FOOD_SOURCE = "FDA drug labeling"  # JSON không ghi URL/nhãn cụ thể → đề nghị bổ sung (xem README)


def drug_food_to_record(obj: dict[str, Any]) -> CorpusRecord:
    inn = str(obj.get("inn_name") or "").strip()
    if not inn:
        raise ValueError("thiếu trường 'inn_name'")
    advice = str(obj.get("clinical_advice_vn") or "").strip()
    evidence = str(obj.get("fda_evidence") or "").strip()
    if not advice and not evidence:
        raise ValueError("thiếu cả 'clinical_advice_vn' lẫn 'fda_evidence'")
    category = str(obj.get("food_category") or "UNKNOWN").strip().upper()
    severity = str(obj.get("severity") or "").strip().upper()
    section = str(obj.get("fda_section") or "").strip()
    name = (obj.get("display_name_vn") or inn).strip()
    food = _FOOD_VI.get(category, category.lower().replace("_", " "))
    ext = obj.get("id")
    external_id = f"dfi-{ext}" if ext is not None else f"dfi-{obj.get('ingredient_id')}-{category.lower()}-{section}"
    lines = [f"# {name} ({inn}) — tương tác với {food}"]
    if severity:
        lines.append(f"Mức cảnh báo: {_SEVERITY_VI.get(severity, severity)}.")
    if advice:
        lines += ["", "## Khuyến nghị lâm sàng", advice]
    if evidence:
        lines += ["", f"## Bằng chứng từ nhãn thuốc FDA (mục {section or 'không rõ'})", evidence]
    return CorpusRecord(
        external_id=external_id,
        title=f"{name} — {food}",
        document_type="DRUG_FOOD_INTERACTION",
        source_name=DRUG_FOOD_SOURCE,
        language="vi",
        aliases=_dedupe([name, inn]),
        categories=[food],
        content="\n".join(lines),
        chunk_strategy="single",
        extra={
            "severity": severity or None, "food_category": category, "fda_section": section or None,
            "inn_name": inn, "source_ingredient_id": obj.get("ingredient_id"), "source_row_id": ext,
        },
    )


# --------------------------------------------------------------------------- phát hiện định dạng + đọc file
def golden_to_record(obj: dict[str, Any]) -> CorpusRecord:
    return CorpusRecord.model_validate(obj)


_CONVERTERS: dict[str, Callable[[dict[str, Any]], CorpusRecord]] = {
    "golden": golden_to_record, "medlineplus": medlineplus_to_record, "drug-food": drug_food_to_record,
}


def detect_format(obj: dict[str, Any]) -> str:
    if "topic_id" in obj and "summary" in obj:
        return "medlineplus"
    if "inn_name" in obj and ("fda_evidence" in obj or "clinical_advice_vn" in obj):
        return "drug-food"
    if "external_id" in obj and "content" in obj:
        return "golden"
    raise ValueError("không nhận diện được định dạng bản ghi")


def convert(obj: Any, fmt: str = "auto") -> CorpusRecord:
    if not isinstance(obj, dict):
        raise ValueError("bản ghi phải là object JSON")
    name = detect_format(obj) if fmt == "auto" else fmt
    try:
        return _CONVERTERS[name](obj)
    except ValidationError as exc:
        raise ValueError(exc.errors()[0]["msg"] + f" ({'.'.join(str(x) for x in exc.errors()[0]['loc'])})") from exc


def _iter_raw(path: Path) -> Iterator[tuple[str, Any]]:
    """Sinh (nhãn vị trí, object|Exception). .jsonl: từng dòng; .json: mảng hoặc một object."""
    text = path.read_text(encoding="utf-8")
    whole = None
    if path.suffix.lower() != ".jsonl":
        try:
            whole = json.loads(text)
        except json.JSONDecodeError as exc:
            if "\n" not in text.strip():  # một dòng mà vẫn lỗi → báo lỗi cả file
                yield "file", exc
                return
    if whole is None:  # .jsonl (hoặc file nhiều dòng không phải một JSON hợp lệ): đọc từng dòng
        for lineno, line in enumerate(text.splitlines(), start=1):
            if line.strip():
                try:
                    yield f"dòng {lineno}", json.loads(line)
                except json.JSONDecodeError as exc:
                    yield f"dòng {lineno}", exc
        return
    for i, item in enumerate(whole if isinstance(whole, list) else [whole], start=1):
        yield f"phần tử {i}", item


def read_records(path: Path, report, limit: int | None = None, fmt: str = "auto") -> Iterator[CorpusRecord]:  # noqa: ANN001
    n = 0
    for where, raw in _iter_raw(path):
        try:
            if isinstance(raw, Exception):
                raise ValueError(f"JSON lỗi: {raw}")
            rec = convert(raw, fmt)
        except ValueError as exc:
            report.invalid.append(f"{path.name} {where}: {exc}")
            continue
        yield rec
        n += 1
        if limit and n >= limit:
            return
