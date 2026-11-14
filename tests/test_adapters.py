import json
from pathlib import Path

import pytest

from app.modules.ai_assistant.rag.ingestion.adapters import (
    convert, detect_format, drug_food_to_record, medlineplus_to_record, read_records, summary_to_markdown,
)
from app.modules.ai_assistant.rag.ingestion.chunker import HARD_CAP_TOKENS, CorpusRecord, HybridChunker, estimate_tokens
from app.modules.ai_assistant.rag.ingestion.cli import IngestReport

SAMPLES = Path(__file__).resolve().parents[1] / "data" / "sample"
NUTRITION = json.loads((SAMPLES / "nutrition_diet_guidelines_sample.json").read_text(encoding="utf-8"))
DFI = json.loads((SAMPLES / "drug_food_interaction_sample.json").read_text(encoding="utf-8"))


# ------------------------------------------------------------------ MedlinePlus
def test_medlineplus_maps_metadata_and_strips_attribution():
    r = medlineplus_to_record(NUTRITION[0])
    assert (r.external_id, r.title, r.source_name, r.source_url) == ("1558", "Alcohol", "MedlinePlus", "https://medlineplus.gov/alcohol.html")
    assert r.document_type == "TOPIC_SUMMARY" and r.language == "en" and r.aliases == ["Drinking"]
    assert r.categories == ["General Nutrition", "Substance Use and Disorders", "Food and Nutrition"]
    assert r.license_info == "NIH: National Institute on Alcohol Abuse and Alcoholism" and "NIH:" not in r.content
    assert r.extra["mesh_headings"] == ["Alcohol Drinking"]


def test_question_lines_become_markdown_headings_and_bullets_are_grouped():
    md = medlineplus_to_record(NUTRITION[0]).content
    heads = [l for l in md.splitlines() if l.startswith("## ")]
    assert heads == ["## How does alcohol affect the body?", "## Why are the effects of alcohol different from person to person?",
                     "## What is moderate drinking?", "## What is a standard drink?", "## Who should not drink alcohol?",
                     "## What is excessive drinking?"]
    assert "- How much you drink\n- How often you drink\n- Your age" in md  # bullet liền nhau, không còn dòng trống xen giữa
    assert "- Are pregnant or trying to become pregnant" in md


def test_one_chunk_per_question_section_and_each_keeps_context_header():
    chunks = HybridChunker(None).chunk_document(medlineplus_to_record(NUTRITION[0]))
    assert len(chunks) == 7
    assert all(c.content.startswith("[Tài liệu: Alcohol | Tên khác: Drinking | Nhóm: General Nutrition, ") for c in chunks)
    who = next(c for c in chunks if "Who should not drink alcohol?" in c.body)
    assert "Are pregnant" in who.body and "Are taking medicines that can interact with alcohol" in who.body  # danh sách không bị cắt rời


def test_text_without_headings_stays_a_single_chunk_under_cap():
    chunks = HybridChunker(None).chunk_document(medlineplus_to_record(NUTRITION[1]))
    assert len(chunks) == 1 and estimate_tokens(chunks[0].body) <= HARD_CAP_TOKENS


def test_merge_small_sections_is_opt_in_and_respects_hard_cap():
    r = medlineplus_to_record(NUTRITION[0])
    off, on = HybridChunker(None).chunk_document(r), HybridChunker(None, min_chunk_tokens=80).chunk_document(r)
    assert len(on) < len(off) and [c.index for c in on] == list(range(len(on)))
    assert all(estimate_tokens(c.body) <= HARD_CAP_TOKENS for c in on)
    big = HybridChunker(None, min_chunk_tokens=10_000).chunk_document(r)  # ngưỡng khổng lồ vẫn không vượt trần
    assert all(estimate_tokens(c.body) <= HARD_CAP_TOKENS for c in big)


def test_summary_without_attribution_or_headings():
    md, attr = summary_to_markdown("T", "Just one paragraph.\n\nAnother one.")
    assert attr is None and md == "# T\n\nJust one paragraph.\n\nAnother one."


# ------------------------------------------------------------------ Drug–food
def test_drug_food_record_is_single_bilingual_chunk_with_severity_metadata():
    r = drug_food_to_record(DFI[0])
    assert r.external_id == "dfi-1" and r.title == "Dextromethorphan — rượu bia" and r.document_type == "DRUG_FOOD_INTERACTION"
    assert r.aliases == ["Dextromethorphan"]  # display_name_vn trùng inn_name → khử trùng lặp
    [chunk] = HybridChunker(None).chunk_document(r)
    assert chunk.content.startswith("[Tài liệu: Dextromethorphan — rượu bia | Tên khác: Dextromethorphan | Nhóm: rượu bia]\nNội dung: ")
    assert "Tránh hoặc hạn chế tối đa rượu bia" in chunk.body and "3 or more alcoholic drinks" in chunk.body  # cả tiếng Việt lẫn bằng chứng gốc
    assert "mục warnings" in chunk.body and "ĐỎ" in chunk.body
    assert chunk.metadata["severity"] == "RED" and chunk.metadata["food_category"] == "ALCOHOL"
    assert chunk.metadata["source_ingredient_id"] == 375 and chunk.metadata["external_id"] == "dfi-1"


def test_drug_food_green_row_and_alias_order():
    r = drug_food_to_record(DFI[1])
    assert r.aliases == ["Cefpodoxim", "Cefpodoxime"] and r.categories == ["thời điểm uống thuốc so với bữa ăn"]
    [chunk] = HybridChunker(None).chunk_document(r)
    assert chunk.metadata["severity"] == "GREEN" and "XANH" in chunk.body


def test_drug_food_unknown_category_and_missing_fields():
    r = drug_food_to_record({"inn_name": "X", "food_category": "GRAPEFRUIT_JUICE", "severity": "YELLOW", "fda_evidence": "e", "ingredient_id": 1, "fda_section": "warnings"})
    assert r.categories == ["grapefruit juice"] and r.external_id == "dfi-1-grapefruit_juice-warnings" and "YELLOW" in r.content
    with pytest.raises(ValueError, match="inn_name"):
        drug_food_to_record({"fda_evidence": "e"})
    with pytest.raises(ValueError, match="clinical_advice_vn"):
        drug_food_to_record({"inn_name": "X"})


# ------------------------------------------------------------------ nhận diện & đọc file
def test_detect_format_per_record():
    assert detect_format(NUTRITION[0]) == "medlineplus" and detect_format(DFI[0]) == "drug-food"
    assert detect_format({"external_id": "a", "content": "b"}) == "golden"
    with pytest.raises(ValueError, match="nhận diện"):
        detect_format({"foo": 1})
    with pytest.raises(ValueError, match="object"):
        convert([1, 2])
    with pytest.raises(ValueError, match="source_name"):
        convert({"external_id": "a", "content": "b", "title": "t"})  # golden thiếu trường bắt buộc


def test_bundled_samples_load_without_errors():
    rep = IngestReport()
    recs = [r for f in sorted(SAMPLES.glob("*")) for r in read_records(f, rep)]
    assert rep.invalid == [] or all("corpus_sample.jsonl dòng 3" in x for x in rep.invalid)  # dòng 3 là ca lỗi cố ý
    assert {r.document_type for r in recs} >= {"TOPIC_SUMMARY", "DRUG_FOOD_INTERACTION", "MONOGRAPH"}


def test_json_array_single_object_and_jsonl(tmp_path):
    rep = IngestReport()
    arr = tmp_path / "a.json"
    arr.write_text(json.dumps(DFI), encoding="utf-8")
    single = tmp_path / "s.json"
    single.write_text(json.dumps(DFI[0]), encoding="utf-8")
    jl = tmp_path / "m.jsonl"
    jl.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in DFI), encoding="utf-8")
    assert [r.external_id for r in read_records(arr, rep)] == ["dfi-1", "dfi-2"]
    assert [r.external_id for r in read_records(single, rep)] == ["dfi-1"]
    assert [r.external_id for r in read_records(jl, rep)] == ["dfi-1", "dfi-2"] and rep.invalid == []


def test_bad_elements_and_broken_files_are_reported_with_position(tmp_path):
    rep = IngestReport()
    f = tmp_path / "x.json"
    f.write_text(json.dumps([DFI[0], {"foo": 1}, "text", {"inn_name": "Z"}, NUTRITION[0]]), encoding="utf-8")
    ids = [r.external_id for r in read_records(f, rep)]
    assert ids == ["dfi-1", "1558"] and len(rep.invalid) == 3
    assert "x.json phần tử 2" in rep.invalid[0] and "x.json phần tử 4" in rep.invalid[2]
    rep2, broken = IngestReport(), tmp_path / "b.json"
    broken.write_text('[{"a": 1,', encoding="utf-8")
    assert list(read_records(broken, rep2)) == [] and "JSON lỗi" in rep2.invalid[0]
    truncated = tmp_path / "t.jsonl"  # dòng cuối bị cắt giữa chừng
    truncated.write_text(json.dumps(DFI[0]) + '\n{"inn_na', encoding="utf-8")
    rep3 = IngestReport()
    assert len(list(read_records(truncated, rep3))) == 1 and "dòng 2" in rep3.invalid[0]


def test_forced_format_overrides_detection():
    with pytest.raises(ValueError):
        convert(DFI[0], "medlineplus")  # ép sai định dạng → báo lỗi, không đoán


def test_limit_applies_to_valid_records_only(tmp_path):
    rep = IngestReport()
    f = tmp_path / "l.json"
    f.write_text(json.dumps([{"foo": 1}, DFI[0], DFI[1]]), encoding="utf-8")
    assert [r.external_id for r in read_records(f, rep, limit=1)] == ["dfi-1"]
