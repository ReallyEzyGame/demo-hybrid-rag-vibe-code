from app.modules.ai_assistant.rag.ingestion.chunker import HARD_CAP_TOKENS, CorpusRecord, HybridChunker, estimate_tokens
from tests.support import HashingEmbeddings


def rec(content, **kw):
    return CorpusRecord(external_id="x", title="Tiểu đường", source_name="S", content=content, **kw)


def test_splits_by_markdown_headers_and_keeps_heading_path():
    c = HybridChunker(None).chunk_document(rec("# Tổng quan\nĐoạn A.\n\n## Triệu chứng\nĐoạn B.\n\n## Điều trị\nĐoạn C."))
    assert [x.body.split("\n")[-1] for x in c] == ["Đoạn A.", "Đoạn B.", "Đoạn C."]
    assert c[1].metadata["heading_path"] == ["Tổng quan", "Triệu chứng"] and [x.index for x in c] == [0, 1, 2]


def test_header_context_is_prepended_and_optional_parts_omitted():
    full = HybridChunker(None).chunk_document(rec("# H\nbody", aliases=["Diabetes", "DM"], categories=["nội tiết"]))[0]
    assert full.content == "[Tài liệu: Tiểu đường | Tên khác: Diabetes, DM | Nhóm: nội tiết]\nNội dung: # H\nbody"
    bare = HybridChunker(None).chunk_document(rec("# H\nbody"))[0]
    assert bare.content.startswith("[Tài liệu: Tiểu đường]\nNội dung: ")


def test_hard_cap_is_enforced_on_giant_section_without_headers():
    text = " ".join(f"Câu số {i} nói về đường huyết và insulin." for i in range(400))
    chunks = HybridChunker(None).chunk_document(rec(text))
    assert len(chunks) > 3 and all(estimate_tokens(c.body) <= HARD_CAP_TOKENS for c in chunks)


def test_semantic_tier_runs_for_long_sections_and_still_respects_cap():
    sentences = [f"Sentence {i} about {'glucose insulin' if i < 60 else 'kidney creatinine'} control and monitoring." for i in range(120)]
    chunks = HybridChunker(HashingEmbeddings()).chunk_document(rec("## Long\n" + " ".join(sentences)))
    assert len(chunks) >= 2 and all(estimate_tokens(c.body) <= HARD_CAP_TOKENS for c in chunks)


def test_empty_content_gives_no_chunks():
    assert HybridChunker(None).chunk_document(rec("   \n\n ")) == []
