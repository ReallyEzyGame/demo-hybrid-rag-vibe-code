import httpx
import pytest

from app.modules.ai_assistant.rag.reranking.bge_client import BgeRerankerClient, RerankerUnavailable
from app.modules.ai_assistant.rate_limit import SlidingWindowRateLimiter
from app.modules.ai_assistant.services.chat_service import retrieval_query, strip_context_header
from app.modules.ai_assistant.services.drug_mentions import DrugMentionExtractor, candidate_names
from app.modules.ai_assistant.models import ChatMessage
from tests.support import FakeResolver, IBUPROFEN, PARACETAMOL


def client(handler):  # noqa: ANN001, ANN201
    return BgeRerankerClient("http://rr/", 1.0, httpx.AsyncClient(transport=httpx.MockTransport(handler)))


async def test_reranker_ok_clamps_scores_and_sends_ngrok_header():
    seen = {}

    def h(req):  # noqa: ANN001, ANN202
        seen["headers"], seen["url"] = req.headers, str(req.url)
        return httpx.Response(200, json={"scores": [0.9, 1.7, -0.2]})

    assert await client(h).score("q", ["a", "b", "c"]) == [0.9, 1.0, 0.0]
    assert seen["url"] == "http://rr/rerank" and seen["headers"]["ngrok-skip-browser-warning"] == "true"


@pytest.mark.parametrize("resp", [
    httpx.Response(500), httpx.Response(200, json={"oops": 1}), httpx.Response(200, json={"scores": [0.1]}),
    httpx.Response(200, json={"scores": ["x", "y"]}), httpx.Response(200, text="not json"),
])
async def test_reranker_bad_responses_raise_unavailable(resp):
    with pytest.raises(RerankerUnavailable):
        await client(lambda r: resp).score("q", ["a", "b"])


async def test_reranker_timeout_raises_unavailable():
    def h(req):  # noqa: ANN001, ANN202
        raise httpx.ReadTimeout("slow")

    with pytest.raises(RerankerUnavailable):
        await client(h).score("q", ["a"])


def test_rate_limiter_sliding_window():
    t = [0.0]
    rl = SlidingWindowRateLimiter(3, 60, clock=lambda: t[0])
    assert [rl.hit("u")[0] for _ in range(3)] == [True] * 3
    ok, retry = rl.hit("u")
    assert not ok and retry == 60 and rl.hit("other")[0]
    t[0] = 30
    assert rl.hit("u") == (False, 30)
    t[0] = 61
    assert rl.hit("u")[0]


def test_candidate_names_keeps_latin_drug_like_tokens_only():
    assert candidate_names("Tôi uống Paracetamol với ibuprofen 400mg, thuốc này sao?", 12) == ["Paracetamol", "ibuprofen"]
    assert len(candidate_names(" ".join(f"drug{i}x" for i in range(50)), 5)) == 5


async def test_mention_extractor_accepts_only_matched():
    r = FakeResolver()
    found = await DrugMentionExtractor(r).extract("Paracetamol và Ibuprofen và Zzzzzz")
    assert {m.drug_id for m in found} == {PARACETAMOL, IBUPROFEN} and "Zzzzzz" in r.calls


def test_strip_header_and_followup_query_rewrite():
    assert strip_context_header("[Tài liệu: A | Nhóm: b]\nNội dung: thân bài") == "thân bài"
    assert strip_context_header("không có header") == "không có header"
    prev = ChatMessage(role="USER", content="Metformin dùng thế nào?")
    assert retrieval_query("còn tác dụng phụ?", [prev]) == "Metformin dùng thế nào? còn tác dụng phụ?"
    assert retrieval_query("Metformin có tác dụng phụ gì đối với người lớn tuổi khi dùng lâu dài?", [prev]).startswith("Metformin có")
