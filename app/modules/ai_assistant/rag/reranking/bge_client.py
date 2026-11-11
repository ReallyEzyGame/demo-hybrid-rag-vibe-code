"""HTTP client gọi bge-reranker-v2-m3 chạy trên Kaggle/Colab qua Ngrok (thiết kế §5.2), timeout cứng 3s.

Contract với notebook của Phong:
    POST {RERANKER_URL}/rerank
    {"query": "<str>", "documents": ["<str>", ...]}
 -> {"scores": [<float 0..1>, ...]}   # cùng thứ tự `documents`, đã chuẩn hóa (sigmoid)
"""
from __future__ import annotations

import logging
from typing import Protocol, Sequence

import httpx

logger = logging.getLogger(__name__)


class RerankerUnavailable(Exception):
    """Reranker timeout/lỗi mạng/phản hồi sai → caller phải rơi về kết quả RRF."""


class RerankerPort(Protocol):
    async def score(self, query: str, documents: Sequence[str]) -> list[float]: ...


class BgeRerankerClient:
    def __init__(self, base_url: str, timeout_s: float = 3.0, client: httpx.AsyncClient | None = None) -> None:
        self._url = base_url.rstrip("/") + "/rerank"
        self._timeout = timeout_s
        self._client = client
        self._headers = {"ngrok-skip-browser-warning": "true"}  # bỏ trang cảnh báo của ngrok free tier

    async def score(self, query: str, documents: Sequence[str]) -> list[float]:
        if not documents:
            return []
        payload = {"query": query, "documents": list(documents)}
        try:
            if self._client is not None:
                resp = await self._client.post(self._url, json=payload, headers=self._headers, timeout=self._timeout)
            else:
                async with httpx.AsyncClient(timeout=self._timeout) as c:
                    resp = await c.post(self._url, json=payload, headers=self._headers)
            resp.raise_for_status()
            scores = resp.json()["scores"]
        except (httpx.HTTPError, KeyError, ValueError, TypeError) as exc:  # gồm cả TimeoutException
            logger.warning("Reranker unavailable: %s", type(exc).__name__)
            raise RerankerUnavailable(str(exc)) from exc
        if len(scores) != len(documents):
            raise RerankerUnavailable("score count mismatch")
        try:
            return [min(1.0, max(0.0, float(s))) for s in scores]
        except (TypeError, ValueError) as exc:
            raise RerankerUnavailable("non-numeric score") from exc
