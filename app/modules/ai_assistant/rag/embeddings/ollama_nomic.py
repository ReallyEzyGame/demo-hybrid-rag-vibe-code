"""Adapter OllamaEmbeddings cho nomic-embed-text-v2 (768d) — thiết kế §4.2. Bắt buộc tiền tố search_document/search_query."""
from __future__ import annotations

from typing import Protocol, Sequence

DOC_PREFIX = "search_document: "
QUERY_PREFIX = "search_query: "


class EmbeddingsPort(Protocol):
    async def aembed_query(self, query: str) -> list[float]: ...

    async def aembed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_query(self, query: str) -> list[float]: ...


class NomicOllamaEmbeddingAdapter:
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "nomic-embed-text-v2", client=None) -> None:  # noqa: ANN001
        if client is None:
            from langchain_ollama import OllamaEmbeddings  # import trễ: test không cần cài Ollama

            client = OllamaEmbeddings(base_url=base_url, model=model)
        self.client = client

    # --- sync (dùng cho SemanticChunker / CLI)
    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return self.client.embed_documents([f"{DOC_PREFIX}{t}" for t in texts])

    def embed_query(self, query: str) -> list[float]:
        return self.client.embed_query(f"{QUERY_PREFIX}{query}")

    # --- async (dùng trong request)
    async def aembed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return await self.client.aembed_documents([f"{DOC_PREFIX}{t}" for t in texts])

    async def aembed_query(self, query: str) -> list[float]:
        return await self.client.aembed_query(f"{QUERY_PREFIX}{query}")
