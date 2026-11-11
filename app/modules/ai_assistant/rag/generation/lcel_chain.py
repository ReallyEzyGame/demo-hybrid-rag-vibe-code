"""Tầng sinh câu trả lời — LangChain LCEL, Gemini (primary) → MedGemma (fallback) MỘT CHIỀU (thiết kế §7)."""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Generic, Literal, TypeVar

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel

from ...config import AiAssistantSettings
from ...prompts import load_prompt
from ...schemas.llm_output import LLMExplainAnswer, LLMGeneratedAnswer

logger = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)

CHAT_HUMAN_TEMPLATE = """<conversation_history>
{history}
</conversation_history>

<personal_health_context>
{personal_context}
</personal_health_context>

<drug_interaction_result>
{interaction_result}
</drug_interaction_result>

<retrieved_evidence>
{evidence}
</retrieved_evidence>

<request_hints>
{hints}
</request_hints>

<user_question>
{question}
</user_question>

{format_instructions}"""

EXPLAIN_HUMAN_TEMPLATE = """<personal_health_context>
{personal_context}
</personal_health_context>

<drug_interaction_result>
{interaction_result}
</drug_interaction_result>

<retrieved_evidence>
{evidence}
</retrieved_evidence>

{format_instructions}"""


class GenerationUnavailable(Exception):
    """Cả Gemini và MedGemma đều không trả được kết quả hợp lệ → graceful abstention."""


@dataclass
class GenerationResult(Generic[T]):
    parsed: T
    model_used: str


@dataclass
class _Route:
    name: str
    llm: BaseChatModel
    timeout_s: float


class LcelGenerationEngine:
    """Mỗi route là chain `prompt | llm | StrOutputParser`, parse Pydantic sau cùng để biết chính xác model nào đã trả lời."""

    def __init__(self, primary: _Route, fallback: _Route | None) -> None:
        self.primary, self.fallback = primary, fallback
        self._parsers: dict[str, PydanticOutputParser] = {
            "chat": PydanticOutputParser(pydantic_object=LLMGeneratedAnswer),
            "explain": PydanticOutputParser(pydantic_object=LLMExplainAnswer),
        }
        self._prompts = {
            "chat": ChatPromptTemplate.from_messages([("system", load_prompt("system_v2.md")), ("human", CHAT_HUMAN_TEMPLATE)]),
            "explain": ChatPromptTemplate.from_messages(
                [("system", load_prompt("explain_ddi_v1.md")), ("human", EXPLAIN_HUMAN_TEMPLATE)]
            ),
        }

    @classmethod
    def from_models(
        cls,
        primary_llm: BaseChatModel,
        fallback_llm: BaseChatModel | None,
        settings: AiAssistantSettings,
    ) -> "LcelGenerationEngine":
        return cls(
            _Route(settings.primary_llm_model, primary_llm, settings.primary_llm_timeout_s),
            _Route(settings.fallback_llm_model, fallback_llm, settings.fallback_llm_timeout_s) if fallback_llm else None,
        )

    async def generate(self, kind: Literal["chat", "explain"], variables: dict[str, Any]) -> GenerationResult[Any]:
        parser = self._parsers[kind]
        inputs = {**variables, "format_instructions": parser.get_format_instructions()}
        # Strict one-way: primary → fallback → raise. Không bao giờ quay lại primary.
        for route in (self.primary, self.fallback):
            if route is None:
                continue
            try:
                chain = self._prompts[kind] | route.llm | StrOutputParser()
                raw = await asyncio.wait_for(chain.ainvoke(inputs), timeout=route.timeout_s)
                if not raw or not str(raw).strip():
                    raise ValueError("empty LLM response")
                return GenerationResult(parser.parse(str(raw)), route.name)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - 429/5xx/timeout/rỗng/sai schema đều chuyển tuyến
                logger.warning("LLM route %s failed: %s", route.name, type(exc).__name__)
        raise GenerationUnavailable("all LLM routes failed")


def build_default_engine(settings: AiAssistantSettings) -> LcelGenerationEngine:
    """Khởi tạo Gemini 3.1 Flash-Lite + MedGemma (Ollama qua Ngrok). Import trễ để test không cần SDK/khóa API."""
    from langchain_google_genai import ChatGoogleGenerativeAI

    key = settings.gemini_api_key.get_secret_value() if settings.gemini_api_key else None
    primary = ChatGoogleGenerativeAI(
        model=settings.primary_llm_model,
        temperature=settings.llm_temperature,
        max_retries=1,
        timeout=settings.primary_llm_timeout_s,
        google_api_key=key,
    )
    fallback = None
    if settings.kaggle_medgemma_ngrok_url:
        from langchain_ollama import ChatOllama

        fallback = ChatOllama(
            base_url=settings.kaggle_medgemma_ngrok_url,
            model=settings.fallback_llm_model,
            temperature=settings.llm_temperature,
            client_kwargs={"timeout": settings.fallback_llm_timeout_s, "headers": {"ngrok-skip-browser-warning": "true"}},
        )
    return LcelGenerationEngine.from_models(primary, fallback, settings)
