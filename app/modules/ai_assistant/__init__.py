"""Module 3 — AI Personal Health Assistant & RAG."""
from .container import AiAssistantContainer, build_container
from .errors import register_exception_handlers
from .router import create_ai_assistant_router

__all__ = ["AiAssistantContainer", "build_container", "create_ai_assistant_router", "register_exception_handlers"]
