"""Mã lỗi M3 + envelope lỗi chung (VitaP_API_Docs.md §2.4, §9.4)."""
from __future__ import annotations

import logging
import uuid
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)


class AppError(Exception):
    def __init__(
        self,
        status_code: int,
        code: str,
        message: str,
        details: list[dict[str, Any]] | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message
        self.details = details or []
        self.headers = headers or {}


def auth_required() -> AppError:
    return AppError(401, "AUTH_REQUIRED", "Cần đăng nhập để tiếp tục")


def login_required() -> AppError:
    return AppError(403, "LOGIN_REQUIRED", "Trợ lý AI chỉ dành cho người dùng đã đăng nhập")


def consent_required() -> AppError:
    return AppError(
        403, "CONSENT_REQUIRED", "Bạn cần đồng ý xử lý dữ liệu sức khỏe và AI để sử dụng Trợ lý AI"
    )


def not_found(what: str = "Tài nguyên") -> AppError:
    return AppError(404, "RESOURCE_NOT_FOUND", f"{what} không tồn tại")


def invalid_message_role() -> AppError:
    return AppError(409, "INVALID_MESSAGE_ROLE", "Chỉ có thể đánh giá tin nhắn của Trợ lý AI")


def rate_limited(retry_after: int) -> AppError:
    return AppError(
        429,
        "RATE_LIMITED",
        "Bạn gửi quá nhiều yêu cầu, vui lòng thử lại sau ít phút",
        headers={"Retry-After": str(max(1, retry_after))},
    )


def dependency_unavailable(message: str = "Dịch vụ tạm thời không khả dụng") -> AppError:
    return AppError(503, "DEPENDENCY_UNAVAILABLE", message)


def _envelope(request: Request, code: str, message: str, details: list[dict[str, Any]] | None = None):
    request_id = request.headers.get("x-request-id") or f"req-{uuid.uuid4().hex[:12]}"
    return {"error": {"code": code, "message": message, "details": details or [], "request_id": request_id}}


def register_exception_handlers(app: FastAPI) -> None:
    """Gắn handler tạo envelope lỗi chuẩn. Bỏ qua nếu repo đã có handler tương đương."""

    @app.exception_handler(AppError)
    async def _app_error(request: Request, exc: AppError):  # noqa: ANN202
        return JSONResponse(
            status_code=exc.status_code,
            content=_envelope(request, exc.code, exc.message, exc.details),
            headers=exc.headers or None,
        )

    @app.exception_handler(RequestValidationError)
    async def _validation(request: Request, exc: RequestValidationError):  # noqa: ANN202
        details = [
            {"field": ".".join(str(p) for p in err.get("loc", ())[1:]) or "body", "reason": err.get("type", "invalid")}
            for err in exc.errors()
        ]
        return JSONResponse(
            status_code=422,
            content=_envelope(request, "VALIDATION_ERROR", "Dữ liệu không hợp lệ", details),
        )

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception):  # noqa: ANN202
        logger.exception("Unhandled error in M3")
        return JSONResponse(
            status_code=500, content=_envelope(request, "INTERNAL_ERROR", "Đã xảy ra lỗi hệ thống")
        )
