from __future__ import annotations

import re
import unicodedata


def strip_accents(text: str) -> str:
    decomposed = unicodedata.normalize("NFD", text)
    out = "".join(c for c in decomposed if unicodedata.category(c) != "Mn")
    return out.replace("đ", "d").replace("Đ", "D")


def fold(text: str) -> str:
    """Chữ thường, bỏ dấu tiếng Việt, gộp khoảng trắng — dùng để khớp cả câu có dấu lẫn không dấu."""
    return re.sub(r"\s+", " ", strip_accents(text).lower()).strip()


def lower_nfc(text: str) -> str:
    """Chữ thường, giữ dấu — cho các mẫu dễ nhầm khi bỏ dấu (vd. 'từ từ' vs 'tự tử')."""
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", text).lower()).strip()
