"""Base ORM riêng của M3.

Nếu repo đã có Base dùng chung (vd. app.database.base.Base), chỉ cần đổi dòng import ở `models.py`
sang Base đó để Alembic autogenerate nhìn thấy các bảng M3.
"""
from __future__ import annotations

from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass
