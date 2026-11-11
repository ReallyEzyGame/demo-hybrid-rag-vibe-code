"""M3 schema delta: citation_index, knowledge_documents.external_id, chỉ mục truy xuất.

Đặt `down_revision` thành revision hiện tại của repo trước khi chạy `alembic upgrade head`.
"""
from pathlib import Path

from alembic import op

revision = "20261111_m3_delta"
down_revision = None  # TODO(team): revision liền trước trong repo chính
branch_labels = None
depends_on = None

_SQL = Path(__file__).resolve().parents[1] / "m3_schema_delta.sql"


def upgrade() -> None:
    op.execute(_SQL.read_text(encoding="utf-8"))


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_knowledge_chunks_content_trgm")
    op.execute("DROP INDEX IF EXISTS uq_knowledge_documents_source_external")
    op.execute("ALTER TABLE knowledge_documents DROP COLUMN IF EXISTS external_id")
    op.execute("ALTER TABLE message_citations DROP COLUMN IF EXISTS citation_index")
