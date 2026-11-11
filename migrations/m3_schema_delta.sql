-- Module 3 — phần bổ sung lên schema v3. Idempotent (chạy lại an toàn).
-- M3-DB-01..04 đã có trong schema v3; các IF NOT EXISTS bên dưới chỉ để DB cũ (v2) cũng chạy được.

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- M3-DB-05 (ĐỀ XUẤT BỔ SUNG): số thứ tự [n] của citation, để khôi phục đúng thứ tự trong lịch sử chat.
ALTER TABLE message_citations ADD COLUMN IF NOT EXISTS citation_index smallint NOT NULL DEFAULT 0;

-- M3-DB-06 (ĐỀ XUẤT BỔ SUNG): khóa nghiệp vụ của tài liệu để nạp lại/skip-existing an toàn (Golden Format: external_id).
ALTER TABLE knowledge_documents ADD COLUMN IF NOT EXISTS external_id varchar(100);
CREATE UNIQUE INDEX IF NOT EXISTS uq_knowledge_documents_source_external
    ON knowledge_documents (source_name, external_id) WHERE external_id IS NOT NULL;

-- Chỉ mục truy xuất (thiết kế §3.2)
CREATE INDEX IF NOT EXISTS idx_knowledge_chunks_vector_hnsw
    ON knowledge_chunks USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);
CREATE INDEX IF NOT EXISTS idx_knowledge_chunks_content_trgm
    ON knowledge_chunks USING gin (content gin_trgm_ops);

-- Chỉ mục truy vấn chat
CREATE INDEX IF NOT EXISTS ix_chat_sessions_user_updated ON chat_sessions (user_id, updated_at);
CREATE INDEX IF NOT EXISTS ix_chat_messages_session_created ON chat_messages (session_id, created_at);
CREATE INDEX IF NOT EXISTS ix_message_citations_message ON message_citations (message_id);
