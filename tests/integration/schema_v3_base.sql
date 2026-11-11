-- Tái hiện các bảng liên quan M3 đúng như schema v3 (FK mặc định NO ACTION, KHÔNG cascade) + stub bảng của module khác.
CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

CREATE TABLE users (id uuid PRIMARY KEY DEFAULT gen_random_uuid());
CREATE TABLE user_consents (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id uuid NOT NULL REFERENCES users(id),
  consent_type varchar(100) NOT NULL,
  is_granted boolean DEFAULT true,
  granted_at timestamptz DEFAULT now(),
  revoked_at timestamptz
);
CREATE TABLE knowledge_documents (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  title varchar(500) NOT NULL, document_type varchar(100), source_name varchar(255) NOT NULL, source_url text,
  version varchar(100), published_at date, effective_from date, effective_until date,
  status varchar(30) DEFAULT 'ACTIVE', license_info text, created_at timestamptz DEFAULT now()
);
CREATE TABLE knowledge_chunks (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  document_id uuid NOT NULL REFERENCES knowledge_documents(id),
  chunk_index int NOT NULL, content text NOT NULL, embedding vector(768), metadata jsonb,
  created_at timestamptz DEFAULT now(), UNIQUE (document_id, chunk_index)
);
CREATE TABLE chat_sessions (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id uuid NOT NULL REFERENCES users(id), title varchar(255), context_mode varchar(30) DEFAULT 'PERSONAL',
  created_at timestamptz DEFAULT now(), updated_at timestamptz
);
CREATE TABLE chat_messages (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  session_id uuid NOT NULL REFERENCES chat_sessions(id), role varchar(20) NOT NULL, content text NOT NULL,
  answer_status varchar(40), warning_level varchar(20) DEFAULT 'NORMAL', context_used jsonb, suggested_followups jsonb,
  model_name varchar(100), created_at timestamptz DEFAULT now()
);
CREATE TABLE message_citations (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  message_id uuid NOT NULL REFERENCES chat_messages(id),
  document_id uuid REFERENCES knowledge_documents(id), chunk_id uuid REFERENCES knowledge_chunks(id), drug_id uuid,
  citation_type varchar(30) NOT NULL, source_title varchar(255), excerpt text
);
CREATE TABLE ai_feedback (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  message_id uuid NOT NULL REFERENCES chat_messages(id), user_id uuid NOT NULL REFERENCES users(id),
  rating varchar(20) NOT NULL, comment text, created_at timestamptz DEFAULT now(), updated_at timestamptz,
  CONSTRAINT uq_ai_feedback_message_user UNIQUE (message_id, user_id)
);
