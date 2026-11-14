# VitaP · Module 3 — AI Personal Health Assistant & RAG

Triển khai backend cho **7 endpoint** `/api/v1/ai/*` theo `M3_AI_Personal_Health_Assistant_Technical_Design.md` (v2.0), `VitaP_API_Docs.md` §9 và `database_schema_v3.md`.
FastAPI + SQLAlchemy 2 (async) + PostgreSQL/pgvector + LangChain (LCEL). **161 test, 0 lỗi** (55 test cần PostgreSQL thật).

## 1. Cấu trúc (khớp thiết kế §10.1)

```
app/modules/ai_assistant/
├── router.py                  # 7 endpoint, guard Guest→403, consent→403, rate limit→429
├── container.py               # build_container(...): điểm nối duy nhất với M1/M2
├── integration.py             # Protocol + DTO cho M1/M2 (HealthContext, CheckDrugInteractions, ResolveDrugEntity, DrugDirectory)
├── schemas/                   # chat_dto.py (API) · llm_output.py (Pydantic cho LLM)
├── services/
│   ├── chat_service.py        # orchestrator: Guard → M1/M2 → RAG → LLM → Guard → lưu DB
│   ├── explain_service.py     # POST /explain/interaction (không lưu DB)
│   ├── personal_context.py    # PersonalContextService: chọn nhóm dữ liệu, bỏ PII, chỉ thuốc ACTIVE
│   ├── interaction_support.py # gọi Rule Engine M2, đánh số bằng chứng INTERACTION_RULE
│   └── drug_mentions.py       # nhận diện thuốc trong câu hỏi qua ResolveDrugEntity (không dùng LLM)
├── repositories/              # chat_repo.py (cô lập quyền sở hữu, consent) · knowledge_repo.py (SQL hybrid + RRF)
├── rag/
│   ├── guards/                # input_guard.py (Lớp 1) · output_guard.py (Lớp 3)
│   ├── retrieval.py           # Hybrid → Rerank (3s) → ngưỡng; fallback RRF / chỉ từ khóa
│   ├── embeddings/ollama_nomic.py · reranking/bge_client.py
│   ├── generation/lcel_chain.py   # Gemini → MedGemma, một chiều
│   └── ingestion/adapters.py · chunker.py · cli.py
├── prompts/system_v2.md · explain_ddi_v1.md
migrations/m3_schema_delta.sql (+ versions/…py cho Alembic)
tests/ · examples/wiring.py · data/sample/corpus_sample.jsonl
```

## 2. Luồng xử lý một tin nhắn

```
POST /chat/sessions/{id}/messages
 → Auth (Guest 403) → consent 403 → rate-limit 429 → session thuộc user? (404)
 → LỚP 1 InputGuard (regex đã bỏ dấu, <20ms): cấp cứu / tự hại · prompt-injection · xin đổi liều/ngưng thuốc · ngoài y tế  ⇒ trả lời cố định, KHÔNG gọi LLM
 → PersonalContext (M1, chỉ PERSONAL mode, chỉ nhóm cần thiết)  +  Rule Engine M2 (nếu hỏi tương tác)
 → Hybrid search (pgvector HNSW + trigram, RRF) → bge-reranker (timeout 3s) → lọc ngưỡng
 → không có tài liệu & không có kết quả Rule Engine ⇒ INSUFFICIENT_EVIDENCE, KHÔNG gọi LLM
 → LỚP 2 Gemini (8s) ⇒ lỗi/timeout/rỗng/sai schema ⇒ MedGemma (15s) ⇒ lỗi ⇒ trả lời “AI đang bận” (HTTP 200)
 → LỚP 3 OutputGuard: kiểm citation · chặn “hoàn toàn an toàn/uống chung không sao/không có tương tác” · chặn chỉ định ngưng thuốc/chẩn đoán chắc chắn
 → lưu user+assistant message + citations trong MỘT transaction
```

## 3. Cài đặt, chạy, test

```bash
pip install -r requirements.txt
psql -f migrations/m3_schema_delta.sql            # hoặc Alembic: đặt down_revision trong migrations/versions/*.py
python -m app.modules.ai_assistant.rag.ingestion.cli \
    --input nutrition_diet_guidelines.json drug_food_interaction.json corpus.jsonl \
    --skip-existing --min-chunk-tokens 80            # thêm --dry-run để thử, --no-semantic để bỏ SemanticChunker
pytest                                             # 106 test không cần DB
M3_TEST_DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost/vitap_test pytest   # đủ 161 (cần PostgreSQL + pgvector ≥0.5, DB test sẽ bị DROP SCHEMA)
```

Nối vào app: xem `examples/wiring.py`. Cấu hình qua biến môi trường (`.env.example`).

## 3b. Dữ liệu nguồn JSON của nhóm → kho tri thức

CLI nhận nhiều file `.json` (mảng hoặc một object) và `.jsonl`; **tự nhận diện định dạng theo từng bản ghi** (`--format` để ép). Chunker lõi không đổi; việc chuyển định dạng nằm ở `rag/ingestion/adapters.py`.

| File | Nhận diện | Cách chuyển | Kết quả chunk |
|---|---|---|---|
| `nutrition_diet_guidelines.json` (MedlinePlus) | có `topic_id` + `summary` | dòng đầu đoạn kết thúc bằng `?` → `## …`; bullet liền nhau gom thành danh sách; dòng cuối `NIH: …` tách vào `license_info`; `also_called`→aliases; `category`+`groups`→categories; `url`→`source_url` | 1 chunk/mục câu hỏi (vd. bài *Alcohol* → 7 chunk, 6 nếu `--min-chunk-tokens 80`) |
| `drug_food_interaction.json` | có `inn_name` + `fda_evidence` | 1 dòng = 1 tài liệu = **1 chunk** (`chunk_strategy="single"`), nội dung song ngữ: khuyến nghị tiếng Việt + trích nhãn FDA; `severity`, `food_category`, `fda_section`, `ingredient_id` gốc lưu trong `metadata` | `external_id = dfi-<id>`, `document_type = DRUG_FOOD_INTERACTION` |
| `*.jsonl` Golden Format | có `external_id` + `content` | không đổi | như cũ |

Thay đổi đi kèm: (1) `HybridChunker(min_chunk_tokens=…)` gộp các mục quá ngắn (mặc định tắt); (2) chunk mang `metadata`, và **chunk `severity = RED` được trích dẫn sẽ nâng `warning_level` tối thiểu lên WARNING** dù LLM chọn INFO; (3) trích đoạn `excerpt` của citation nay bắt đầu từ câu khớp câu hỏi nhất thay vì 300 ký tự đầu chunk (tránh trường hợp đoạn hiển thị không chứa câu chứng minh).

**Vấn đề chất lượng dữ liệu cần nhóm xử lý trước khi nạp thật**
- **`drug_food_interaction.json` id 1 bị lệch thuốc**: ingredient là *Dextromethorphan* nhưng `fda_evidence` nói về *acetaminophen* (tổn thương gan khi uống ≥3 ly rượu mỗi ngày). Nhiều khả năng là nhãn thuốc phối hợp bị gán cho một hoạt chất. Chatbot sẽ trích dẫn rằng dextromethorphan gây hại gan, điều này có thể sai. Nên rà soát toàn bộ các dòng có `fda_evidence` nhắc tên hoạt chất khác. Lời khuyên `clinical_advice_vn` cũng có vẻ sinh theo mẫu (cùng một câu "gan, hạ đường huyết hoặc ức chế TKTW" cho mọi dòng RED/ALCOHOL), không bám sát bằng chứng.
- **`ingredient_id` là số nguyên (375)**, còn `ingredients.id` trong schema v3 là UUID: không join được. Hiện chỉ lưu vào `metadata.source_ingredient_id`. Nếu M2 muốn dùng dữ liệu này tất định, cần ánh xạ qua `inn_name` ↔ `ingredients.normalized_name`.
- **Không có `source_url`/mã nhãn** trong file thuốc–thực phẩm ⇒ citation không có link; `source_name` đang giả định `"FDA drug labeling"` (hằng `DRUG_FOOD_SOURCE`). Nên bổ sung URL DailyMed/openFDA của nhãn.
- Chỉ thấy `severity` RED/GREEN trong mẫu; giá trị khác được giữ nguyên chữ và **không** kích hoạt nâng cảnh báo.
- Schema v3 chưa có bảng thuốc–thực phẩm. Dùng qua RAG thì chạy được ngay; nếu muốn cảnh báo tất định kiểu "thuốc đang dùng + rượu" thì nên có bảng riêng do M2 truy vấn. Không có dòng nào cho một thuốc không có nghĩa là an toàn; Output Guard vẫn chặn các câu khẳng định đó.

## 4. Hợp đồng cần các thành viên khác đáp ứng

| Bên | Việc | Ghi chú |
|---|---|---|
| Trọng (M1/M2/Auth) | Adapter 4 Protocol trong `integration.py` | Giả định `async`; nếu service là sync, bọc `anyio.to_thread.run_sync`. **`DrugDirectory.get_drug_summaries` là interface bổ sung** (M3 cần tên thuốc để diễn giải; §10.2 chưa liệt kê). |
| Trọng (Auth) | `get_principal` trả object có `user_id`, `is_guest` | Token thiếu/sai → 401; Guest hợp lệ vẫn trả principal (M3 tự trả 403 LOGIN_REQUIRED). |
| Phong (Reranker) | `POST {URL}/rerank` `{"query","documents":[…]}` → `{"scores":[0..1,…]}` cùng thứ tự | Phải đã chuẩn hóa (sigmoid). Gửi header `ngrok-skip-browser-warning` đã được xử lý phía client. |
| Phong (benchmark) | Chốt `MIN_RERANK_SCORE` (mặc định **0.35, chưa benchmark**) và 2 ngưỡng fallback (`FALLBACK_MIN_VECTOR_SIMILARITY=0.50`, `…LEXICAL_SCORE=0.30`) | Fallback giữ lại một sàn tin cậy khi reranker chết, thay vì đưa thẳng Top-5 RRF. |
| Database | Chạy `m3_schema_delta.sql` | Thêm `message_citations.citation_index`, `knowledge_documents.external_id` (+ unique), chỉ mục HNSW/trigram. |

## 5. Quyết định thiết kế & điểm lệch giữa các tài liệu (cần team xác nhận)

1. **`warning_level`**: API trả `INFO/WARNING/CRITICAL` (theo §9), schema v3 lưu `NORMAL/CAUTION/WARNING/CRITICAL`. Code ánh xạ hai chiều (`enums.warning_to_db/from_db`): INFO↔NORMAL, CAUTION→WARNING.
2. **Tên cột DB ≠ tên trường API**: `context_used`→`personal_context_used`, `model_name`→`model_used`, `consent_type` (thiết kế ghi `scope`).
3. **Trạng thái tương tác**: dùng 4 trạng thái của contract M2 (`INTERACTION_FOUND / UNRESOLVED_DRUG / INSUFFICIENT_DATA / NO_KNOWN_INTERACTION_IN_DATABASE`), không dùng bộ tên trong thiết kế §6.2. Khối prompt là `<drug_interaction_result>` (thiết kế còn một chỗ ghi `<interaction_engine_result>`).
4. **Mở rộng so với thiết kế**: (a) yêu cầu *tự ý đổi liều/ngưng thuốc* được chặn **tất định ở Lớp 1** (NEEDS_CLINICIAN, không gọi LLM) thay vì chờ LLM; (b) câu hỏi “bị bệnh gì” gắn gợi ý để Output Guard ép `NEEDS_CLINICIAN`; (c) khi Rule Engine M2 lỗi, chat vẫn chạy nhưng Output Guard cấm mọi kết luận về tương tác, còn `/explain/interaction` trả 503 (không trả âm tính giả).
5. **`/explain/interaction`**: LLM hỏng hoặc bị Guard từ chối ⇒ vẫn 200 với giải thích mẫu tất định, `model_used="rule-template"`. `include_personal_context` mặc định **false**. Hai thuốc trùng nhau ⇒ 422.
6. **Consent áp cho cả 7 endpoint** (theo §9), kể cả `DELETE`: người dùng đã rút consent sẽ không tự xóa được lịch sử qua API này — nên cân nhắc cho phép riêng `DELETE`.
7. **Personal context**: không bao giờ gọi M1 ở GENERAL; câu hỏi kiến thức thuần (không ngôi thứ nhất, không nhắc thuốc/chỉ số) cũng không lấy gì; nhắc thuốc/tương tác ⇒ luôn kèm dị ứng + thuốc ACTIVE. Bệnh nền chỉ lấy ACTIVE/CHRONIC; dữ liệu gắn nhãn “người dùng tự khai báo”. M1 lỗi ⇒ vẫn trả lời nhưng kèm ghi chú “chưa được cá nhân hóa”.
8. **Nhận diện thuốc trong câu hỏi** dựa vào `ResolveDrugEntity` trên các từ ASCII ≥4 ký tự (tối đa 12/câu); chỉ chấp nhận `MATCHED`. Không dùng LLM đoán tên thuốc. Hạn chế: tên thuốc viết có dấu/phiên âm tiếng Việt sẽ không được nhận ra.
9. **Chunk lưu kèm header ngữ cảnh** (`[Tài liệu: … | Tên khác: … | Nhóm: …]\nNội dung: …`) để cả vector lẫn trigram đều “biết” chủ đề; excerpt trích dẫn đã cắt header. `estimate_tokens` ≈ ký tự/4 (không dùng tokenizer thật).
10. **Câu hỏi nối tiếp ngắn** (≤6 từ) được ghép với câu hỏi trước để truy xuất đúng chủ đề.
11. Citation `PERSONAL_PROFILE` **chưa dùng**: nhóm dữ liệu cá nhân được báo qua `personal_context_used`.

## 6. Hạn chế đã biết

- **Rate limiter in-memory** (1 tiến trình). Chạy nhiều worker cần cài `RateLimiter` dùng Redis (Protocol có sẵn trong `rate_limit.py`).
- **`langchain-experimental` đã ngừng bảo trì** (SemanticChunker). Có thể chạy ingestion với `--no-semantic` (vẫn có header-split + hard cap 500 token).
- Nhánh trigram so khớp câu hỏi tiếng Việt với tài liệu tiếng Anh gần như chỉ hữu ích cho tên thuốc/mã (HbA1c…); phần liên ngôn ngữ dựa vào nomic-embed-text-v2 — **chất lượng retrieval thật chưa được đo** (test dùng embedding băm từ giả).
- Test không gọi Gemini/MedGemma/Ollama/Ngrok thật; LLM, embedding, reranker, M1, M2 đều là đồ giả có kịch bản. Cần một lượt smoke test với dịch vụ thật trước demo.
- Regex Lớp 1 là heuristic: ưu tiên không bỏ sót cấp cứu nên có thể báo nhầm (vd. phủ định “tôi không bị đau ngực dữ dội”). Bộ 30 câu trong thiết kế §11.1 đã được mã hóa thành test (`tests/test_input_guard.py`), cần Minh Trí bổ sung ca thật.
