# IT2 AI — FastAPI Semantic Search Service (ChromaDB)

Microservice Python cung cấp API tìm kiếm ngữ nghĩa (Semantic Search) trên kho **10,564 chunks** văn bản pháp quy DUT ([rag_chunks.jsonl](../dut_rag_dataset_processed/rag_chunks.jsonl)) bằng **FastAPI + ChromaDB**.

---

## 1. Cài đặt môi trường Local (Windows PowerShell)

```powershell
cd IT2_AI
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

---

## 2. Nạp dữ liệu vào ChromaDB

### Chạy nhanh cho Sprint Demo (nhóm lĩnh vực sinh viên cốt lõi, ~2–4 phút trên CPU):
```powershell
python scripts/ingest.py --fields "Công tác sinh viên;Đào tạo;Khảo thí;Học liệu, truyền thông"
```

### Chạy nạp toàn bộ 10,564 chunks:
```powershell
python scripts/ingest.py
```
> *Mẹo:* Nếu chạy trên máy không có GPU với `BAAI/bge-m3`, bạn có thể chạy `scripts/ingest.py` trên Kaggle GPU T4 rồi copy thư mục `dut_rag_chroma_db/` về local. Nếu đổi `EMBEDDING_MODEL`, thêm cờ `--reset`.

---

## 3. Khởi chạy FastAPI Server

```powershell
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

- **Swagger UI:** `http://localhost:8000/docs`
- **Health Check:** `GET http://localhost:8000/health`
- **Vector Search:** `POST http://localhost:8000/api/vector/search` (có alias `POST http://localhost:8000/api/search` tương thích trực tiếp với `IT2_BE`)

---

## 4. API Contract (Dành cho Phúc & Huy tích hợp)

### Request (`POST /api/vector/search`)
```json
{
  "query": "Điều kiện xét học bổng khuyến khích học tập",
  "top_k": 5
}
```

### Optional filters
```json
{
  "query": "Cảnh báo học tập",
  "top_k": 5,
  "min_score": 0.3,
  "filters": {
    "field": "Đào tạo",
    "document_number": "2244/QĐ-ĐHBK"
  }
}
```

### Response (`200 OK`)
```json
{
  "query": "Điều kiện xét học bổng khuyến khích học tập",
  "top_k": 5,
  "total_found": 5,
  "embedding_model": "bkai-foundation-models/vietnamese-bi-encoder",
  "results": [
    {
      "chunk_id": "0509_điều_4_1",
      "score": 0.7842,
      "distance": 0.2158,
      "content": "1. Sinh viên có kết quả học tập...",
      "full_rag_text": "[VĂN BẢN: ...]\n\n[NỘI DUNG]:\n1. Sinh viên...",
      "metadata": {
        "document_id": "0509",
        "document_number": "3121/QĐ-BGDĐT",
        "document_title": "Quyết định Ban hành Quy chế...",
        "field": "Công tác sinh viên",
        "issuer": "Bộ Giáo dục và Đào tạo",
        "status": "Còn hiệu lực",
        "chapter_title": "Chương II: ...",
        "article_number": "Điều 4",
        "article_title": "Đối tượng và tiêu chuẩn xét cấp học bổng",
        "clause_number": "1",
        "token_count_estimate": 165
      }
    }
  ]
}
```

---

## 5. Chạy Unit Tests

```powershell
pytest -v
```
