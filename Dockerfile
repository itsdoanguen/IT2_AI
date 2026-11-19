# syntax=docker/dockerfile:1
FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/app/.cache/huggingface

COPY requirements.txt .
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu torch -r requirements.txt

# Pre-bake embedding model weights into the image so container cold-starts stay fast.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('bkai-foundation-models/vietnamese-bi-encoder')"

COPY app ./app
COPY scripts ./scripts
COPY data ./data
RUN if [ -f data/dut_rag_chroma_db.tar.gz ] && [ ! -f dut_rag_chroma_db/chroma.sqlite3 ]; then \
      tar -xzf data/dut_rag_chroma_db.tar.gz -C /app && rm -f data/dut_rag_chroma_db.tar.gz; \
    fi

EXPOSE 8000

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
