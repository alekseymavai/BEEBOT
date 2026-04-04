# ---- Frontend build stage ----
FROM node:22-alpine AS frontend-build

WORKDIR /app/web

COPY web/package.json ./
RUN npm install

COPY web/ ./
RUN npm run build

# ---- Unified backend + frontend stage ----
FROM python:3.12-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Limit thread memory (critical for 2 GB VPS)
ENV OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false

# Python dependencies (full — бот + веб в одном образе)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download embedding model at build time (fastembed caches ONNX model)
RUN python -c "from fastembed import TextEmbedding; TextEmbedding(model_name='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')"

# Copy application code
COPY src/ src/
COPY data/processed/ data/processed/
COPY data/texts/ data/texts/
COPY data/pdfs/ data/pdfs/

# Copy built Vue frontend
COPY --from=frontend-build /app/web/dist /app/web/dist

# Unified: бот (polling) + веб-панель (uvicorn :8088) в одном процессе
EXPOSE 8088
CMD ["python", "-m", "src.bot"]
