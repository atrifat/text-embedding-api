FROM python:3.13-slim

# Install basic tools: wget, curl, unzip
RUN apt-get update && \
    apt-get install -y wget curl unzip && \
    rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

COPY --chown=user ./requirements.txt requirements.txt
RUN --mount=type=cache,target=/home/user/.cache/pip pip install --upgrade -r requirements.txt

COPY --chown=user . /app

ENV APP_PORT=7860
ENV APP_HOST="0.0.0.0"
ENV ENVIRONMENT="production"
ENV TOKENIZERS_PARALLELISM=false

ENV DEFAULT_MODEL="text-embedding-3-large"
ENV WARMUP_ENABLED=true
ENV CUDA_CACHE_CLEAR_ENABLED=true
ENV EMBEDDING_BATCH_SIZE=8
ENV EMBEDDINGS_CACHE_ENABLED=true
ENV EMBEDDINGS_CACHE_MAXSIZE=2048
ENV REPORT_CACHED_TOKENS=false

CMD ["uvicorn", "app:app", "--host", "${APP_HOST}", "--port", "${APP_PORT}"]
# CMD ["python", "app.py"]
