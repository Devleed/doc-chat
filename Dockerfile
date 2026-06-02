FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY pyproject.toml uv.lock ./

RUN pip install --no-cache-dir uv \
  && uv sync --no-dev

ENV PATH="/app/.venv/bin:${PATH}"

COPY rag ./rag

EXPOSE 8000

CMD ["uvicorn", "rag.server:app", "--host", "0.0.0.0", "--port", "8000"]
