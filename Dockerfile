FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Dependencies first so code-only changes reuse this layer
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

# Run as an unprivileged user; the app writes uploads, indexes and logs under /app
RUN useradd --create-home --uid 1000 app && chown -R app:app /app
USER app

EXPOSE 8080

# Secrets (API keys) are injected at runtime by ECS from Secrets Manager, never baked into the image
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8080"]
