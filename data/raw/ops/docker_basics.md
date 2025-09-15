# Docker basics for this app

**Images** are templates; **containers** are running instances. Use **volumes** so your index/logs persist.

## Image strategy
- Base: `python:3.11-slim` to keep images small.
- Install only what you need (pin versions).
- Use a **multi-stage build** if you compile extras, then copy artifacts into a slim runtime.

### Example `Dockerfile`
```dockerfile
FROM python:3.11-slim AS base
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

### System deps for PDFs etc. (add as needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential poppler-utils && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml poetry.lock* ./  # or requirements.txt
RUN pip install --no-cache-dir -U pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
