FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install uv==0.13.0

ENV UV_LINK_MODE=copy \
    UV_CONCURRENT_DOWNLOADS=8

RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,from=wheels,target=/wheels \
    if python -c "from pathlib import Path; import hashlib, sys; marker = Path('/wheels/requirements.sha256'); sys.exit(not (marker.is_file() and marker.read_text().strip() == hashlib.sha256(Path('requirements.txt').read_bytes()).hexdigest()))"; then \
        uv pip install --system --no-index --find-links=/wheels -r requirements.txt \
        || uv pip install --system --find-links=/wheels -r requirements.txt; \
    else \
        uv pip install --system --find-links=/wheels -r requirements.txt; \
    fi

COPY . .

EXPOSE 8001

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]
