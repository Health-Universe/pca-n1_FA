# Base Build Image
FROM --platform=linux/amd64 python:3.10-slim-bookworm as build
WORKDIR /app

ENV LANG=C.UTF-8 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

COPY install-packages.sh .
RUN chmod +x ./install-packages.sh && \
    ./install-packages.sh && \
    uv venv && \
    . .venv/bin/activate && uv pip install setuptools wheel

COPY . /app
RUN uv pip install --no-cache -r /app/requirements.txt

# Runtime Image
FROM --platform=linux/amd64 python:3.10-slim-bookworm as final
WORKDIR /app

ENV LANG=C.UTF-8 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PATH='/app/.venv/bin:$PATH'

COPY --from=build --chown=app:app /app /app

EXPOSE 8501

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8501"]
