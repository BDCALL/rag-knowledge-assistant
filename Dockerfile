# syntax=docker/dockerfile:1.7

# ---- build stage: compile a wheel and install deps into a venv ----
FROM python:3.12-slim AS build
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /app
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install .

# ---- runtime stage: slim image, non-root user ----
FROM python:3.12-slim AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PATH="/opt/venv/bin:$PATH"
RUN useradd --create-home --uid 10001 app
COPY --from=build /opt/venv /opt/venv
USER app
WORKDIR /home/app
COPY --chown=app data ./data
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=3s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status == 200 else 1)"
# `rag-assistant` wraps uvicorn and installs JSON logging with request IDs.
# Try the CLI with: docker compose run --rm api rag-cli demo
CMD ["rag-assistant"]
