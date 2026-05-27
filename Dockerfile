FROM ghcr.io/astral-sh/uv:python3.14-bookworm-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    UV_COMPILE_BYTECODE=1 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    BACKEND_PORT=8000

WORKDIR /app

# Install runtime system deps: curl for healthcheck, libgomp for OpenMP (scikit-learn, numpy)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy resolver inputs first for layer caching
COPY pyproject.toml uv.lock /app/

# Install dependencies into /app/.venv according to the lockfile
RUN uv sync --frozen --no-cache

# Ensure venv is first on PATH
ENV PATH="/app/.venv/bin:${PATH}"

# Copy application source
COPY src /app/src
COPY streamlit_frontend /app/streamlit_frontend
COPY run_all.sh /app/run_all.sh

RUN chmod +x /app/run_all.sh && mkdir -p /app/uploads /app/src/csvfiles

# Expose default dev port; Cloud Run will override via $PORT automatically.
EXPOSE 8501

# Healthcheck via Streamlit (backend kept private).
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

CMD ["/app/run_all.sh"]
