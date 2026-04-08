# ─────────────────────────────────────────────────────────
#  ContentGuard — Content Moderation OpenEnv — Dockerfile
#  Runs the FastAPI server via uvicorn on port 7860.
#  HuggingFace Spaces Docker SDK uses port 7860 by default.
# ─────────────────────────────────────────────────────────

FROM python:3.10-slim

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Working directory
WORKDIR /app

# Install Python dependencies first (layer cache optimisation)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Expose port (HuggingFace Spaces default)
EXPOSE 7860

# Health check — validators ping /reset, but /health is lighter
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# Start the FastAPI server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
