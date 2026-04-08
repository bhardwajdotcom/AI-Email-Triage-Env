# ─── AI Email Triage Environment — Hugging Face Spaces Dockerfile ────────────
# Based on Python 3.11 slim for a lean image.
# HuggingFace Spaces exposes port 7860 by default.

FROM python:3.11-slim

# Metadata
LABEL maintainer="AI Email Triage Team"
LABEL description="OpenEnv-compatible AI Email Triage & Response Environment"
LABEL version="1.0.0"

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY models.py .
COPY rewards.py .
COPY tasks.py .
COPY environment.py .
COPY server.py .
COPY inference.py .
COPY openenv.yaml .
COPY data/ ./data/

# Create non-root user (HuggingFace Spaces runs as user 1000)
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose port (HuggingFace Spaces standard)
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# Start the FastAPI server
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
