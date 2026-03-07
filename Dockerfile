# Video Translator Docker Image
# Multi-stage build for optimized production image

# ==================== Build Stage ====================
FROM python:3.11-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ==================== Runtime Stage ====================
FROM python:3.11-slim as runtime

WORKDIR /app

# Install runtime dependencies including FFmpeg
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application code
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY pyproject.toml README.md ./

# Create directories for models and output
RUN mkdir -p /app/models_cache /app/output /app/temp

# Set environment variables
ENV PYTHONPATH="/app/src:$PYTHONPATH"
ENV MODEL_CACHE_DIR="/app/models_cache"
ENV OUTPUT_DIR="/app/output"
ENV TEMP_DIR="/app/temp"
ENV DEVICE="auto"
ENV PRECISION="bf16"

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python -c "import video_translator; print('OK')" || exit 1

# Default command
CMD ["python", "-m", "video_translator.cli", "--help"]

# ==================== Development Stage ====================
FROM runtime as dev

# Install development dependencies
RUN pip install --no-cache-dir \
    pytest \
    pytest-asyncio \
    black \
    ruff \
    mypy

# Install package in editable mode
COPY . .
RUN pip install -e .

CMD ["bash"]
