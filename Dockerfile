# ========================================================
# Deepfake Video Detection — Production GPU Container
# Base: PyTorch 2.2.0 with CUDA 12.1 and cuDNN 8 runtime
# ========================================================

FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime AS base

# System metadata
LABEL maintainer="Deepfake Detection Research Team"
LABEL description="Inference service and technical dashboard for ResNeXt-50 + Attention + LSTM deepfake detector"

# Prevent interactive prompts during apt install
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install required system libraries for OpenCV and video decoding
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root application user
RUN useradd -m -u 1000 appuser

WORKDIR /app

# Install Python requirements
COPY requirements.txt pyproject.toml ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir fastapi uvicorn python-multipart httpx

# Copy application source code and configurations
COPY src/ ./src/
COPY configs/ ./configs/

# Create mount points for models and data with appropriate permissions
RUN mkdir -p /app/checkpoints /app/reports /app/data && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose API and dashboard port
EXPOSE 8000

# Health check probe
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default environment configuration
ENV DEEPFAKE_CHECKPOINT_PATH="/app/checkpoints/ffpp_c23_baseline_soft_lstm/best.pth"
ENV DEEPFAKE_DEVICE="cuda"

# Start FastAPI application with uvicorn
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
