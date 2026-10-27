# ── Dockerfile for Hugging Face Spaces / Render ──
FROM python:3.11-slim

# Prevent Python from buffering stdout/stderr
ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PORT=7860

# Install required system packages (Poppler for pdf2image, Tesseract for OCR)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    poppler-utils \
    tesseract-ocr \
    tesseract-ocr-eng \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Upgrade pip and install CPU-only PyTorch first (drastically reduces image size & memory)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Copy requirements and install Python dependencies
COPY backend/requirements.txt backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

# Create non-root user (required by Hugging Face Spaces standard)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Set working directory inside user home
WORKDIR $HOME/app

# Copy all application files
COPY --chown=user . $HOME/app

# Create storage directories with proper permissions
RUN mkdir -p data/uploads data/chroma

# Expose default port
EXPOSE 7860

# Start FastAPI application using Uvicorn
CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-7860}"]
