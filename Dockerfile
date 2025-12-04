# Use an official Python runtime as a parent image
FROM python:3.14-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies for PDF processing, marker-pdf, Pillow, and Flash Attention compilation
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    wget \
    poppler-utils \
    tesseract-ocr \
    build-essential \
    gcc \
    g++ \
    ninja-build \
    zlib1g-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libfreetype6-dev \
    liblcms2-dev \
    libwebp-dev \
    libopenjp2-7-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container at /app
COPY ./requirements.txt /app/requirements.txt

# Install PyTorch first (required for flash-attn compilation)
#RUN pip install --no-cache-dir torch>=2.0.0 torchvision --extra-index-url https://download.pytorch.org/whl/cu121

# Install remaining packages including flash-attn (which requires torch during build)
RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121

# First copy only the files needed for downloading models
COPY app/services/ocr/base.py /app/app/services/ocr/base.py
COPY app/services/ocr/nanonets.py /app/app/services/ocr/nanonets.py
COPY app/services/ocr/olmocr.py /app/app/services/ocr/olmocr.py
# Don't copy marker.py yet - it will be copied with the rest of the app
COPY app/services/ocr/__init__.py /app/app/services/ocr/__init__.py

COPY app/__init__.py /app/app/__init__.py
COPY app/services/__init__.py /app/app/services/__init__.py
COPY app/config.py /app/app/config.py
COPY download_models.py /app/download_models.py

# Set default environment variables for model download
ENV ENABLED_SERVICES="marker,nanonets,olmocr"
ENV HF_HUB_OFFLINE="0"

# Run the download script to populate the cache
# This layer will be cached as long as the download-related files don't change.
RUN python download_models.py

# After models are downloaded, set offline mode as default
ENV HF_HUB_OFFLINE="1"

# Now copy the rest of the application
COPY . /app

# Default environment variables (can be overridden at runtime)
ENV WORKERS=1
ENV PORT=80
ENV HOST=0.0.0.0

# Run with Gunicorn for production, with fallback to uvicorn for development
CMD ["sh", "-c", "if [ \"${WORKERS:-1}\" = \"1\" ]; then uvicorn app.main:app --host ${HOST:-0.0.0.0} --port ${PORT:-80}; else gunicorn app.main:app -c gunicorn_config.py; fi"]% mdso-gpu[erwan]➜  mon-devis-sans-oublis-backend-ocr git:(main) ✗ 