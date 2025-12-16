# Use an official Python runtime as a parent image
# PyTorch CUDA wheels are not available for Python 3.14 yet.
# Using 3.12 ensures we can install CUDA-enabled torch builds.
FROM python:3.12-slim

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

# Upgrade pip and install CUDA-enabled PyTorch explicitly.
# We do this outside of requirements.txt to avoid pip later overriding it with a
# CPU-only torch wheel from PyPI.
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir \
      torch torchvision \
      --index-url https://download.pytorch.org/whl/cu121

# Install remaining packages
# IMPORTANT: don't use `--upgrade` here, otherwise pip may replace the CUDA torch
# wheel (e.g. `2.x.y+cu121`) with a CPU wheel from PyPI (`2.x.y`) because the
# local version suffix has lower precedence.
RUN pip install --no-cache-dir -r /app/requirements.txt

# First copy only the files needed for downloading models
# Only marker models need to be downloaded in the backend
# VLLM models are loaded in separate containers
COPY app/services/ocr/base.py /app/app/services/ocr/base.py
COPY app/services/ocr/vllm_base.py /app/app/services/ocr/vllm_base.py
COPY app/services/ocr/marker.py /app/app/services/ocr/marker.py
COPY app/services/ocr/__init__.py /app/app/services/ocr/__init__.py

COPY app/__init__.py /app/app/__init__.py
COPY app/services/__init__.py /app/app/services/__init__.py
COPY app/config.py /app/app/config.py
COPY download_models.py /app/download_models.py

# Optional: download marker models during image build.
#
# IMPORTANT: if you install a CUDA-enabled torch wheel, importing torch during
# `docker build` may fail because NVIDIA driver libraries are not mounted at
# build time. Therefore this is opt-in.
ARG DOWNLOAD_MODELS=0

# Only download marker models - VLLM models are loaded in separate containers
ENV ENABLED_SERVICES="marker"
ENV HF_HUB_OFFLINE="0"

# Run the download script to populate the cache (opt-in)
RUN if [ "${DOWNLOAD_MODELS}" = "1" ]; then \
      python download_models.py; \
    else \
      echo "Skipping model download during build (DOWNLOAD_MODELS=${DOWNLOAD_MODELS})"; \
    fi

# Now copy the rest of the application
COPY . /app

# Default environment variables (can be overridden at runtime)
ENV WORKERS=1
ENV PORT=80
ENV HOST=0.0.0.0

# Run with Gunicorn for production, with fallback to uvicorn for development
CMD ["sh", "-c", "if [ \"${WORKERS:-1}\" = \"1\" ]; then uvicorn app.main:app --host ${HOST:-0.0.0.0} --port ${PORT:-80}; else gunicorn app.main:app -c gunicorn_config.py; fi"]