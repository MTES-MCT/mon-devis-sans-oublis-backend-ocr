# Use NVIDIA CUDA 12.8 development image for Flash Attention compilation
# CUDA 12.8 is required for PyTorch 2.7.0 and Flash Attention compatibility
FROM nvidia/cuda:12.8.0-cudnn9-devel-ubuntu22.04

# Set the working directory in the container
WORKDIR /app

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# Install Python 3.11 and system dependencies
RUN apt-get update && apt-get install -y \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa -y \
    && apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3.11-distutils \
    python3-pip \
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
    git \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default python3
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

# Upgrade pip
RUN python3.11 -m pip install --upgrade pip setuptools wheel

# Copy the requirements file into the container at /app
COPY ./requirements.txt /app/requirements.txt
# Install PyTorch 2.7.0 with CUDA 12.8 (required for marker-pdf and Flash Attention)
RUN pip install --no-cache-dir torch==2.7.0 torchvision --index-url https://download.pytorch.org/whl/cu128

# Install Flash Attention 2 compatible with PyTorch 2.7.0
ENV MAX_JOBS=4
RUN pip install --no-cache-dir flash-attn --no-build-isolation

# Install remaining packages
RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt --index-url https://download.pytorch.org/whl/cu128

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
CMD ["sh", "-c", "if [ \"${WORKERS:-1}\" = \"1\" ]; then uvicorn app.main:app --host ${HOST:-0.0.0.0} --port ${PORT:-80}; else gunicorn app.main:app -c gunicorn_config.py; fi"]
