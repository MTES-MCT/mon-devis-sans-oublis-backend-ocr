# Use CUDA development image with Python 3.11 for flash-attn support
FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

# Set non-interactive to avoid prompts during build
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Install Python 3.11 and essential tools
RUN apt-get update && apt-get install -y \
    software-properties-common \
    tzdata \
    && ln -fs /usr/share/zoneinfo/UTC /etc/localtime \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3.11-distutils \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Upgrade pip
RUN python -m pip install --upgrade pip

# Set the working directory in the container
WORKDIR /app

# Set HuggingFace cache directory to /scratch to avoid disk space issues
ENV HF_HOME=/scratch/huggingface_cache
ENV HUGGINGFACE_HUB_CACHE=/scratch/huggingface_cache/hub
ENV TRANSFORMERS_CACHE=/scratch/huggingface_cache/transformers
ENV HF_DATASETS_CACHE=/scratch/huggingface_cache/datasets

# Set CUDA environment for flash-attn
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# HF caches are configured via ENV; /scratch is a host bind mount provided at runtime via docker-compose

# Install system dependencies for PDF processing and marker-pdf
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    wget \
    poppler-utils \
    tesseract-ocr \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container at /app
COPY ./requirements.txt /app/requirements.txt

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121

# Skipping model downloads during build; models will be cached at runtime in /scratch
# Application code is mounted at runtime via docker-compose bind mount

# Run uvicorn when the container launches
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]
