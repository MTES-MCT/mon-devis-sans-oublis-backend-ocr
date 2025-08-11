# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Set HuggingFace cache directory to /scratch to avoid disk space issues
ENV HF_HOME=/scratch/huggingface_cache
ENV HUGGINGFACE_HUB_CACHE=/scratch/huggingface_cache/hub
ENV TRANSFORMERS_CACHE=/scratch/huggingface_cache/transformers
ENV HF_DATASETS_CACHE=/scratch/huggingface_cache/datasets

# Create the cache directory with proper permissions
RUN mkdir -p /scratch/huggingface_cache && chmod 777 /scratch/huggingface_cache

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
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container at /app
COPY ./requirements.txt /app/requirements.txt

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121

# Skipping model downloads during build; models will be cached at runtime in /scratch
# Application code is mounted at runtime via docker-compose bind mount

# Run uvicorn when the container launches
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]
