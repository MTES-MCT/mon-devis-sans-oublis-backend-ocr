# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

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

# Copy and run the test script to verify marker installation
COPY test_marker_import.py /app/test_marker_import.py
RUN python /app/test_marker_import.py

# First copy only the files needed for downloading models
COPY app/services/ocr/base.py /app/app/services/ocr/base.py
COPY app/services/ocr/nanonets.py /app/app/services/ocr/nanonets.py
COPY app/services/ocr/olmocr.py /app/app/services/ocr/olmocr.py
# Don't copy marker.py yet - it will be copied with the rest of the app
COPY app/services/ocr/__init__.py /app/app/services/ocr/__init__.py

COPY app/__init__.py /app/app/__init__.py
COPY app/services/__init__.py /app/app/services/__init__.py
COPY download_models.py /app/download_models.py

# Run the download script to populate the cache
# This layer will be cached as long as the download-related files don't change.
RUN python download_models.py

# Now copy the rest of the application
COPY . /app

# Run uvicorn when the container launches
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]
