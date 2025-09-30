# AGENTS.md

This file provides guidance to coding assistant when working with code in this repository.

## Overview

FastAPI backend service providing OCR (Optical Character Recognition) capabilities through multiple OCR engines. The service accepts PDF and image files and returns extracted text.

## Architecture

### Core Components

**Service Layer (`app/services/ocr/`):**
- `base.py`: Abstract base class `BaseOCRService` that all OCR services inherit from
- Services are auto-discovered via the registration system in `__init__.py`
- Each service implements `process_images(images: List[Image.Image]) -> List[str]`
- Three OCR engines available:
  - **marker** (`marker.py`): Uses marker-pdf library, converts images to PDF then processes
  - **nanonets** (`nanonets.py`): Uses nanonets/Nanonets-OCR-s model via transformers
  - **olmocr** (`olmocr.py`): Uses allenai/olmOCR-7B-0225-preview model (Qwen2-VL based)

**API Layer (`app/api/routes.py`):**
- `POST /ocr/{service_name}`: Main OCR endpoint accepting file uploads
- `GET /services`: Lists available OCR services
- `GET /health`: Health check with GPU memory statistics
- Handles PDF to image conversion, APNG frame extraction, and memory management
- All OCR processing runs in threadpool to avoid blocking async event loop

**Authentication (`app/main.py`):**
- API key authentication via `x-api-key` header
- Key stored in `API_KEY` environment variable

### Key Design Patterns

**Plugin Architecture:**
- OCR services auto-register by inheriting from `BaseOCRService` and setting `_service_name` class attribute
- Discovery happens automatically in `app/services/ocr/__init__.py` via `discover_services()`
- Allows adding new OCR engines by simply creating new service files

**Memory Management:**
- Explicit GPU memory cleanup with `torch.cuda.empty_cache()` and `gc.collect()`
- Per-image processing with cleanup between images
- OOM error handling with retry logic
- Temporary file cleanup in finally blocks

**Model Loading:**
- `download_models.py` script pre-downloads models during Docker build
- Separates model download layer from application code layer for better Docker caching
- Reduces container startup time in production

## Common Commands

### Development

```bash
# Build and run with Docker Compose
docker-compose up --build

# Run in detached mode
docker-compose up -d --build

# View logs
docker-compose logs -f api

# Check container status
docker-compose ps

# Stop containers
docker-compose down
```

### Testing the API

```bash
# Test Nanonets OCR
curl -X POST \
  -F "file=@/path/to/document.pdf" \
  -H "x-api-key: mysecretapikey" \
  http://localhost:8000/ocr/nanonets

# Test Marker OCR
curl -X POST \
  -F "file=@/path/to/document.pdf" \
  -H "x-api-key: mysecretapikey" \
  http://localhost:8000/ocr/marker

# Test OlmOCR
curl -X POST \
  -F "file=@/path/to/document.pdf" \
  -H "x-api-key: mysecretapikey" \
  http://localhost:8000/ocr/olmocr

# List available services
curl http://localhost:8000/services

# Health check
curl http://localhost:8000/health
```

### Deployment

Production deployment on server at `/mon-devis-sans-oublis-backend-ocr/`:

```bash
# SSH to server
ssh root@SERVER_IP

# Navigate to project
cd /mon-devis-sans-oublis-backend-ocr/

# Pull latest changes
git pull origin main

# Rebuild and restart
docker-compose down
docker-compose up -d --build

# Check logs
docker-compose logs -f api
```

## Environment Configuration

Required environment variables in `.env` file:

- `API_KEY`: Authentication key for API access
- `HF_HOME`: Hugging Face cache directory (set to `/root/.cache/huggingface` in docker-compose)

## Adding New OCR Services

To add a new OCR engine:

1. Create new file in `app/services/ocr/` (e.g., `newocr.py`)
2. Inherit from `BaseOCRService`
3. Set `_service_name` class attribute
4. Implement `process_images(images: List[Image.Image]) -> List[str]` method
5. The service will be auto-discovered and registered

Example:
```python
from .base import BaseOCRService
from typing import List
from PIL import Image

class NewOCRService(BaseOCRService):
    _service_name = "newocr"

    def __init__(self):
        # Initialize your OCR model
        pass

    def process_images(self, images: List[Image.Image]) -> List[str]:
        # Process images and return text
        return ["extracted text"]
```

## Important Notes

- GPU support required for optimal performance (NVIDIA GPU with CUDA)
- Hugging Face models cached in Docker volume to avoid re-downloading
- Service processes images sequentially to manage GPU memory
- Marker service converts all images to single PDF before processing
- PDF pages converted to images at 150 DPI
- APNG files supported with frame extraction
- All endpoints require API key authentication except health check