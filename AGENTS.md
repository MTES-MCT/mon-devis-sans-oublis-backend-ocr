# AGENTS.md

This file provides guidance to coding assistant when working with code in this repository.

## Overview

FastAPI backend service providing OCR (Optical Character Recognition) capabilities through multiple OCR engines. The service accepts PDF and image files and returns extracted text.

## Architecture

### Core Components

**Service Layer (`app/services/ocr/`):**
- `base.py`: Abstract base class `BaseOCRService` that all OCR services inherit from
- `sglang_base.py`: Base class `SGLangOCRService` for SGLang-based services with async processing and speculative decoding
- Services are auto-discovered via the registration system in `__init__.py`
- Two types of service implementations:
  1. **Direct Model Services**: `process_images(images: List[Image.Image]) -> List[str]`
  2. **SGLang Services**: `process_images_async(images: List[Image.Image]) -> List[str]` (async)
- Three OCR engines available:
  - **marker** (`marker.py`): Direct model using marker-pdf library, converts images to PDF then processes (PROD)
  - **deepseek-ocr** (`deepseek.py`): SGLang-based using deepseek-ai/DeepSeek-OCR model (DEV)
  - **glm-ocr** (`glm.py`): SGLang-based using zai-org/GLM-OCR with speculative decoding (DEV)

**API Layer (`app/api/routes.py`):**
- `POST /ocr/{service_name}`: Main OCR endpoint accepting file uploads
- `GET /services`: Lists available OCR services
- `GET /health`: Health check with GPU memory statistics
- Handles PDF to image conversion, APNG frame extraction, and memory management
- **Dual processing modes**:
  - VLLM services: Async batch processing with concurrent image handling
  - Direct model services (marker): Threadpool execution for sync operations

**Authentication (`app/main.py`):**
- API key authentication via `x-api-key` header
- Key stored in `API_KEY` environment variable

### Key Design Patterns

**Plugin Architecture:**
- OCR services auto-register by inheriting from `BaseOCRService` or `VLLMOCRService` and setting `_service_name` class attribute
- Discovery happens automatically in `app/services/ocr/__init__.py` via `discover_services()`
- Allows adding new OCR engines by simply creating new service files
- VLLM services automatically get async processing, semaphore-based concurrency control, and OpenAI client integration

**Memory Management:**
- Explicit GPU memory cleanup with `torch.cuda.empty_cache()` and `gc.collect()`
- Per-image processing with cleanup between images
- OOM error handling with retry logic
- Temporary file cleanup in finally blocks

**Model Loading:**
- **Direct models** (marker): `download_models.py` script pre-downloads during Docker build
- **VLLM models**: Loaded in separate VLLM containers, cached in shared volume
- Separates model download layer from application code layer for better Docker caching
- VLLM containers handle their own model downloads on first startup
- Reduces container startup time in production (models cached after first run)

## Common Commands

### Development

```bash
# Build and run with Docker Compose
docker-compose -f docker-compose.dev.yml up --build

# Run in detached mode
docker-compose -f docker-compose.dev.yml up -d --build

# View logs
docker-compose -f docker-compose.dev.yml logs -f api

# View VLLM service logs
docker-compose -f docker-compose.dev.yml logs -f nanonets-vllm
docker-compose -f docker-compose.dev.yml logs -f deepseek-vllm
docker-compose -f docker-compose.dev.yml logs -f hunyuan-vllm

# Check container status
docker-compose -f docker-compose.dev.yml ps

# Stop containers
docker-compose -f docker-compose.dev.yml down
```

### Testing the API

```bash
# Test Marker OCR (direct model)
curl -X POST \
  -F "file=@/path/to/document.pdf" \
  -H "x-api-key: mysecretapikey" \
  http://localhost:8001/ocr/marker

# Test Nanonets OCR (VLLM)
curl -X POST \
  -F "file=@/path/to/document.pdf" \
  -H "x-api-key: mysecretapikey" \
  http://localhost:8001/ocr/nanonets

# Test DeepSeek OCR (VLLM)
curl -X POST \
  -F "file=@/path/to/document.pdf" \
  -H "x-api-key: mysecretapikey" \
  http://localhost:8001/ocr/deepseek-ocr

# Test Hunyuan OCR (VLLM)
curl -X POST \
  -F "file=@/path/to/document.pdf" \
  -H "x-api-key: mysecretapikey" \
  http://localhost:8001/ocr/hunyuan-ocr

# List available services
curl http://localhost:8001/services

# Health check
curl http://localhost:8001/health
```

### VLLM Service Management

```bash
# Check VLLM service health (from host)
docker-compose -f docker-compose.dev.yml exec nanonets-vllm curl http://localhost:8000/health

# Restart a VLLM service
docker-compose -f docker-compose.dev.yml restart nanonets-vllm

# Check VLLM models endpoint
docker-compose -f docker-compose.dev.yml exec nanonets-vllm curl http://localhost:8000/v1/models

# Monitor GPU usage
watch -n 1 nvidia-smi
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

### Adding an SGLang-based Service

1. Add SGLang container to `docker-compose.dev.yml` with appropriate speculative decoding settings
2. Add endpoint configuration to `app/config.py`
3. Create service file in `app/services/ocr/` (e.g., `newocr.py`)
4. Inherit from `SGLangOCRService`
5. Set `_service_name`, `_model_name`, and `_system_prompt`
6. The service will be auto-discovered and registered

Example:
```python
from .sglang_base import SGLangOCRService
from app.config import config

class NewOCRService(SGLangOCRService):
    _service_name = "new-ocr"
    _model_name = "new-ocr"  # Matches --served-model-name in docker-compose
    
    def __init__(self):
        self._endpoint = config.NEW_OCR_ENDPOINT
        self._system_prompt = "Your optimized prompt here"
        super().__init__()
```

### Adding a Direct Model Service

1. Create service file in `app/services/ocr/`
2. Inherit from `BaseOCRService`
3. Set `_service_name` class attribute
4. Implement `process_images(images: List[Image.Image]) -> List[str]`

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
- **SGLang services**: Process images concurrently with semaphore control (max 5 concurrent by default)
- **Marker service**: Converts all images to single PDF before processing
- PDF pages converted to images at 150 DPI
- APNG files supported with frame extraction
- All endpoints require API key authentication except health check
- SGLang containers download models on first startup (can take 10-30 minutes)
- Subsequent startups are fast as models are cached in shared volume

## SGLang Architecture

### Service Types

The project now uses two types of OCR services:

1. **Direct Model Services (Marker)** - PROD
   - Location: Inside FastAPI backend
   - Processing: Synchronous via threadpool
   - GPU: Dynamic allocation

2. **SGLang Services (DeepSeek, GLM-OCR)** - DEV
   - Location: Separate Docker containers using `lmsysorg/sglang:dev`
   - API: OpenAI-compatible HTTP
   - Processing: Async with concurrent batching
   - GPU: Pre-allocated per container via `--mem-fraction-static`
   - GLM-OCR uses speculative decoding for high throughput

### Service Configuration

Control which services are enabled via environment variable:

```bash
# Enable all services (dev)
ENABLED_SERVICES=deepseek-ocr,glm-ocr

# Only marker (prod)
ENABLED_SERVICES=marker
```

### GPU Memory Allocation (DEV)

- **GLM-OCR SGLang**: 40% (--mem-fraction-static 0.4)
- **DeepSeek SGLang**: 40% (--mem-fraction-static 0.4)

Note: Only run one SGLang service at a time, or adjust the memory fractions to fit both.