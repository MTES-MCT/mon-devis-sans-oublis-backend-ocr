# Mon Devis Sans Oublis - Backend OCR

This is a FastAPI backend for OCR processing with configurable workers and services support.

## Features

- **Multiple OCR Models**: Support for Marker, Nanonets, and OlmOCR
- **Configurable Workers**: Run with multiple Gunicorn workers for production scalability
- **Service Selection**: Enable/disable specific OCR services based on your needs
- **GPU Support**: Optimized for GPU acceleration when available
- **Docker Support**: Easy deployment with Docker and Docker Compose

## Configuration

### Environment Variables

The application can be configured using environment variables. See `.env.example` for all available options:

| Variable | Description | Default | Example |
|----------|-------------|---------|---------|
| `API_KEY` | API key for authentication | - | `your-secure-api-key` |
| `ENABLED_SERVICES` | Comma-separated list of enabled OCR services | `marker,nanonets,olmocr` | `marker` |
| `WORKERS` | Number of Gunicorn worker processes | `1` | `4` |
| `WORKER_CLASS` | Gunicorn worker class | `uvicorn.workers.UvicornWorker` | - |
| `TIMEOUT` | Request timeout in seconds | `120` | `180` |
| `LOG_LEVEL` | Logging level | `info` | `debug` |
| `PORT` | Server port | `80` | `8080` |
| `CUDA_VISIBLE_DEVICES` | GPU device ID | `0` | `0,1` |

### Service Configuration Examples

#### Production Setup (Marker only with 4 workers)
```bash
ENABLED_SERVICES=marker
WORKERS=4
LOG_LEVEL=info
```

#### Development Setup (All services with 1 worker)
```bash
ENABLED_SERVICES=marker,nanonets,olmocr
WORKERS=1
LOG_LEVEL=debug
```

## Running the Application

### Development Mode

Using the development Docker Compose configuration with all services enabled:

```bash
# Copy and configure the development environment file
cp .env.dev .env

# Build and run with development configuration
docker-compose -f docker-compose.dev.yml up --build
```

The development setup:
- Enables all OCR services (marker, nanonets, olmocr)
- Runs with a single worker for easier debugging
- Enables hot-reload for code changes
- Exposes the service on port 8001

### Production Mode

Using the production Docker Compose configuration with Marker and 4 workers:

```bash
# Copy and configure the production environment file
cp .env.prod .env
# Edit .env to set your production API key

# Build and run with production configuration
docker-compose -f docker-compose.prod.yml up -d --build
```

The production setup:
- Enables only the Marker service for optimal performance
- Runs with 4 Gunicorn workers for better concurrency
- Optimized timeouts and connection settings
- Exposes the service on port 8080

### Custom Configuration

You can also run with custom settings by modifying the environment variables:

```bash
# Run with 2 workers and only Marker + OlmOCR
docker run -d \
  -e API_KEY=your-api-key \
  -e ENABLED_SERVICES=marker,olmocr \
  -e WORKERS=2 \
  -p 8000:80 \
  ocr-backend:latest
```

## API Usage

### Available Endpoints

- `GET /services` - List enabled OCR services
- `GET /health` - Health check with GPU memory status
- `POST /ocr/{service_name}` - Process a document with specified OCR service

### Example: Marker OCR

```bash
curl -X POST \
  -F "file=@/path/to/your/document.pdf" \
  -H "x-api-key: your-api-key" \
  http://localhost:8000/ocr/marker
```

### Example: Nanonets OCR

```bash
curl -X POST \
  -F "file=@/path/to/your/document.pdf" \
  -H "x-api-key: your-api-key" \
  http://localhost:8000/ocr/nanonets
```

### Example: OlmOCR

```bash
curl -X POST \
  -F "file=@/path/to/your/document.pdf" \
  -H "x-api-key: your-api-key" \
  http://localhost:8000/ocr/olmocr
```

### Check Available Services

```bash
curl -H "x-api-key: your-api-key" \
  http://localhost:8000/services
```

Response:
```json
{
  "services": ["marker", "nanonets", "olmocr"]
}
```

### Health Check

```bash
curl -H "x-api-key: your-api-key" \
  http://localhost:8000/health
```

Response:
```json
{
  "status": "healthy",
  "services": ["marker"],
  "gpu": {
    "available": true,
    "allocated_gb": 2.5,
    "reserved_gb": 3.0,
    "total_gb": 8.0,
    "free_gb": 5.0
  }
}
```

## Deployment

### Server Information
- **Server IP**: SERVER_IP
- **User**: root | erwan
- **Project Location**: `/mon-devis-sans-oublis-backend-ocr/`

### Redeployment Process

1. **Connect to the server via SSH**:
   ```bash
   ssh root@SERVER_IP
   ```

2. **Navigate to the project directory**:
   ```bash
   cd /mon-devis-sans-oublis-backend-ocr/
   ```

3. **Pull the latest changes**:
   ```bash
   git pull origin main
   ```

4. **Deploy with desired configuration**:
   
   For production (Marker with 4 workers):
   ```bash
   docker-compose -f docker-compose.prod.yml down
   docker-compose -f docker-compose.prod.yml up -d --build
   ```
   
   For development (all services):
   ```bash
   docker-compose -f docker-compose.dev.yml down
   docker-compose -f docker-compose.dev.yml up -d --build
   ```

5. **Verify deployment**:
   ```bash
   docker-compose -f docker-compose.prod.yml ps
   docker-compose -f docker-compose.prod.yml logs --tail=50
   ```

### Nginx Configuration

For production deployment behind Nginx:

```nginx
server {
    server_name ocr.mon-devis-sans-oublis.beta.gouv.fr;
    
    client_max_body_size 100M;  # Allow large file uploads
    proxy_read_timeout 300s;     # Increase timeout for OCR processing
    
    location / {
        proxy_pass http://127.0.0.1:8080/;  # Note: port 8080 for production
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # WebSocket support (if needed)
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
```

Enable the site and obtain SSL certificate:
```bash
ln -s /etc/nginx/sites-available/ocr.mon-devis-sans-oublis.beta.gouv.fr /etc/nginx/sites-enabled/
certbot --nginx -d ocr.mon-devis-sans-oublis.beta.gouv.fr
nginx -s reload
```

## Performance Tuning

### Worker Configuration

The number of workers should be configured based on your server's resources:

- **CPU-bound tasks**: `workers = (2 × CPU cores) + 1`
- **GPU-bound tasks**: `workers = number of GPUs` (usually 1-2 per GPU)
- **Memory considerations**: Each worker loads models into memory

Example configurations:

```bash
# Small server (2 CPU cores, 4GB RAM)
WORKERS=2
MAX_REQUESTS=500
TIMEOUT=120

# Medium server (4 CPU cores, 16GB RAM, 1 GPU)
WORKERS=4
MAX_REQUESTS=1000
TIMEOUT=180

# Large server (8 CPU cores, 32GB RAM, 2 GPUs)
WORKERS=8
MAX_REQUESTS=2000
TIMEOUT=240
```

### Service Selection

For optimal performance, enable only the services you need:

```bash
# High accuracy, slower processing
ENABLED_SERVICES=marker

# Fast processing, moderate accuracy
ENABLED_SERVICES=nanonets

# Balance of speed and accuracy
ENABLED_SERVICES=marker,olmocr
```

## Troubleshooting

### Check Docker logs
```bash
# For production
docker-compose -f docker-compose.prod.yml logs -f ocr-backend

# For development
docker-compose -f docker-compose.dev.yml logs -f ocr-backend-dev
```

### Monitor GPU usage
```bash
nvidia-smi -l 1
```

### Test service availability
```bash
# Check if specific service is enabled
curl -H "x-api-key: your-api-key" http://localhost:8000/services

# Test OCR processing
curl -X POST \
  -F "file=@test.pdf" \
  -H "x-api-key: your-api-key" \
  http://localhost:8000/ocr/marker
```

### Common Issues

1. **Out of Memory (OOM)**:
   - Reduce number of workers
   - Enable fewer services
   - Increase Docker memory limits

2. **Slow Processing**:
   - Increase number of workers (if CPU/memory allows)
   - Ensure GPU is properly configured
   - Check `CUDA_VISIBLE_DEVICES` setting

3. **Service Not Found**:
   - Verify service is in `ENABLED_SERVICES`
   - Check Docker logs for initialization errors
   - Ensure models are properly downloaded

## Important Notes

- The Hugging Face model cache is persisted in a Docker volume to avoid re-downloading models
- GPU support requires NVIDIA Docker runtime
- Always backup the `.env` file before making changes
- Monitor memory usage when increasing worker count
- Each worker loads its own copy of the models into memory
