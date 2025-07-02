# Mon Devis Sans Oublis - Backend OCR

This is a FastAPI backend for OCR processing.

## Running the application

Build and run the application using Docker Compose:
```bash
docker-compose up --build
```

## API Usage

The application provides an OCR endpoint that can process PDF and image files.

### Nanonets OCR

To use the Nanonets OCR model, send a POST request to `/ocr/nanonets`:
```bash
curl -X POST \
  -F "file=@/path/to/your/document.pdf" \
  -H "x-api-key: mysecretapikey" \
  http://localhost:8000/ocr/nanonets
```

### OlmOCR

To use the OlmOCR model, send a POST request to `/ocr/olmocr`:
```bash
curl -X POST \
  -F "file=@/path/to/your/document.pdf" \
  -H "x-api-key: mysecretapikey" \
  http://localhost:8000/ocr/olmocr
