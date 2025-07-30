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
  https://ocr.mon-devis-sans-oublis.beta.gouv.fr/ocr/olmocr

### Nginx Configuration

Create or edit the Nginx configuration file at `/etc/nginx/sites-available/ocr.mon-devis-sans-oublis.beta.gouv.fr`:

```nginx
server {
    server_name ocr.mon-devis-sans-oublis.beta.gouv.fr;


    location / {
        proxy_pass http://127.0.0.1:8000/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
```

Enable the site and obtain SSL certificate:
```bash
ln -s /etc/nginx/sites-available/ocr.mon-devis-sans-oublis.beta.gouv.fr /etc/nginx/sites-enabled/
certbot --nginx -d ocr.mon-devis-sans-oublis.beta.gouv.fr
```

## Deployment

### Server Information
- **Server IP**: 51.159.149.211
- **User**: root | erwan
- **Project Location**: `/mon-devis-sans-oublis-backend-ocr/`

### Redeployment Process

To redeploy the application on the production server, follow these steps:

1. **Connect to the server via SSH**:
   ```bash
   ssh root@51.159.149.211
   ```

2. **Navigate to the project directory**:
   ```bash
   cd /mon-devis-sans-oublis-backend-ocr/
   ```

3. **Pull the latest changes from the repository**:
   ```bash
   git pull origin main
   ```

4. **Stop the running containers (optional)**:
   ```bash
   docker-compose down
   ```

5. **Rebuild and start the containers**:
   ```bash
   docker-compose up -d --build
   ```





   Press `Ctrl+C` to exit the log view.


### Verifying the Deployment

After deployment, you can verify the application is running:

1. **Check container status**:
   ```bash
   docker-compose ps
   ```


### Environment Variables

The application uses a `.env` file for configuration. Make sure the following environment variables are set on the server:

- `API_KEY`: The API key for authentication (you can put )
- Any other service-specific API keys

### Troubleshooting

If you encounter issues during deployment:

1. **Check Docker logs**:
   ```bash
   docker-compose logs api
   ```

2. **Ensure Docker and Docker Compose are installed**:
   ```bash
   docker --version
   docker-compose --version
   ```

3. **Verify GPU availability (if using GPU-based OCR)**:
   ```bash
   nvidia-smi












### Important Notes



- The Hugging Face model cache is persisted in a Docker volume to avoid re-downloading models
- GPU support is enabled in the docker-compose configuration for better OCR performance
- Always backup the `.env` file before making changes
