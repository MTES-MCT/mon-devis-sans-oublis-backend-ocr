from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import APIKeyHeader
from app.api.routes import router as api_router
from app.config import config

api_key_header = APIKeyHeader(name=config.API_KEY_NAME, auto_error=False)

async def get_api_key(api_key_header: str = Depends(api_key_header)):
    if not config.API_KEY or not api_key_header or api_key_header != config.API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API Key",
        )
    return api_key_header

app = FastAPI(
    title="OCR Backend Service",
    description="Multi-model OCR service with configurable workers and services",
    dependencies=[Depends(get_api_key)]
)

app.include_router(api_router)

@app.on_event("startup")
async def startup_event():
    """Log startup information"""
    print(f"Starting OCR Backend Service")
    print(f"Enabled services: {config.get_enabled_services()}")
    print(f"Workers configured: {config.WORKERS}")
    print(f"API Key protection: {'Enabled' if config.API_KEY else 'Disabled'}")