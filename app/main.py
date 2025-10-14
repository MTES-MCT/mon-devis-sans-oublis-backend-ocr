import os
import sentry_sdk
from fastapi import Depends, FastAPI, HTTPException, status
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

sentry_sdk.init(
    dsn=os.getenv("SENTRY_DNS"),
    # Add data like request headers and IP for users,
    # see https://docs.sentry.io/platforms/python/data-management/data-collected/ for more info
    send_default_pii=True,
)

app = FastAPI(
    title="OCR Backend Service",
    description="Multi-model OCR service with configurable workers and services",
    dependencies=[Depends(get_api_key)]
)

@app.get("/sentry-debug")
async def trigger_error():
    division_by_zero = 1 / 0

app.include_router(api_router)

@app.on_event("startup")
async def startup_event():
    """Log startup information"""
    print(f"Starting OCR Backend Service")
    print(f"Enabled services: {config.get_enabled_services()}")
    print(f"Workers configured: {config.WORKERS}")
    print(f"API Key protection: {'Enabled' if config.API_KEY else 'Disabled'}")