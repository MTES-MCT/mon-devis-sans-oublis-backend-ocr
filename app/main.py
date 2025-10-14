import os
import logging
import sentry_sdk
from sentry_sdk.integrations.fastapi import FastApiIntegration
from sentry_sdk.integrations.logging import LoggingIntegration
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import APIKeyHeader
from fastapi.responses import JSONResponse
from app.api.routes import router as api_router
from app.config import config
from app.exceptions import OCRException

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

api_key_header = APIKeyHeader(name=config.API_KEY_NAME, auto_error=False)

async def get_api_key(api_key_header: str = Depends(api_key_header)):
    if not config.API_KEY or not api_key_header or api_key_header != config.API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API Key",
        )
    return api_key_header

def before_send_filter(event, hint):
    """Add custom context to Sentry events"""
    if "exc_info" in hint:
        exc_type, exc_value, tb = hint["exc_info"]
        if isinstance(exc_value, OCRException):
            event["contexts"]["ocr"] = {
                "exception_type": exc_type.__name__,
                "details": getattr(exc_value, 'details', {})
            }
    return event

# Enhanced Sentry initialization
sentry_sdk.init(
    dsn=os.getenv("SENTRY_DSN"),  # Fixed typo from SENTRY_DNS
    environment=os.getenv("ENVIRONMENT", "production"),
    integrations=[
        FastApiIntegration(
            transaction_style="endpoint",
            failed_request_status_codes=[400, 403, 404, 422, 429, 500, 501, 502, 503, 504]
        ),
        LoggingIntegration(
            level=logging.INFO,
            event_level=logging.ERROR
        )
    ],
    traces_sample_rate=1.0 if os.getenv("ENVIRONMENT") == "development" else 0.1,
    send_default_pii=True,
    attach_stacktrace=True,
    before_send=before_send_filter,
    # Monitor file cleanup
    profiles_sample_rate=1.0 if os.getenv("ENVIRONMENT") == "development" else 0.1,
)

app = FastAPI(
    title="OCR Backend Service",
    description="Multi-model OCR service with configurable workers and services",
    dependencies=[Depends(get_api_key)]
)

# Exception handler for all OCR exceptions
@app.exception_handler(OCRException)
async def ocr_exception_handler(request, exc: OCRException):
    """Handle all OCR custom exceptions with proper status codes and Sentry logging"""
    
    # Log to Sentry with context
    with sentry_sdk.push_scope() as scope:
        scope.set_tag("exception_type", exc.__class__.__name__)
        scope.set_context("exception_details", exc.details)
        scope.set_context("request", {
            "url": str(request.url),
            "method": request.method,
            "headers": dict(request.headers)
        })
        sentry_sdk.capture_exception(exc)
    
    # Log locally for debugging
    logger.error(f"OCR Exception: {exc.message}", extra={"details": exc.details})
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.message,
            "details": exc.details,
            "type": exc.__class__.__name__
        }
    )

app.include_router(api_router)

@app.on_event("startup")
async def startup_event():
    """Log startup information"""
    print(f"Starting OCR Backend Service")
    print(f"Enabled services: {config.get_enabled_services()}")
    print(f"Workers configured: {config.WORKERS}")
    print(f"API Key protection: {'Enabled' if config.API_KEY else 'Disabled'}")