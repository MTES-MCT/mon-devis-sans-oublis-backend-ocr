import logging
import os
import sys

import sentry_sdk
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
from sentry_sdk import logger as sentry_logger
from sentry_sdk.integrations.fastapi import FastApiIntegration
from sentry_sdk.integrations.logging import LoggingIntegration

from app.api.routes import router as api_router
from app.config import config
from app.exceptions import OCRException

# Configure logging with proper format and handlers
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ],
    force=True  # Force reconfiguration even if already configured
)

# Set log level for our application modules
logging.getLogger('app').setLevel(logging.INFO)

# Get logger for this module
logger = logging.getLogger(__name__)

api_key_header = APIKeyHeader(name=config.API_KEY_NAME, auto_error=False)

async def get_api_key(api_key_header: str = Depends(api_key_header)):
    # If no API keys are configured, authentication is disabled
    if not config.API_KEYS:
        return api_key_header
    
    # Validate the provided API key against the list of valid keys
    if not api_key_header or not config.is_valid_api_key(api_key_header):
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

# Enhanced Sentry initialization with logs enabled
sentry_sdk.init(
    dsn=os.getenv("SENTRY_DSN"),  # Fixed typo from SENTRY_DNS
    environment=os.getenv("ENVIRONMENT", "production"),
    enable_logs=True,  # Enable structured logging
    integrations=[
        FastApiIntegration(
            transaction_style="endpoint",
            failed_request_status_codes=[400, 403, 404, 422, 429, 500, 501, 502, 503, 504]
        ),
        LoggingIntegration(
            level=logging.INFO,
            event_level=logging.ERROR,
            sentry_logs_level=logging.INFO  # Send INFO and above to Sentry logs
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
    sentry_logger.error(
        'OCR exception occurred',
        attributes={
            'exception.type': exc.__class__.__name__,
            'exception.message': exc.message,
            'exception.status_code': exc.status_code,
            'exception.details': exc.details,
            'request.url': str(request.url),
            'request.method': request.method
        }
    )
    
    # Also capture as exception in Sentry
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
    api_key_count = config.get_api_key_count()
    startup_info = {
        'service': 'OCR Backend Service',
        'enabled_services': config.get_enabled_services(),
        'workers': config.WORKERS,
        'api_key_protection': 'Enabled' if api_key_count > 0 else 'Disabled',
        'api_key_count': api_key_count
    }
    
    # Log to console for backward compatibility
    print(f"Starting {startup_info['service']}")
    print(f"Enabled services: {startup_info['enabled_services']}")
    print(f"Workers configured: {startup_info['workers']}")
    print(f"API Key protection: {startup_info['api_key_protection']}")
    if api_key_count > 0:
        print(f"Configured API keys: {api_key_count}")
    
    # Log to Sentry
    sentry_logger.info(
        'OCR service starting up',
        attributes={
            'ocr.enabled_services': startup_info['enabled_services'],
            'ocr.workers': startup_info['workers'],
            'ocr.api_key_protection': startup_info['api_key_protection'],
            'ocr.api_key_count': startup_info['api_key_count'],
            'service.name': 'ocr-backend'
        }
    )