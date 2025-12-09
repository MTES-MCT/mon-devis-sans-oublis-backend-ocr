import os
from typing import List, Optional

class Config:
    """Configuration class for OCR service"""
    
    # API Configuration
    API_KEY: Optional[str] = os.getenv("API_KEY")
    API_KEY_NAME: str = "x-api-key"
    
    # Service Configuration
    ENABLED_SERVICES: List[str] = os.getenv(
        "ENABLED_SERVICES",
        "marker,nanonets,deepseek-ocr,hunyuan-ocr"  # Default: all services enabled
    ).split(",")
    
    # VLLM Endpoints
    NANONETS_ENDPOINT: str = os.getenv("NANONETS_ENDPOINT", "http://nanonets-vllm:8000/v1")
    DEEPSEEK_ENDPOINT: str = os.getenv("DEEPSEEK_ENDPOINT", "http://deepseek-vllm:8000/v1")
    HUNYUAN_ENDPOINT: str = os.getenv("HUNYUAN_ENDPOINT", "http://hunyuan-vllm:8000/v1")
    
    # Concurrency Control
    MAX_CONCURRENT_OCR_REQUESTS: int = int(os.getenv("MAX_CONCURRENT_OCR_REQUESTS", "3"))
    
    # VLLM Health Check Timeout
    VLLM_HEALTH_TIMEOUT: int = int(os.getenv("VLLM_HEALTH_TIMEOUT", "30"))
    
    # Worker Configuration
    WORKERS: int = int(os.getenv("WORKERS", "1"))
    WORKER_CLASS: str = os.getenv("WORKER_CLASS", "uvicorn.workers.UvicornWorker")
    
    # Server Configuration
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "80"))
    
    # Gunicorn Configuration
    WORKER_CONNECTIONS: int = int(os.getenv("WORKER_CONNECTIONS", "1000"))
    MAX_REQUESTS: int = int(os.getenv("MAX_REQUESTS", "1000"))
    MAX_REQUESTS_JITTER: int = int(os.getenv("MAX_REQUESTS_JITTER", "50"))
    TIMEOUT: int = int(os.getenv("TIMEOUT", "120"))
    GRACEFUL_TIMEOUT: int = int(os.getenv("GRACEFUL_TIMEOUT", "30"))
    KEEPALIVE: int = int(os.getenv("KEEPALIVE", "5"))
    
    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "info")
    ACCESS_LOG: bool = os.getenv("ACCESS_LOG", "true").lower() == "true"
    
    @classmethod
    def get_enabled_services(cls) -> List[str]:
        """Get list of enabled services"""
        return [s.strip() for s in cls.ENABLED_SERVICES if s.strip()]
    
    @classmethod
    def is_service_enabled(cls, service_name: str) -> bool:
        """Check if a specific service is enabled"""
        return service_name in cls.get_enabled_services()
    
    @classmethod
    def get_vllm_endpoint(cls, service_name: str) -> Optional[str]:
        """Get VLLM endpoint for a service"""
        endpoints = {
            "nanonets": cls.NANONETS_ENDPOINT,
            "deepseek-ocr": cls.DEEPSEEK_ENDPOINT,
            "hunyuan-ocr": cls.HUNYUAN_ENDPOINT,
        }
        return endpoints.get(service_name)

config = Config()