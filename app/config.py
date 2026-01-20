import os
from typing import List, Optional, Set

class Config:
    """Configuration class for OCR service"""
    
    # API Configuration
    # Support multiple API keys via comma-separated list
    _API_KEYS_RAW: Optional[str] = os.getenv("API_KEY")
    API_KEYS: Set[str] = set()
    
    if _API_KEYS_RAW:
        # Parse comma-separated keys, strip whitespace, and filter empty strings
        API_KEYS = {key.strip() for key in _API_KEYS_RAW.split(",") if key.strip()}
    
    API_KEY_NAME: str = "x-api-key"
    
    # Service Configuration
    ENABLED_SERVICES: List[str] = os.getenv(
        "ENABLED_SERVICES",
        "marker,deepseek-ocr"  # Default: all services enabled
    ).split(",")
    
    # VLLM Endpoints
    DEEPSEEK_ENDPOINT: str = os.getenv("DEEPSEEK_ENDPOINT", "http://deepseek-vllm:8000/v1")
    
    # Concurrency Control
    MAX_CONCURRENT_OCR_REQUESTS: int = int(os.getenv("MAX_CONCURRENT_OCR_REQUESTS", "3"))
    
    # PDF Processing Configuration
    PDF_DPI: int = int(os.getenv("PDF_DPI", "72"))  # Lower DPI for faster processing, OCR models work well with 72 DPI
    
    # Image Encoding Configuration
    IMAGE_ENCODE_QUALITY: int = int(os.getenv("IMAGE_ENCODE_QUALITY", "85"))  # JPEG quality (1-100), 85 is good balance for OCR
    IMAGE_ENCODE_FORMAT: str = os.getenv("IMAGE_ENCODE_FORMAT", "JPEG")  # Image format for VLLM encoding (JPEG or PNG)
    
    # VLLM Timeout Configuration
    VLLM_HEALTH_TIMEOUT: int = int(os.getenv("VLLM_HEALTH_TIMEOUT", "30"))
    VLLM_REQUEST_TIMEOUT: int = int(os.getenv("VLLM_REQUEST_TIMEOUT", "300"))  # 5 minutes for OCR processing
    VLLM_MAX_TOKENS: int = int(os.getenv("VLLM_MAX_TOKENS", "4096"))  # Maximum tokens for response
    
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
    
    def is_valid_api_key(cls, api_key: str) -> bool:
        """
        Validate if the provided API key is in the list of valid keys.
        
        Args:
            api_key: The API key to validate
            
        Returns:
            True if the key is valid, False otherwise
        """
        if not cls.API_KEYS:
            # If no API keys are configured, authentication is disabled
            return True
        return api_key in cls.API_KEYS
    
    @classmethod
    def get_api_key_count(cls) -> int:
        """
        Get the number of configured API keys.
        
        Returns:
            Number of valid API keys configured
        """
        return len(cls.API_KEYS)

config = Config()