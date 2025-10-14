"""
Custom exception classes for OCR service error handling.

These exceptions provide structured error information with appropriate HTTP status codes
and detailed context for debugging and user feedback.
"""


class OCRException(Exception):
    """Base exception for OCR service"""
    
    def __init__(self, message: str, details: dict = None, status_code: int = 500):
        self.message = message
        self.details = details or {}
        self.status_code = status_code
        super().__init__(self.message)


class InvalidFileFormatError(OCRException):
    """Raised when file format is not supported"""
    
    def __init__(self, file_format: str, supported_formats: list = None):
        super().__init__(
            message=f"Unsupported file format: {file_format}",
            details={
                "format": file_format, 
                "supported": supported_formats or [".pdf", ".png", ".jpg", ".jpeg", ".bmp", ".gif"]
            },
            status_code=400
        )


class CorruptedImageError(OCRException):
    """Raised when image file cannot be opened or is corrupted"""
    
    def __init__(self, filename: str, error_detail: str = None):
        super().__init__(
            message=f"Cannot process image file: {filename}",
            details={
                "filename": filename, 
                "error": error_detail
            },
            status_code=422
        )


class OCRProcessingError(OCRException):
    """Raised when OCR processing fails"""
    
    def __init__(self, service_name: str, error_detail: str = None):
        super().__init__(
            message=f"OCR processing failed with service: {service_name}",
            details={
                "service": service_name, 
                "error": error_detail
            },
            status_code=500
        )


class GPUMemoryError(OCRException):
    """Raised when GPU runs out of memory"""
    
    def __init__(self, service_name: str, retry_attempted: bool = False):
        super().__init__(
            message="GPU out of memory error",
            details={
                "service": service_name, 
                "retry_attempted": retry_attempted
            },
            status_code=503
        )