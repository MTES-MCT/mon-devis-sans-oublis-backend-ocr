from .sglang_base import SGLangOCRService
from app.config import config


class GLMOCRService(SGLangOCRService):
    """GLM OCR service using SGLang with speculative decoding"""
    
    _service_name = "glm-ocr"
    _model_name = "glm-ocr"  # Matches --served-model-name in docker-compose
    
    def __init__(self):
        self._endpoint = config.GLM_OCR_ENDPOINT
        self._system_prompt = "Text Recognition:"
        super().__init__()
