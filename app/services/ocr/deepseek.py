from .sglang_base import SGLangOCRService
from app.config import config


class DeepSeekOCRService(SGLangOCRService):
    """DeepSeek OCR service using SGLang"""
    
    _service_name = "deepseek-ocr"
    _model_name = "deepseek-ocr"  # Matches --served-model-name in docker-compose
    
    def __init__(self):
        self._endpoint = config.DEEPSEEK_ENDPOINT
        self._system_prompt = "Convert the document to markdown."
        super().__init__()
