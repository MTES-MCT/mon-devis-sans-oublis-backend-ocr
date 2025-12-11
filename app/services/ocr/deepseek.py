from typing import List
from PIL import Image
from .vllm_base import VLLMOCRService
from app.config import config


class DeepSeekOCRService(VLLMOCRService):
    """DeepSeek OCR service using VLLM"""
    
    _service_name = "deepseek-ocr"
    _model_name = "deepseek-ai/DeepSeek-OCR"
    
    def __init__(self):
        # Set endpoint before calling parent __init__
        self._endpoint = config.DEEPSEEK_ENDPOINT
        
        self._system_prompt = "Extract all text from this document."
        
        super().__init__()