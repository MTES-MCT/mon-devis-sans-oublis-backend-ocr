from typing import List
from PIL import Image
from .vllm_base import VLLMOCRService
from app.config import config


class HunyuanOCRService(VLLMOCRService):
    """Hunyuan OCR service using VLLM"""
    
    _service_name = "hunyuan-ocr"
    _model_name = "tencent/HunyuanOCR"
    
    def __init__(self):
        # Set endpoint before calling parent __init__
        self._endpoint = config.HUNYUAN_ENDPOINT
        
        # Set optimized prompt for Hunyuan
        self._system_prompt = (
            "Perform OCR on this image and extract all visible text with high accuracy. "
            "Preserve the document structure, including paragraphs, lists, tables, and "
            "formatting. Return clean, well-structured text."
        )
        
        super().__init__()