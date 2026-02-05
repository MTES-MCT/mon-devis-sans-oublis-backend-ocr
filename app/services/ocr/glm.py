from typing import List
from PIL import Image
from .vllm_base import VLLMOCRService
from app.config import config


class GLMOCRService(VLLMOCRService):
    """GLM OCR service using VLLM"""
    
    _service_name = "glm-ocr"
    _model_name = "zai-org/GLM-OCR"
    
    def __init__(self):
        # Set endpoint before calling parent __init__
        self._endpoint = config.GLM_OCR_ENDPOINT
        
        # Simple prompt for OCR
        self._system_prompt = "Convert the document to markdown."
        
        super().__init__()
