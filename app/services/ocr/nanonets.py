from typing import List
from PIL import Image
from .vllm_base import VLLMOCRService
from app.config import config


class NanonetsOCRService(VLLMOCRService):
    """Nanonets OCR service using VLLM"""
    
    _service_name = "nanonets"
    _model_name = "nanonets/Nanonets-OCR2-3B"
    
    def __init__(self):
        # Set endpoint before calling parent __init__
        self._endpoint = config.NANONETS_ENDPOINT
        
        # Set optimized prompt for Nanonets
        self._system_prompt = (
            "Extract and return all the text from this image. "
            "Include all text elements and maintain the reading order and line breaks. "
            "If there are tables, convert them to markdown format while including "
            "line breaks in the cells using <br/> tag. "
            "If there are mathematical equations, convert them to LaTeX format. "
            "Escape characters if necessary. "
            "Cells might contain long text, do not create new cells on your own."
        )
        
        super().__init__()