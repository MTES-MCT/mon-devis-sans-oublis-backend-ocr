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
        
        # Use "Free OCR." prompt as per official documentation
        # Simple prompts work best - complex prompts cause empty responses
        self._system_prompt = "Convert the document to markdown."
        
        # Configure DeepSeek-specific parameters
        # Based on: https://github.com/deepseek-ai/DeepSeek-OCR-V1
        self._extra_body = {
            "skip_special_tokens": False,
            # Custom logits processor parameters for better OCR quality
            "vllm_xargs": {
                "ngram_size": 30,
                "window_size": 90,
                # whitelist: <td>, </td> for table detection
                "whitelist_token_ids": [128821, 128822],
            },
        }
        
        super().__init__()