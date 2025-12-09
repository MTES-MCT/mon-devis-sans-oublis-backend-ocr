import base64
import io
import asyncio
import logging
from typing import List, Optional
from abc import abstractmethod
from PIL import Image
from openai import AsyncOpenAI
from .base import BaseOCRService
from app.config import config

logger = logging.getLogger(__name__)


class VLLMOCRService(BaseOCRService):
    """
    Base class for VLLM-based OCR services.
    
    This class provides common functionality for services that use VLLM
    with OpenAI-compatible API for vision-based OCR.
    """
    
    _service_name = "vllm_base"
    _endpoint: Optional[str] = None
    _system_prompt: str = "Extract all text from this image."
    _model_name: str = "default"
    
    def __init__(self):
        """Initialize VLLM service with OpenAI client"""
        super().__init__()
        
        if not self._endpoint:
            raise ValueError(f"Endpoint not configured for {self._service_name}")
        
        # Initialize OpenAI client pointing to VLLM server
        self.client = AsyncOpenAI(
            base_url=self._endpoint,
            api_key="EMPTY",  # VLLM doesn't require API key
            timeout=config.VLLM_HEALTH_TIMEOUT,
        )
        
        # Semaphore for concurrent request control
        self.semaphore = asyncio.Semaphore(config.MAX_CONCURRENT_OCR_REQUESTS)
        
        logger.info(f"Initialized {self._service_name} with endpoint {self._endpoint}")
    
    @staticmethod
    def encode_image_to_base64(image: Image.Image) -> str:
        """
        Encode PIL Image to base64 string for API transmission.
        
        Args:
            image: PIL Image object
            
        Returns:
            Base64 encoded image string
        """
        buffered = io.BytesIO()
        
        # Convert to RGB if necessary
        if image.mode not in ('RGB', 'L'):
            image = image.convert('RGB')
        
        # Save as JPEG for better compression
        image.save(buffered, format="JPEG", quality=95)
        buffered.seek(0)
        
        # Encode to base64
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
        return f"data:image/jpeg;base64,{img_str}"
    
    async def process_single_image(self, image: Image.Image) -> str:
        """
        Process a single image asynchronously using VLLM.
        
        Args:
            image: PIL Image object
            
        Returns:
            Extracted text from the image
        """
        async with self.semaphore:
            try:
                # Encode image to base64
                base64_image = self.encode_image_to_base64(image)
                
                # Create message with vision content
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": base64_image
                                }
                            },
                            {
                                "type": "text",
                                "text": self._system_prompt
                            }
                        ]
                    }
                ]
                
                # Call VLLM via OpenAI API
                response = await self.client.chat.completions.create(
                    model=self._model_name,
                    messages=messages,
                    max_tokens=4096,
                    temperature=0,  # Deterministic output for OCR
                )
                
                # Extract text from response
                text = response.choices[0].message.content
                return text if text else ""
                
            except Exception as e:
                logger.error(f"Error processing image with {self._service_name}: {e}")
                return ""
    
    async def process_images_async(self, images: List[Image.Image]) -> List[str]:
        """
        Process multiple images concurrently using VLLM.
        
        Args:
            images: List of PIL Image objects
            
        Returns:
            List of extracted text strings, one per image
        """
        if not images:
            return []
        
        # Process all images concurrently with semaphore control
        tasks = [self.process_single_image(img) for img in images]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle any exceptions in results
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Error processing image {i} with {self._service_name}: {result}")
                processed_results.append("")
            else:
                processed_results.append(result)
        
        return processed_results
    
    def process_images(self, images: List[Image.Image]) -> List[str]:
        """
        Synchronous wrapper for backward compatibility.
        
        This should not be called directly - use process_images_async instead.
        Only implemented to satisfy base class interface.
        """
        raise NotImplementedError(
            f"{self._service_name} is async-only. Use process_images_async() instead."
        )