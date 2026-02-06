import base64
import io
import asyncio
import logging
import time
from typing import List, Optional
from PIL import Image
from openai import AsyncOpenAI, APIError, APITimeoutError
from .base import BaseOCRService
from app.config import config

logger = logging.getLogger(__name__)


class SGLangProcessingError(Exception):
    """Exception raised when SGLang processing fails"""
    pass


class SGLangOCRService(BaseOCRService):
    """
    Base class for SGLang-based OCR services.
    
    This class provides common functionality for services that use SGLang
    with OpenAI-compatible API for vision-based OCR.
    
    SGLang provides high-throughput inference with speculative decoding support.
    
    Subclasses should set:
    - _service_name: Unique identifier for the service
    - _model_name: Model name as served by SGLang (--served-model-name)
    - _endpoint: SGLang API endpoint URL
    - _system_prompt: Text prompt for OCR
    """
    
    _service_name = "sglang_base"
    _endpoint: Optional[str] = None
    _system_prompt: str = "Extract all text from this image."
    _model_name: str = "default"
    
    def __init__(self):
        """Initialize SGLang service with OpenAI client"""
        super().__init__()
        
        if not self._endpoint:
            raise ValueError(f"Endpoint not configured for {self._service_name}")
        
        # Initialize OpenAI client pointing to SGLang server
        self.client = AsyncOpenAI(
            base_url=self._endpoint,
            api_key="EMPTY",  # SGLang doesn't require API key
            timeout=config.VLLM_REQUEST_TIMEOUT,
        )
        
        # Semaphore for concurrent request control
        self.semaphore = asyncio.Semaphore(config.MAX_CONCURRENT_OCR_REQUESTS)
        
        # Retry configuration
        self.max_retries = 3
        self.retry_delay = 1.0  # Initial delay in seconds
        
        logger.info(f"Initialized {self._service_name} with endpoint {self._endpoint}")
    
    @staticmethod
    def encode_image_to_base64(image: Image.Image) -> str:
        """
        Encode PIL Image to base64 string for API transmission.
        Uses JPEG encoding for faster processing and smaller file sizes.
        """
        buffered = io.BytesIO()
        
        # Convert to RGB if necessary (JPEG requires RGB)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        encode_format = config.IMAGE_ENCODE_FORMAT
        
        if encode_format == "JPEG":
            image.save(buffered, format="JPEG", quality=config.IMAGE_ENCODE_QUALITY, optimize=True)
            mime_type = "image/jpeg"
        else:
            image.save(buffered, format="PNG", optimize=True)
            mime_type = "image/png"
        
        buffered.seek(0)
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
        return f"data:{mime_type};base64,{img_str}"
    
    async def process_single_image(self, image: Image.Image, retry_count: int = 0) -> str:
        """
        Process a single image asynchronously using SGLang with retry logic.
        """
        async with self.semaphore:
            try:
                image_start = time.time()
                logger.info(f"[{self._service_name}] Starting image processing (attempt {retry_count + 1}/{self.max_retries})...")
                
                # Encode image to base64
                encode_start = time.time()
                base64_image = self.encode_image_to_base64(image)
                encode_time = time.time() - encode_start
                logger.debug(f"[{self._service_name}] Image encoding took {encode_time:.3f}s")
                
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
                
                logger.info(f"[{self._service_name}] Sending request to SGLang endpoint: {self._endpoint}")
                
                # Call SGLang via OpenAI API
                request_params = {
                    "model": self._model_name,
                    "messages": messages,
                    "max_tokens": config.VLLM_MAX_TOKENS,
                    "temperature": 0.0,  # Deterministic output for OCR
                }
                
                # Time the SGLang API call
                api_start = time.time()
                response = await self.client.chat.completions.create(**request_params)
                api_time = time.time() - api_start
                logger.info(f"[{self._service_name}] SGLang API call took {api_time:.2f}s")
                
                # Extract text from response
                text = response.choices[0].message.content
                
                if not text:
                    logger.warning(f"[{self._service_name}] Received empty response from SGLang")
                    raise SGLangProcessingError("Empty response from SGLang")
                
                total_time = time.time() - image_start
                logger.info(f"[{self._service_name}] Successfully processed image in {total_time:.2f}s")
                logger.info(f"[{self._service_name}] Response length: {len(text)} chars")
                
                return text
                
            except APITimeoutError as e:
                logger.error(f"[{self._service_name}] Timeout error (attempt {retry_count + 1}/{self.max_retries}): {str(e)}")
                
                if retry_count < self.max_retries - 1:
                    delay = self.retry_delay * (2 ** retry_count)
                    logger.info(f"[{self._service_name}] Retrying after {delay}s...")
                    await asyncio.sleep(delay)
                    return await self.process_single_image(image, retry_count + 1)
                
                raise SGLangProcessingError(f"Timeout after {self.max_retries} attempts: {str(e)}")
                
            except APIError as e:
                logger.error(f"[{self._service_name}] API error (attempt {retry_count + 1}/{self.max_retries}): {str(e)}")
                
                if hasattr(e, 'status_code') and 500 <= e.status_code < 600 and retry_count < self.max_retries - 1:
                    delay = self.retry_delay * (2 ** retry_count)
                    logger.info(f"[{self._service_name}] Retrying after {delay}s...")
                    await asyncio.sleep(delay)
                    return await self.process_single_image(image, retry_count + 1)
                
                raise SGLangProcessingError(f"API error: {str(e)}")
                
            except SGLangProcessingError as e:
                if retry_count < self.max_retries - 1:
                    delay = self.retry_delay * (2 ** retry_count)
                    logger.info(f"[{self._service_name}] Retrying after {delay}s...")
                    await asyncio.sleep(delay)
                    return await self.process_single_image(image, retry_count + 1)
                raise
                
            except Exception as e:
                logger.error(f"[{self._service_name}] Unexpected error: {type(e).__name__}: {str(e)}")
                logger.exception(f"[{self._service_name}] Full traceback:")
                raise SGLangProcessingError(f"Unexpected error: {type(e).__name__}: {str(e)}")
    
    async def process_images_async(self, images: List[Image.Image]) -> List[str]:
        """
        Process multiple images concurrently using SGLang.
        """
        if not images:
            return []
        
        batch_start = time.time()
        num_images = len(images)
        logger.info(f"[{self._service_name}] Starting batch processing of {num_images} images")
        
        # Process all images concurrently with semaphore control
        tasks = [self.process_single_image(img) for img in images]
        results = await asyncio.gather(*tasks, return_exceptions=False)
        
        batch_time = time.time() - batch_start
        avg_time = batch_time / num_images if num_images > 0 else 0
        logger.info(f"[{self._service_name}] Batch processing completed: {num_images} images in {batch_time:.2f}s (avg: {avg_time:.2f}s per image)")
        
        return results
    
    def process_images(self, images: List[Image.Image]) -> List[str]:
        """
        Synchronous wrapper for backward compatibility.
        """
        raise NotImplementedError(
            f"{self._service_name} is async-only. Use process_images_async() instead."
        )
