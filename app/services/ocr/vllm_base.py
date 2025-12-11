import base64
import io
import asyncio
import logging
import time
from typing import List, Optional
from abc import abstractmethod
from PIL import Image
from openai import AsyncOpenAI, APIError, APITimeoutError
from .base import BaseOCRService
from app.config import config

logger = logging.getLogger(__name__)


class VLLMProcessingError(Exception):
    """Exception raised when VLLM processing fails"""
    pass


class VLLMOCRService(BaseOCRService):
    """
    Base class for VLLM-based OCR services.
    
    This class provides common functionality for services that use VLLM
    with OpenAI-compatible API for vision-based OCR.
    
    Subclasses should set:
    - _service_name: Unique identifier for the service
    - _model_name: VLLM model name
    - _endpoint: VLLM API endpoint URL
    - _system_prompt: Text prompt for OCR (default: "Extract all text from this image.")
    - _extra_body: Optional dict with VLLM-specific parameters (default: None)
    """
    
    _service_name = "vllm_base"
    _endpoint: Optional[str] = None
    _system_prompt: str = "Extract all text from this image."
    _model_name: str = "default"
    _extra_body: Optional[dict] = None  # Subclasses can override with model-specific config
    
    def __init__(self):
        """Initialize VLLM service with OpenAI client"""
        super().__init__()
        
        if not self._endpoint:
            raise ValueError(f"Endpoint not configured for {self._service_name}")
        
        # Initialize OpenAI client pointing to VLLM server
        self.client = AsyncOpenAI(
            base_url=self._endpoint,
            api_key="EMPTY",  # VLLM doesn't require API key
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
        
        Args:
            image: PIL Image object
            
        Returns:
            Base64 encoded image string
        """
        buffered = io.BytesIO()
        
        # Convert to RGB if necessary
        if image.mode not in ('RGB', 'L'):
            image = image.convert('RGB')
        
        # Save as PNG (DeepSeek might not handle JPEG well)
        image.save(buffered, format="PNG")
        buffered.seek(0)
        
        # Encode to base64
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
        return f"data:image/png;base64,{img_str}"
    
    async def process_single_image(self, image: Image.Image, retry_count: int = 0) -> str:
        """
        Process a single image asynchronously using VLLM with retry logic.
        
        Args:
            image: PIL Image object
            retry_count: Current retry attempt (internal use)
            
        Returns:
            Extracted text from the image
            
        Raises:
            VLLMProcessingError: If processing fails after all retries
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
                logger.debug(f"[{self._service_name}] Image encoded to base64 (size: {len(base64_image)} chars)")
                
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
                
                logger.info(f"[{self._service_name}] Sending request to VLLM endpoint: {self._endpoint}")
                logger.debug(f"[{self._service_name}] System prompt: {self._system_prompt}")
                logger.debug(f"[{self._service_name}] Image base64 prefix: {base64_image[:100]}...")
                logger.debug(f"[{self._service_name}] Request: model={self._model_name}, max_tokens={config.VLLM_MAX_TOKENS}, temperature=0")
                logger.debug(f"[{self._service_name}] Message structure: {messages[0]['content'][0]['type']}, text: {messages[0]['content'][1]['text']}")
                
                # Call VLLM via OpenAI API
                # Use max_tokens=2096 to match successful test
                request_params = {
                    "model": self._model_name,
                    "messages": messages,
                    "max_tokens": 2096,
                    "temperature": 0.0,  # Deterministic output for OCR
                }
                
                # Add extra_body if provided by subclass
                if self._extra_body:
                    request_params["extra_body"] = self._extra_body
                    logger.debug(f"[{self._service_name}] Using extra_body: {self._extra_body}")
                
                # Time the VLLM API call
                vllm_start = time.time()
                response = await self.client.chat.completions.create(**request_params)
                vllm_time = time.time() - vllm_start
                logger.info(f"[{self._service_name}] VLLM API call took {vllm_time:.2f}s")
                
                # Log full response for debugging
                logger.debug(f"[{self._service_name}] Full response object: {response}")
                logger.debug(f"[{self._service_name}] Response model: {response.model}")
                logger.debug(f"[{self._service_name}] Choices count: {len(response.choices)}")
                
                if response.choices and len(response.choices) > 0:
                    choice = response.choices[0]
                    logger.debug(f"[{self._service_name}] Finish reason: {choice.finish_reason}")
                    logger.debug(f"[{self._service_name}] Message: {choice.message}")
                    logger.debug(f"[{self._service_name}] Content type: {type(choice.message.content)}")
                    logger.debug(f"[{self._service_name}] Content value: {repr(choice.message.content)}")
                
                # Extract text from response
                text = response.choices[0].message.content
                
                if not text:
                    logger.warning(f"[{self._service_name}] Received empty response from VLLM")
                    logger.warning(f"[{self._service_name}] Full response: {response.model_dump_json()}")
                    # If we get an empty response, raise an error to trigger retry
                    raise VLLMProcessingError("Empty response from VLLM")
                
                total_time = time.time() - image_start
                logger.info(f"[{self._service_name}] Successfully processed image in {total_time:.2f}s (VLLM: {vllm_time:.2f}s, encoding: {encode_time:.3f}s)")
                logger.info(f"[{self._service_name}] Response length: {len(text)} chars")
                logger.debug(f"[{self._service_name}] Response text preview: {text[:200]}...")
                
                return text
                
            except APITimeoutError as e:
                logger.error(f"[{self._service_name}] Timeout error (attempt {retry_count + 1}/{self.max_retries}): {str(e)}")
                
                if retry_count < self.max_retries - 1:
                    delay = self.retry_delay * (2 ** retry_count)  # Exponential backoff
                    logger.info(f"[{self._service_name}] Retrying after {delay}s...")
                    await asyncio.sleep(delay)
                    return await self.process_single_image(image, retry_count + 1)
                
                raise VLLMProcessingError(f"Timeout after {self.max_retries} attempts: {str(e)}")
                
            except APIError as e:
                logger.error(f"[{self._service_name}] API error (attempt {retry_count + 1}/{self.max_retries}): {str(e)}")
                
                # Only retry on 5xx errors (server errors)
                if hasattr(e, 'status_code') and 500 <= e.status_code < 600 and retry_count < self.max_retries - 1:
                    delay = self.retry_delay * (2 ** retry_count)
                    logger.info(f"[{self._service_name}] Retrying after {delay}s...")
                    await asyncio.sleep(delay)
                    return await self.process_single_image(image, retry_count + 1)
                
                raise VLLMProcessingError(f"API error: {str(e)}")
                
            except VLLMProcessingError as e:
                # Already a VLLMProcessingError, check if we should retry
                if retry_count < self.max_retries - 1:
                    delay = self.retry_delay * (2 ** retry_count)
                    logger.info(f"[{self._service_name}] Retrying after {delay}s...")
                    await asyncio.sleep(delay)
                    return await self.process_single_image(image, retry_count + 1)
                
                # Re-raise the error after all retries exhausted
                raise
                
            except Exception as e:
                logger.error(f"[{self._service_name}] Unexpected error processing image: {type(e).__name__}: {str(e)}")
                logger.exception(f"[{self._service_name}] Full traceback:")
                raise VLLMProcessingError(f"Unexpected error: {type(e).__name__}: {str(e)}")
    
    async def process_images_async(self, images: List[Image.Image]) -> List[str]:
        """
        Process multiple images concurrently using VLLM.
        
        Args:
            images: List of PIL Image objects
            
        Returns:
            List of extracted text strings, one per image
            
        Raises:
            VLLMProcessingError: If any image fails to process after retries
        """
        if not images:
            return []
        
        batch_start = time.time()
        num_images = len(images)
        logger.info(f"[{self._service_name}] Starting batch processing of {num_images} images")
        
        # Process all images concurrently with semaphore control
        tasks = [self.process_single_image(img) for img in images]
        results = await asyncio.gather(*tasks, return_exceptions=False)  # Don't catch exceptions
        
        batch_time = time.time() - batch_start
        avg_time = batch_time / num_images if num_images > 0 else 0
        logger.info(f"[{self._service_name}] Batch processing completed: {num_images} images in {batch_time:.2f}s (avg: {avg_time:.2f}s per image)")
        
        return results
    
    def process_images(self, images: List[Image.Image]) -> List[str]:
        """
        Synchronous wrapper for backward compatibility.
        
        This should not be called directly - use process_images_async instead.
        Only implemented to satisfy base class interface.
        """
        raise NotImplementedError(
            f"{self._service_name} is async-only. Use process_images_async() instead."
        )