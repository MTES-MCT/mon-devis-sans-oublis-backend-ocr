import torch
import gc
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration, pipeline
from PIL import Image
from typing import List
import json
from .base import BaseOCRService

class OlmOCRService(BaseOCRService):
    _service_name = "olmocr"

    def __init__(self):
        processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            "allenai/olmOCR-7B-0225-preview", torch_dtype=torch.bfloat16, device_map="auto"
        )
        self.pipeline = pipeline(
            "image-text-to-text", model=model, processor=processor
        )
        # Store device for memory management
        self.device = next(model.parameters()).device

    def process_images(self, images: List[Image.Image]) -> List[str]:
        results = []
        
        try:
            for image in images:
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image},
                            {
                                "type": "text",
                                "text": "Extract all text from this document image, preserving the original reading order and layout structure. Return the plain text representation.",
                            },
                        ],
                    }
                ]
                
                try:
                    # Process with explicit memory management
                    with torch.cuda.amp.autocast(enabled=True):
                        ocr_result = self.pipeline(messages, max_new_tokens=8096)
                    
                    if ocr_result and isinstance(ocr_result, list) and "generated_text" in ocr_result[0]:
                        generated_content = ocr_result[0]["generated_text"]
                        if isinstance(generated_content, list):
                            assistant_messages = [
                                message["content"]
                                for message in generated_content
                                if message.get("role") == "assistant"
                            ]
                            if assistant_messages:
                                last_message = assistant_messages[-1]
                                try:
                                    json_data = json.loads(last_message)
                                    results.append(json_data.get("natural_text", last_message))
                                except json.JSONDecodeError:
                                    results.append(last_message)
                                continue
                        elif isinstance(generated_content, str):
                            try:
                                json_data = json.loads(generated_content)
                                results.append(json_data.get("natural_text", generated_content))
                            except json.JSONDecodeError:
                                results.append(generated_content)
                            continue
                    results.append("") # Append empty string if no result
                    
                except torch.cuda.OutOfMemoryError:
                    # Clear GPU memory and retry
                    torch.cuda.empty_cache()
                    gc.collect()
                    results.append("")  # Skip this image if OOM persists
                    continue
                    
                except Exception as e:
                    print(f"Error processing image: {e}")
                    results.append("")
                    continue
                
                finally:
                    # Clear cache after each image to prevent memory accumulation
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        
        finally:
            # Final cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
        return results