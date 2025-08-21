import torch
import gc
from transformers import AutoProcessor, AutoModelForImageTextToText, pipeline
from PIL import Image
from typing import List
from .base import BaseOCRService

class NanonetsOCRService(BaseOCRService):
    _service_name = "nanonets"

    def __init__(self):
        seed = 42
        torch.Generator().manual_seed(seed)
        processor = AutoProcessor.from_pretrained("nanonets/Nanonets-OCR-s")
        model = AutoModelForImageTextToText.from_pretrained(
            "nanonets/Nanonets-OCR-s", torch_dtype=torch.bfloat16, device_map="auto"
        )
        self.pipeline = pipeline(
            "image-text-to-text",
            model=model,
            processor=processor,
        )
        # Store device for memory management
        self.device = next(model.parameters()).device

    def process_images(self, images: List[Image.Image]) -> List[str]:
        prompt = "Extract and return all the text from this image. Include all text elements and maintain the reading order and line breaks. If there are tables, convert them to markdown format while including line breaks in the cells using <br/> tag. If there are mathematical equations, convert them to LaTeX format. Escape characters if necessary. Cells might contain long text, do not create new cells on your own."
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
                                "text": prompt,
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
                                results.append(assistant_messages[-1])
                                continue
                        elif isinstance(generated_content, str):
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