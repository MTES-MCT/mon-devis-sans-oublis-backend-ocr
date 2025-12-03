import torch
import gc
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from PIL import Image
from typing import List
import json
from .base import BaseOCRService

class OlmOCRService(BaseOCRService):
    _service_name = "olmocr"

    def __init__(self):
        self.processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            "allenai/olmOCR-7B-0225-preview",
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="flash_attention_2"
        )
        # Store device for memory management
        self.device = next(self.model.parameters()).device

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
                    # Process directly with model and processor to avoid multiprocessing issues
                    text = self.processor.apply_chat_template(
                        messages, add_generation_prompt=True
                    )
                    inputs = self.processor(
                        text=[text],
                        images=[image],
                        return_tensors="pt",
                        padding=True
                    ).to(self.device)
                    
                    # Generate with explicit memory management
                    with torch.cuda.amp.autocast(enabled=True):
                        with torch.no_grad():
                            generated_ids = self.model.generate(
                                **inputs,
                                max_new_tokens=8096,
                                do_sample=False
                            )
                    
                    # Decode the output
                    generated_text = self.processor.batch_decode(
                        generated_ids[:, inputs.input_ids.shape[1]:],
                        skip_special_tokens=True
                    )[0]
                    
                    # Try to parse JSON response
                    try:
                        json_data = json.loads(generated_text)
                        results.append(json_data.get("natural_text", generated_text))
                    except json.JSONDecodeError:
                        results.append(generated_text if generated_text else "")
                    
                    # Clean up inputs immediately
                    del inputs, generated_ids
                    
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