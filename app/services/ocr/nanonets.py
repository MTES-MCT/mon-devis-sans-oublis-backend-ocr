import torch
import gc
from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image
from typing import List
from .base import BaseOCRService

class NanonetsOCRService(BaseOCRService):
    _service_name = "nanonets"

    def __init__(self):
        seed = 42
        torch.Generator().manual_seed(seed)
        self.processor = AutoProcessor.from_pretrained("nanonets/Nanonets-OCR-s")
        self.model = AutoModelForImageTextToText.from_pretrained(
            "nanonets/Nanonets-OCR-s",
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="flash_attention_2"
        )
        # Store device for memory management
        self.device = next(self.model.parameters()).device

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