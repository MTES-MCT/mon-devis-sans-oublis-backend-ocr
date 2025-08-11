import os
if "LOCAL_RANK" not in os.environ:
    os.environ["LOCAL_RANK"] = "0"

import torch
from transformers import AutoModelForCausalLM, AutoProcessor
from PIL import Image
from typing import List
from .base import BaseOCRService

# Import the required utilities from the dots_ocr package
try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    # If qwen_vl_utils is not available, we'll need to handle this
    process_vision_info = None

class DotsOCRService(BaseOCRService):
    _service_name = "dotsocr"

    def __init__(self):
        # Try to load the model with flash attention for better performance
        # Fall back to standard attention if flash attention is not available
        model_kwargs = {
            "torch_dtype": torch.bfloat16,
            "device_map": "auto",
            "trust_remote_code": True
        }
        
        try:
            # Try with flash attention first
            self.model = AutoModelForCausalLM.from_pretrained(
                "rednote-hilab/dots.ocr",
                attn_implementation="flash_attention_2",
                **model_kwargs
            )
            print("DotsOCR: Using flash attention for better performance")
        except Exception as e:
            # Fall back to standard attention
            print(f"DotsOCR: Flash attention not available ({e}), using standard attention")
            self.model = AutoModelForCausalLM.from_pretrained(
                "rednote-hilab/dots.ocr",
                **model_kwargs
            )
        self.processor = AutoProcessor.from_pretrained(
            "rednote-hilab/dots.ocr",
            trust_remote_code=True
        )
        
        # Default prompt for OCR extraction
        self.default_prompt = "Extract all text from this document image, preserving the original reading order and layout structure. Return the plain text representation."

    def process_images(self, images: List[Image.Image]) -> List[str]:
        """
        Process a list of PIL images and return a list of extracted text.
        """
        results = []
        
        for image in images:
            try:
                # Save image temporarily if needed (the model expects a path)
                # In production, you might want to use a temporary file or pass the image directly
                import tempfile
                with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
                    image.save(tmp_file.name)
                    image_path = tmp_file.name
                
                # Prepare the message format expected by the model
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "image": image_path
                            },
                            {
                                "type": "text",
                                "text": self.default_prompt
                            }
                        ]
                    }
                ]
                
                # Apply chat template
                text = self.processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                
                # Process vision information if the utility is available
                if process_vision_info:
                    image_inputs, video_inputs = process_vision_info(messages)
                else:
                    # Fallback: process the image directly
                    image_inputs = [image]
                    video_inputs = None
                
                # Prepare inputs for the model
                inputs = self.processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs if video_inputs else None,
                    padding=True,
                    return_tensors="pt",
                )
                
                # Move to CUDA if available
                if torch.cuda.is_available():
                    inputs = inputs.to("cuda")
                
                # Generate output with a reasonable token limit for OCR
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=8096  # Adjust based on your needs
                )
                
                # Trim the generated IDs to get only the new tokens
                generated_ids_trimmed = [
                    out_ids[len(in_ids):] 
                    for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                
                # Decode the output
                output_text = self.processor.batch_decode(
                    generated_ids_trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False
                )
                
                # Clean up temporary file
                if 'image_path' in locals():
                    try:
                        os.unlink(image_path)
                    except:
                        pass
                
                # Extract the text from the output
                if output_text and isinstance(output_text, list) and len(output_text) > 0:
                    extracted_text = output_text[0]
                    results.append(extracted_text)
                else:
                    results.append("")
                    
            except Exception as e:
                print(f"Error processing image with DotsOCR: {str(e)}")
                results.append("")  # Append empty string on error
        
        return results