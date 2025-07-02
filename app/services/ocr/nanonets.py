import torch
from transformers import AutoProcessor, AutoModelForImageTextToText, pipeline
from PIL import Image
from typing import List
from .base import BaseOCRService

class NanonetsOCRService(BaseOCRService):
    _service_name = "nanonets"

    def __init__(self):
        processor = AutoProcessor.from_pretrained("nanonets/Nanonets-OCR-s")
        model = AutoModelForImageTextToText.from_pretrained(
            "nanonets/Nanonets-OCR-s", torch_dtype=torch.bfloat16, device_map="auto"
        )
        self.pipeline = pipeline(
            "image-text-to-text",
            model=model,
            processor=processor,
        )

    def process_images(self, images: List[Image.Image]) -> List[str]:
        results = []
        for image in images:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {
                            "type": "text",
                            "text": "Extract and return all the text from this image. Include all text elements and maintain the reading order. If there are tables, convert them to markdown format. If there are mathematical equations, convert them to LaTeX format.",
                        },
                    ],
                }
            ]
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
        return results