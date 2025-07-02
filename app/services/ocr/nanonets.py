from fastapi.concurrency import run_in_threadpool
from PIL import Image
from transformers import AutoTokenizer, AutoProcessor, AutoModelForImageTextToText
import torch
from PyPDF2 import PdfReader
import io
import fitz  # PyMuPDF
import os
from .base import BaseOCRService

class NanonetsOCRService(BaseOCRService):
    def __init__(self):
        model_path = "nanonets/Nanonets-OCR-s"

        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            torch_dtype=torch.float16,  # Use float16 for mixed precision
            device_map="auto",
            trust_remote_code=True,
        )
        self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.processor = AutoProcessor.from_pretrained(model_path)

    async def process_file(self, file_path: str, filename: str):
        if filename.lower().endswith(".pdf"):
            return await run_in_threadpool(self.process_pdf, file_path)
        else:
            return await run_in_threadpool(self.process_image, file_path)

    def process_pdf(self, file_path: str):
        text = ""
        doc = fitz.open(file_path)
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            pix = page.get_pixmap()
            img_bytes = pix.tobytes("ppm")
            image = Image.open(io.BytesIO(img_bytes))

            # Create a temporary image file path
            temp_image_path = f"temp_page_{page_num}.png"
            image.save(temp_image_path)

            text += self.ocr_page_with_nanonets_s(temp_image_path)

            # Clean up the temporary image file
            os.remove(temp_image_path)

        return text

    def process_image(self, file_path: str):
        return self.ocr_page_with_nanonets_s(file_path)

    def ocr_page_with_nanonets_s(self, image_path, max_new_tokens=4096):
        prompt = """Extract the text from the above document as if you were reading it naturally. Return the tables in html format. Return the equations in LaTeX representation. If there is an image in the document and image caption is not present, add a small description of the image inside the <img></img> tag; otherwise, add the image caption inside <img></img>. Watermarks should be wrapped in brackets. Ex: <watermark>OFFICIAL COPY</watermark>. Page numbers should be wrapped in brackets. Ex: <page_number>14</page_number> or <page_number>9/22</page_number>. Prefer using ☐ and ☑ for check boxes."""
        image = Image.open(image_path)
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ]},
        ]
        text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[image], padding=True, return_tensors="pt")
        inputs = inputs.to(self.model.device)

        output_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]

        output_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        return output_text[0]

ocr_service = NanonetsOCRService()