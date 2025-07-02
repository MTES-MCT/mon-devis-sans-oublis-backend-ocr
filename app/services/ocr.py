import torch
from transformers import AutoProcessor, AutoModelForImageTextToText, pipeline, Qwen2VLForConditionalGeneration
from PIL import Image
import json
import io
import fitz
import os
from fastapi.concurrency import run_in_threadpool


class OCRFactory:
    def __init__(self):
        self.models = {}
        self.processors = {}
        self.pipelines = {}
        self._load_models()

    def _load_models(self):
        # Load Nanonets-OCR-s
        try:
            self.processors["nanonets"] = AutoProcessor.from_pretrained(
                "nanonets/Nanonets-OCR-s"
            )
            self.models["nanonets"] = AutoModelForImageTextToText.from_pretrained(
                "nanonets/Nanonets-OCR-s", torch_dtype=torch.bfloat16, device_map="auto"
            )
            self.pipelines["nanonets"] = pipeline(
                "image-text-to-text",
                model=self.models["nanonets"],
                processor=self.processors["nanonets"],
            )
        except Exception as e:
            print(f"Error loading Nanonets-OCR-s: {e}")

        # Load olmOCR
        try:
            self.processors["olmocr"] = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
            self.models["olmocr"] = Qwen2VLForConditionalGeneration.from_pretrained(
                "allenai/olmOCR-7B-0225-preview", torch_dtype=torch.bfloat16, device_map="auto"
            )
            self.pipelines["olmocr"] = pipeline(
                "image-text-to-text", model=self.models["olmocr"], processor=self.processors["olmocr"]
            )
        except Exception as e:
            print(f"Error loading olmOCR: {e}")
            
    def get_pipeline(self, model_name: str):
        return self.pipelines.get(model_name)

    async def process_file(self, file_path: str, filename: str, model_name: str):
        if filename.lower().endswith(".pdf"):
            return await run_in_threadpool(self.process_pdf, file_path, model_name)
        else:
            return await run_in_threadpool(self.run_hf_ocr, file_path, model_name)
            
    def process_pdf(self, file_path: str, model_name: str):
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
            
            text += self.run_hf_ocr(temp_image_path, model_name)
            
            # Clean up the temporary image file
            os.remove(temp_image_path)

        return text

    def run_hf_ocr(self, image_path, model_name="nanonets"):
        if image_path is None:
            return "No image provided for OCR."

        try:
            pil_image = Image.open(image_path).convert("RGB")
            
            selected_pipe = self.get_pipeline(model_name)
            if not selected_pipe:
                raise RuntimeError(f"Model {model_name} could not be initialized or is not available.")

            if model_name == "nanonets":
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": pil_image},
                            {
                                "type": "text",
                                "text": "Extract and return all the text from this image. Include all text elements and maintain the reading order. If there are tables, convert them to markdown format. If there are mathematical equations, convert them to LaTeX format.",
                            },
                        ],
                    }
                ]
            else:  # olmOCR
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": pil_image},
                            {
                                "type": "text",
                                "text": "Extract all text from this document image, preserving the original reading order and layout structure. Return the plain text representation.",
                            },
                        ],
                    }
                ]
            max_tokens = 8096
            ocr_results = selected_pipe(messages, max_new_tokens=max_tokens)
            
            if not isinstance(ocr_results, list) or not ocr_results:
                return "Error: OCR model did not return expected output."

            generated_content = ocr_results[0].get("generated_text")
            if not generated_content:
                 return "Error: Could not find 'generated_text' in the model output."
            
            # The output of the pipeline is a list of conversations.
            # We want the last message from the assistant.
            if isinstance(generated_content, list):
                assistant_messages = [
                    message["content"]
                    for message in generated_content
                    if message.get("role") == "assistant"
                ]
                if assistant_messages:
                    last_message = assistant_messages[-1]
                    if model_name == "olmocr" and isinstance(last_message, str):
                        try:
                            json_data = json.loads(last_message)
                            return json_data.get("natural_text", last_message)
                        except json.JSONDecodeError:
                            return last_message # Return as is if not valid JSON
                    return last_message

            # Fallback for simple string output
            if isinstance(generated_content, str):
                 if model_name == "olmocr":
                    try:
                        json_data = json.loads(generated_content)
                        return json_data.get("natural_text", generated_content)
                    except json.JSONDecodeError:
                         return generated_content
                 return generated_content

            return "Error: Could not parse assistant's response from model output."

        except Exception as e:
            print(f"Error during Hugging Face OCR processing: {e}")
            return f"Error during Hugging Face OCR: {str(e)}"

ocr_factory = OCRFactory()