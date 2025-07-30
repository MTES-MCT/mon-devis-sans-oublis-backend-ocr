import io
import tempfile
import os
from typing import List, Union
from PIL import Image
import img2pdf
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered
from .base import BaseOCRService


class MarkerOCRService(BaseOCRService):
    _service_name = "marker"

    def __init__(self):
        super().__init__()
        # Initialize the marker converter
        try:
            self.converter = PdfConverter(
                artifact_dict=create_model_dict(),
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize Marker OCR service: {e}. "
                "Please install with: pip install marker-pdf[full]"
            )

    def process_pdf_file(self, pdf_path: str) -> str:
        """
        Process a PDF file directly with marker.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Extracted text from the PDF
        """
        try:
            rendered = self.converter(pdf_path)
            text, _, _ = text_from_rendered(rendered)
            return text
        except Exception as e:
            return ""

    def process_images(self, images: List[Image.Image]) -> List[str]:
        """
        Process a list of PIL images and return extracted text using marker OCR.
        
        Args:
            images: List of PIL Image objects to process
            
        Returns:
            List containing a single extracted text string for all images combined
        """
        if not images:
            return [""]
        
        try:
            # Convert PIL images to bytes first
            image_bytes_list = []
            for i, img in enumerate(images):
                try:
                    # Convert PIL Image to bytes
                    img_byte_arr = io.BytesIO()
                    # Ensure image is in RGB mode for PDF
                    if img.mode not in ('RGB', 'L'):
                        img = img.convert('RGB')
                    # Save as JPEG for better compression
                    img.save(img_byte_arr, format='JPEG', quality=95)
                    img_byte_arr.seek(0)
                    img_bytes = img_byte_arr.getvalue()
                    image_bytes_list.append(img_bytes)
                except Exception as e:
                    # Try PNG format as fallback
                    img_byte_arr = io.BytesIO()
                    img.save(img_byte_arr, format='PNG')
                    img_byte_arr.seek(0)
                    image_bytes_list.append(img_byte_arr.getvalue())
            
            if not image_bytes_list:
                return [""]
            
            # Convert all image bytes to a single PDF
            pdf_bytes = img2pdf.convert(image_bytes_list)
            
            # Create temporary file for the combined PDF
            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as pdf_file:
                pdf_file.write(pdf_bytes)
                pdf_path = pdf_file.name
            
            try:
                # Process the combined PDF with marker
                text = self.process_pdf_file(pdf_path)
                return [text]
                
            finally:
                # Clean up temporary PDF file
                try:
                    os.unlink(pdf_path)
                except OSError:
                    pass
        
        except Exception as e:
            # Handle errors during PDF creation
            return [""]