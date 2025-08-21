import io
import tempfile
import os
import gc
import torch
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
            # Process with memory management
            rendered = self.converter(pdf_path)
            text, _, _ = text_from_rendered(rendered)
            
            # Clear any GPU memory if marker uses CUDA
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Clean up rendered object
            del rendered
            gc.collect()
            
            return text
        except Exception as e:
            print(f"Error processing PDF with marker: {e}")
            # Ensure cleanup on error
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
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
        
        pdf_path = None
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
                    
                    # Clean up the BytesIO object
                    img_byte_arr.close()
                    del img_byte_arr
                    
                except Exception as e:
                    print(f"Error converting image {i}: {e}, trying PNG format")
                    # Try PNG format as fallback
                    img_byte_arr = io.BytesIO()
                    img.save(img_byte_arr, format='PNG')
                    img_byte_arr.seek(0)
                    image_bytes_list.append(img_byte_arr.getvalue())
                    img_byte_arr.close()
                    del img_byte_arr
            
            if not image_bytes_list:
                return [""]
            
            # Convert all image bytes to a single PDF
            pdf_bytes = img2pdf.convert(image_bytes_list)
            
            # Clear image bytes list to free memory
            del image_bytes_list
            gc.collect()
            
            # Create temporary file for the combined PDF
            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as pdf_file:
                pdf_file.write(pdf_bytes)
                pdf_path = pdf_file.name
            
            # Clear pdf_bytes to free memory
            del pdf_bytes
            gc.collect()
            
            # Process the combined PDF with marker
            text = self.process_pdf_file(pdf_path)
            return [text]
                
        except Exception as e:
            print(f"Error in marker process_images: {e}")
            # Handle errors during PDF creation
            return [""]
            
        finally:
            # Clean up temporary PDF file
            if pdf_path and os.path.exists(pdf_path):
                try:
                    os.unlink(pdf_path)
                except OSError as e:
                    print(f"Error removing temporary file {pdf_path}: {e}")
            
            # Final memory cleanup
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()