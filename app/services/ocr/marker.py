import io
import tempfile
import os
from typing import List
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
            # Convert all PIL images to a single PDF bytes using img2pdf
            pdf_bytes = img2pdf.convert(images)
            
            # Create temporary file for the combined PDF
            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as pdf_file:
                pdf_file.write(pdf_bytes)
                pdf_path = pdf_file.name
            
            try:
                # Process the combined PDF with marker
                rendered = self.converter(pdf_path)
                text, _, _ = text_from_rendered(rendered)
                
                # Return as a single result
                return [text]
                
            except Exception as e:
                # Handle any errors during processing
                print(f"Error processing images with marker: {e}")
                return [""]
            finally:
                # Clean up temporary PDF file
                try:
                    os.unlink(pdf_path)
                except OSError:
                    pass
        
        except Exception as e:
            # Handle errors during PDF creation
            print(f"Error converting images to PDF: {e}")
            return [""]