"""
Optimized marker service that can process PDFs directly
"""
import tempfile
import os
from typing import Union
from fastapi import UploadFile
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered


class MarkerDirectService:
    """Direct PDF processing service for marker"""
    
    def __init__(self):
        self.converter = PdfConverter(
            artifact_dict=create_model_dict(),
        )
    
    def process_pdf_upload(self, file: UploadFile) -> str:
        """
        Process an uploaded PDF file directly with marker.
        
        Args:
            file: UploadFile object containing PDF
            
        Returns:
            Extracted text from the PDF
        """
        # Save uploaded file to temporary location
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
            tmp_file.write(file.file.read())
            tmp_path = tmp_file.name
        
        try:
            # Process with marker
            rendered = self.converter(tmp_path)
            text, _, _ = text_from_rendered(rendered)
            return text
        except Exception as e:
            print(f"Error processing PDF with marker: {e}")
            import traceback
            traceback.print_exc()
            return ""
        finally:
            # Clean up
            try:
                os.unlink(tmp_path)
            except OSError:
                pass