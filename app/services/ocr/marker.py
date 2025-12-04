import io
import tempfile
import os
import threading
import time
import gc
from typing import List, Optional
from PIL import Image
import img2pdf
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered
from marker.config.parser import ConfigParser
from .base import BaseOCRService


class MarkerOCRService(BaseOCRService):
    _service_name = "marker"
    _initialization_lock = threading.Lock()
    _worker_converters = {}  # Store converter per worker process

    def __init__(self):
        super().__init__()
        self.lock = threading.Lock()  # Lock to serialize PDF processing
        self._converter = None
        self._ensure_initialized()

    def _ensure_initialized(self):
        """Ensure each worker has its own converter instance"""
        # Get current process ID
        pid = os.getpid()
        
        # Check if this worker already has a converter
        if pid in MarkerOCRService._worker_converters:
            self._converter = MarkerOCRService._worker_converters[pid]
            return
        
        with MarkerOCRService._initialization_lock:
            # Double-check after acquiring lock
            if pid in MarkerOCRService._worker_converters:
                self._converter = MarkerOCRService._worker_converters[pid]
                return
            
            # Initialize the marker converter for this worker
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    # Prevent re-downloading if models exist
                    if os.path.exists('/root/.cache/datalab/models'):
                        os.environ['HF_HUB_OFFLINE'] = '1'
                    
                    # Add small delay between worker initializations to avoid conflicts
                    if len(MarkerOCRService._worker_converters) > 0:
                        time.sleep(0.5 * len(MarkerOCRService._worker_converters))
                    
                    # Configure marker to force OCR on all pages
                    config = {
                        "force_ocr": True,  # Force OCR on all pages
                        "strip_existing_ocr": True,  # Remove existing OCR artifacts and re-OCR
                        "disable_image_extraction": True,  # Disable image extraction to force OCR on images
                    }
                    config_parser = ConfigParser(config)
                    config_dict = config_parser.generate_config_dict()
                    
                    # Create converter with force_ocr enabled
                    converter = PdfConverter(
                        config=config_dict,
                        artifact_dict=create_model_dict(),
                    )
                    
                    # Store converter for this worker
                    MarkerOCRService._worker_converters[pid] = converter
                    self._converter = converter
                    
                    print(f"Marker OCR service initialized for worker {pid} on attempt {attempt + 1}")
                    break
                    
                except Exception as e:
                    if attempt < max_retries - 1:
                        print(f"Failed to initialize Marker OCR for worker {pid} (attempt {attempt + 1}/{max_retries}): {e}")
                        time.sleep(2 ** attempt)  # Exponential backoff
                        gc.collect()  # Force garbage collection
                    else:
                        raise RuntimeError(
                            f"Failed to initialize Marker OCR service for worker {pid} after {max_retries} attempts: {e}"
                        )
    
    @property
    def converter(self):
        """Get the converter for this worker"""
        if self._converter is None:
            self._ensure_initialized()
        return self._converter

    def process_pdf_file(self, pdf_path: str) -> str:
        """
        Process a PDF file directly with marker.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Extracted text from the PDF
        """
        with self.lock:  # Serialize access to prevent concurrent PDF processing
            try:
                rendered = self.converter(pdf_path)
                text, _, _ = text_from_rendered(rendered)
                
                # Clean up to prevent memory leaks
                del rendered
                gc.collect()
                
                return text
            except Exception as e:
                print(f"Error processing PDF with Marker: {e}")
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
                
                # Force garbage collection to free memory
                gc.collect()

        except Exception as e:
            print(f"Error in process_images: {e}")
            gc.collect()
            return [""]