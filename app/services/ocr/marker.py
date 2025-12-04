import io
import tempfile
import os
import threading
import time
import gc
from typing import List, Optional
from PIL import Image
import img2pdf
import torch
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered
from marker.config.parser import ConfigParser
from surya.ocr import run_ocr
from surya.model.detection.model import load_model as load_det_model, load_processor as load_det_processor
from surya.model.recognition.model import load_model as load_rec_model
from surya.model.recognition.processor import load_processor as load_rec_processor
from .base import BaseOCRService


class MarkerOCRService(BaseOCRService):
    _service_name = "marker"
    _initialization_lock = threading.Lock()
    _worker_converters = {}  # Store converter per worker process

    def __init__(self):
        super().__init__()
        self.lock = threading.Lock()  # Lock to serialize PDF processing
        self._converter = None
        self._surya_models = None
        self._ensure_initialized()

    def _ensure_initialized(self):
        """Ensure each worker has its own models"""
        # Get current process ID
        pid = os.getpid()
        
        # Check if this worker already has models
        if pid in MarkerOCRService._worker_converters:
            stored_data = MarkerOCRService._worker_converters[pid]
            self._converter = stored_data.get('converter')
            self._surya_models = stored_data.get('surya_models')
            return
        
        with MarkerOCRService._initialization_lock:
            # Double-check after acquiring lock
            if pid in MarkerOCRService._worker_converters:
                stored_data = MarkerOCRService._worker_converters[pid]
                self._converter = stored_data.get('converter')
                self._surya_models = stored_data.get('surya_models')
                return
            
            # Initialize the models for this worker
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    # Prevent re-downloading if models exist
                    if os.path.exists('/root/.cache/datalab/models'):
                        os.environ['HF_HUB_OFFLINE'] = '1'
                    
                    # Add small delay between worker initializations to avoid conflicts
                    if len(MarkerOCRService._worker_converters) > 0:
                        time.sleep(0.5 * len(MarkerOCRService._worker_converters))
                    
                    # Load Surya OCR models for direct image processing
                    print(f"Loading Surya OCR models for worker {pid}...")
                    det_model = load_det_model()
                    det_processor = load_det_processor()
                    rec_model = load_rec_model()
                    rec_processor = load_rec_processor()
                    
                    surya_models = {
                        'det_model': det_model,
                        'det_processor': det_processor,
                        'rec_model': rec_model,
                        'rec_processor': rec_processor
                    }
                    
                    # Configure marker for PDF processing (when needed)
                    config = {
                        "force_ocr": True,
                        "strip_existing_ocr": True,
                    }
                    config_parser = ConfigParser(config)
                    config_dict = config_parser.generate_config_dict()
                    
                    converter = PdfConverter(
                        config=config_dict,
                        artifact_dict=create_model_dict(),
                    )
                    
                    # Store both models for this worker
                    MarkerOCRService._worker_converters[pid] = {
                        'converter': converter,
                        'surya_models': surya_models
                    }
                    self._converter = converter
                    self._surya_models = surya_models
                    
                    print(f"Marker OCR service with Surya models initialized for worker {pid} on attempt {attempt + 1}")
                    break
                    
                except Exception as e:
                    if attempt < max_retries - 1:
                        print(f"Failed to initialize Marker OCR for worker {pid} (attempt {attempt + 1}/{max_retries}): {e}")
                        time.sleep(2 ** attempt)
                        gc.collect()
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
        Process a list of PIL images and return extracted text using Surya OCR directly.
        
        Args:
            images: List of PIL Image objects to process
            
        Returns:
            List containing a single extracted text string for all images combined
        """
        if not images:
            return [""]

        try:
            # Ensure models are initialized
            if self._surya_models is None:
                self._ensure_initialized()
            
            # Convert images to RGB if needed
            processed_images = []
            for img in images:
                if img.mode not in ('RGB', 'L'):
                    img = img.convert('RGB')
                processed_images.append(img)
            
            # Use Surya OCR directly on images
            langs = ["en"]  # Default to English, can be made configurable
            
            with self.lock:
                try:
                    predictions = run_ocr(
                        processed_images,
                        [langs] * len(processed_images),
                        self._surya_models['det_model'],
                        self._surya_models['det_processor'],
                        self._surya_models['rec_model'],
                        self._surya_models['rec_processor']
                    )
                    
                    # Extract text from all pages
                    all_text = []
                    for page_pred in predictions:
                        page_text = []
                        for text_line in page_pred.text_lines:
                            page_text.append(text_line.text)
                        all_text.append('\n'.join(page_text))
                    
                    # Combine all pages with double newline separator
                    combined_text = '\n\n'.join(all_text)
                    
                    # Clean up
                    del predictions
                    del processed_images
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
                    gc.collect()
                    
                    return [combined_text]
                    
                except Exception as e:
                    print(f"Error in Surya OCR processing: {e}")
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
                    gc.collect()
                    return [""]

        except Exception as e:
            print(f"Error in process_images: {e}")
            gc.collect()
            return [""]