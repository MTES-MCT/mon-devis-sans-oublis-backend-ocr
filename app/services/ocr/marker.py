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
            self._surya_models = stored_data.get('artifact_dict')
            return
        
        with MarkerOCRService._initialization_lock:
            # Double-check after acquiring lock
            if pid in MarkerOCRService._worker_converters:
                stored_data = MarkerOCRService._worker_converters[pid]
                self._converter = stored_data.get('converter')
                self._surya_models = stored_data.get('artifact_dict')
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
                    
                    # Load marker's model dict (includes Surya OCR models)
                    print(f"Loading marker models (including Surya OCR) for worker {pid}...")
                    artifact_dict = create_model_dict()
                    
                    # Configure marker for full OCR
                    config = {
                        "force_ocr": True,
                        "strip_existing_ocr": True,
                    }
                    config_parser = ConfigParser(config)
                    config_dict = config_parser.generate_config_dict()
                    
                    converter = PdfConverter(
                        config=config_dict,
                        artifact_dict=artifact_dict,
                    )
                    
                    # Store both for this worker
                    MarkerOCRService._worker_converters[pid] = {
                        'converter': converter,
                        'artifact_dict': artifact_dict
                    }
                    self._converter = converter
                    self._surya_models = artifact_dict
                    
                    print(f"Marker OCR service initialized for worker {pid} on attempt {attempt + 1}")
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
        Process a list of PIL images and return extracted text by using marker's OCR on image-based PDFs.
        
        Args:
            images: List of PIL Image objects to process
            
        Returns:
            List containing a single extracted text string for all images combined
        """
        if not images:
            return [""]

        try:
            # Use marker's processors directly from artifact_dict
            if self._surya_models is None:
                self._ensure_initialized()
            
            # Get OCR models from artifact_dict
            ocr_recognizer = self._surya_models.get('ocr_recognizer')
            ocr_rec_processor = self._surya_models.get('ocr_rec_processor')
            ocr_detector = self._surya_models.get('ocr_detector')
            ocr_det_processor = self._surya_models.get('ocr_det_processor')
            
            if not all([ocr_recognizer, ocr_rec_processor, ocr_detector, ocr_det_processor]):
                print("OCR models not found in artifact_dict, falling back to PDF processing")
                # Fall back to original PDF-based approach
                return self._process_via_pdf(images)
            
            # Convert images to RGB if needed
            processed_images = []
            for img in images:
                if img.mode not in ('RGB', 'L'):
                    img = img.convert('RGB')
                processed_images.append(img)
            
            with self.lock:
                try:
                    # Import marker's OCR function
                    from marker.ocr.recognition import run_recognition
                    from marker.ocr.detection import run_detection
                    
                    all_text = []
                    langs = [["en"]] * len(processed_images)  # Default to English
                    
                    # Detect text regions
                    det_predictions = run_detection(processed_images, ocr_detector, ocr_det_processor)
                    
                    # OCR the detected regions
                    rec_predictions = run_recognition(
                        processed_images,
                        langs,
                        det_predictions,
                        ocr_recognizer,
                        ocr_rec_processor
                    )
                    
                    # Extract text from predictions
                    for page_pred in rec_predictions:
                        page_text = []
                        for text_line in page_pred.text_lines:
                            page_text.append(text_line.text)
                        all_text.append('\n'.join(page_text))
                    
                    combined_text = '\n\n'.join(all_text)
                    
                    # Clean up
                    del det_predictions, rec_predictions, processed_images
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()
                    
                    return [combined_text]
                    
                except ImportError as e:
                    print(f"Could not import marker OCR functions: {e}")
                    print("Falling back to PDF processing")
                    return self._process_via_pdf(images)
                except Exception as e:
                    print(f"Error in direct OCR processing: {e}")
                    import traceback
                    traceback.print_exc()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()
                    # Fall back to PDF processing
                    return self._process_via_pdf(images)

        except Exception as e:
            print(f"Error in process_images: {e}")
            import traceback
            traceback.print_exc()
            gc.collect()
            return [""]
    
    def _process_via_pdf(self, images: List[Image.Image]) -> List[str]:
        """
        Fallback method: Convert images to PDF and process with marker.
        
        Args:
            images: List of PIL Image objects to process
            
        Returns:
            List containing extracted text
        """
        try:
            # Convert PIL images to bytes first
            image_bytes_list = []
            for img in images:
                try:
                    img_byte_arr = io.BytesIO()
                    if img.mode not in ('RGB', 'L'):
                        img = img.convert('RGB')
                    img.save(img_byte_arr, format='JPEG', quality=95)
                    img_byte_arr.seek(0)
                    image_bytes_list.append(img_byte_arr.getvalue())
                except Exception:
                    img_byte_arr = io.BytesIO()
                    img.save(img_byte_arr, format='PNG')
                    img_byte_arr.seek(0)
                    image_bytes_list.append(img_byte_arr.getvalue())

            if not image_bytes_list:
                return [""]

            # Convert to PDF
            pdf_bytes = img2pdf.convert(image_bytes_list)

            # Process with marker
            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as pdf_file:
                pdf_file.write(pdf_bytes)
                pdf_path = pdf_file.name

            try:
                text = self.process_pdf_file(pdf_path)
                return [text]
            finally:
                try:
                    os.unlink(pdf_path)
                except OSError:
                    pass
                gc.collect()
                
        except Exception as e:
            print(f"Error in _process_via_pdf: {e}")
            gc.collect()
            return [""]