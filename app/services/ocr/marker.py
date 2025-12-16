import gc
import io
import logging
import os
import tempfile
import threading
import time
from typing import List

import img2pdf
from PIL import Image

from .base import BaseOCRService, OCRInputType

logger = logging.getLogger(__name__)


def _configure_marker_environment() -> None:
    """Configure marker runtime defaults for this API service.

    Goals for this backend endpoint:
    - Always OCR full pages (even if a PDF contains an OCR layer or text)
    - Never return extracted image placeholders as the primary output
    - Prefer GPU execution when available

    Note: marker reads many settings from environment variables in its
    `marker.settings` module at import time. Therefore this function must run
    before importing marker modules.
    """
    # Force OCR for all pages (treat this service as an OCR engine, not a text extractor)
    os.environ["OCR_ALL_PAGES"] = os.getenv("MARKER_OCR_ALL_PAGES", "true")

    # Compatibility with alternative naming seen in some marker versions / wrappers
    os.environ.setdefault("FORCE_OCR", os.getenv("MARKER_FORCE_OCR", "true"))

    # If the PDF contains a bad OCR layer, prefer re-OCR.
    # Some marker versions use STRIP_EXISTING_OCR, others use STRIP_OCR.
    os.environ["STRIP_EXISTING_OCR"] = os.getenv("MARKER_STRIP_EXISTING_OCR", "true")
    os.environ.setdefault("STRIP_OCR", os.environ["STRIP_EXISTING_OCR"])

    # Avoid returning figure/image placeholders like ![](_page_0_Figure_0.jpeg)
    os.environ["DISABLE_IMAGE_EXTRACTION"] = os.getenv(
        "MARKER_DISABLE_IMAGE_EXTRACTION",
        "true",
    )

    # Improve OCR detection robustness for scanned PDFs.
    # Marker uses a settings-driven DPI for PDF rasterization.
    # Different versions/wrappers may use different env var names.
    marker_dpi = os.getenv("MARKER_PDF_DPI", "200")
    os.environ.setdefault("PDF_DPI", marker_dpi)
    os.environ.setdefault("RENDER_DPI", marker_dpi)
    os.environ.setdefault("PAGE_DPI", marker_dpi)

    # Prefer GPU if available.
    # IMPORTANT: if `import torch` fails (missing CUDA driver libs, wrong wheel,
    # etc.), transformers will silently disable torch-backed symbols (like
    # PreTrainedModel), which breaks surya/marker.
    try:
        import torch
    except Exception as e:
        logger.exception(
            "Failed to import torch inside the API container. "
            "This usually means the container has no access to NVIDIA driver libs "
            "(missing `gpus: all` / NVIDIA Container Toolkit), or torch was installed "
            "with incompatible binaries. Error: %s",
            e,
        )
        raise

    if torch.cuda.is_available():
        os.environ.setdefault("TORCH_DEVICE", "cuda")
    else:
        os.environ.setdefault("TORCH_DEVICE", "cpu")


def _strip_marker_image_placeholders(text: str) -> str:
    """Remove marker image placeholders from markdown output.

    Marker can emit lines like `![](_page_0_Figure_0.jpeg)` for extracted images.

    If removing placeholders would result in an (almost) empty output, keep the
    original text to avoid returning a blank response.
    """
    cleaned_lines: List[str] = []
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("![](") and "_page_" in s:
            continue
        cleaned_lines.append(line)

    cleaned = "\n".join(cleaned_lines).strip()
    if len(cleaned) < 20:
        return text.strip()
    return cleaned


class MarkerOCRService(BaseOCRService):
    _service_name = "marker"
    _initialization_lock = threading.Lock()
    _worker_converters = {}  # Store converter per worker process

    def __init__(self):
        super().__init__()
        self.lock = threading.Lock()  # Lock to serialize PDF processing
        self._converter = None

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
                    # Configure env-driven marker settings before importing marker.
                    _configure_marker_environment()

                    # Import marker after env configuration (marker.settings reads env at import time)
                    from marker.config.parser import ConfigParser
                    from marker.converters.pdf import PdfConverter
                    from marker.models import create_model_dict

                    # Prevent re-downloading if models exist
                    if os.path.exists('/root/.cache/datalab/models'):
                        os.environ['HF_HUB_OFFLINE'] = '1'

                    # Add small delay between worker initializations to avoid conflicts
                    if len(MarkerOCRService._worker_converters) > 0:
                        time.sleep(0.5 * len(MarkerOCRService._worker_converters))

                    # Build the converter the same way `marker_single` does.
                    marker_config = {
                        "output_format": "markdown",
                        "force_ocr": True,
                        "strip_existing_ocr": True,
                        "disable_image_extraction": True,
                        "use_llm": False,
                        "workers": 1,
                    }
                    config_parser = ConfigParser(marker_config)

                    converter = PdfConverter(
                        config=config_parser.generate_config_dict(),
                        artifact_dict=create_model_dict(),
                        processor_list=config_parser.get_processors(),
                        renderer=config_parser.get_renderer(),
                        llm_service=config_parser.get_llm_service(),
                    )

                    # Store converter for this worker
                    MarkerOCRService._worker_converters[pid] = converter
                    self._converter = converter

                    logger.info(
                        "Marker OCR service initialized for worker %s on attempt %s",
                        pid,
                        attempt + 1,
                    )
                    break

                except Exception as e:
                    if attempt < max_retries - 1:
                        logger.warning(
                            "Failed to initialize Marker OCR for worker %s (attempt %s/%s): %s",
                            pid,
                            attempt + 1,
                            max_retries,
                            e,
                        )
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

    def warmup(self) -> None:
        self._ensure_initialized()

    def preferred_input_type(self, file_extension: str) -> OCRInputType:
        # Marker can process PDFs directly. For image uploads, routes will still
        # convert to PIL images and we will re-wrap them into a PDF.
        return "pdf" if file_extension == ".pdf" else "images"

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
                # Import lazily (see note about marker env reading at import time)
                from marker.output import text_from_rendered

                rendered = self.converter(pdf_path)
                text, _, _ = text_from_rendered(rendered)
                text = _strip_marker_image_placeholders(text)

                # Clean up to prevent memory leaks
                del rendered
                gc.collect()

                return text
            except Exception as e:
                logger.exception("Error processing PDF with Marker: %s", e)
                gc.collect()
                raise

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
            logger.exception("Error in process_images: %s", e)
            gc.collect()
            return [""]