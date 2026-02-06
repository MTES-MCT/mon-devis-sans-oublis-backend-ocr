from fastapi import APIRouter, File, UploadFile, Depends, HTTPException, status
from app.services.ocr import get_service, OCR_SERVICES
from app.models.ocr import OCRResponse
from app.config import config
from app.services.ocr.base import BaseOCRService
from app.services.ocr.sglang_base import SGLangOCRService, SGLangProcessingError
from app.exceptions import (
    OCRException,
    InvalidFileFormatError,
    CorruptedImageError,
    OCRProcessingError,
    GPUMemoryError
)
from typing import Callable, List, Optional, TypeVar
from PIL import Image
import fitz
import io
import os
import shutil
import gc
import torch
import tempfile
import uuid
import logging
import time
from fastapi.concurrency import run_in_threadpool
from apng import APNG
from sentry_sdk import logger as sentry_logger

logger = logging.getLogger(__name__)

# Define supported formats (shared by all services)
SUPPORTED_FORMATS = [".pdf", ".png", ".jpg", ".jpeg", ".bmp", ".gif"]
IMAGE_FORMATS = [".png", ".jpg", ".jpeg", ".bmp", ".gif"]

T = TypeVar("T")


def cleanup_temp_file(temp_file_path: Optional[str], *, filename: str) -> None:
    """Best-effort deletion of a temporary file.

    This project guarantees no persistent storage of user uploads; failures are
    logged but should not fail the request.
    """
    if not temp_file_path:
        return
    if not os.path.exists(temp_file_path):
        return

    try:
        os.remove(temp_file_path)
    except OSError as e:
        logger.warning(f"Failed to delete temp file {temp_file_path}: {e}")
        sentry_logger.error(
            'Failed to delete temporary file',
            attributes={
                'file.name': filename,
                'temp.file.path': temp_file_path,
                'error.type': type(e).__name__
            }
        )

        # Try to schedule cleanup later
        import atexit

        def _cleanup(path: str = temp_file_path) -> None:
            try:
                if os.path.exists(path):
                    os.remove(path)
            except OSError:
                pass

        atexit.register(_cleanup)


async def run_in_threadpool_with_gpu_retry(
    func: Callable[..., T],
    *args,
    service_name: str,
) -> T:
    """Run a blocking OCR call in the threadpool with a single CUDA OOM retry."""
    try:
        return await run_in_threadpool(func, *args)
    except torch.cuda.OutOfMemoryError:
        sentry_logger.warning(
            'GPU out of memory, attempting retry',
            attributes={
                'ocr.service': service_name,
                'retry.attempt': 1
            }
        )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        try:
            return await run_in_threadpool(func, *args)
        except torch.cuda.OutOfMemoryError:
            sentry_logger.error(
                'GPU out of memory after retry',
                attributes={
                    'ocr.service': service_name,
                    'retry.attempted': True,
                    'retry.success': False
                }
            )
            raise GPUMemoryError(service_name=service_name, retry_attempted=True)


def save_upload_to_temp_file(file: UploadFile, file_extension: str) -> str:
    """Persist an `UploadFile` to a unique temporary path.

    The caller is responsible for deleting the returned file path.

    This is used both for:
    - PDF -> image conversion (image-based services)
    - Direct PDF processing (PDF-native services)
    """
    if file_extension not in SUPPORTED_FORMATS:
        sentry_logger.warning(
            'Unsupported file format',
            attributes={
                'file.name': file.filename,
                'file.extension': file_extension,
                'error.type': 'InvalidFileFormatError'
            }
        )
        raise InvalidFileFormatError(file_extension, SUPPORTED_FORMATS)

    # Best-effort rewind: depending on the client/middleware the stream might not
    # be at position 0.
    try:
        file.file.seek(0)
    except Exception:
        pass

    with tempfile.NamedTemporaryFile(
        mode='wb',
        suffix=file_extension,
        prefix=f"ocr_{uuid.uuid4().hex}_",
        delete=False
    ) as temp_file:
        temp_file_path = temp_file.name
        shutil.copyfileobj(file.file, temp_file)

    return temp_file_path

def is_apng(file_path):
    try:
        apng = APNG.open(file_path)
        frames = list(apng.frames)
        return len(frames) > 1
    except Exception:
        return False

def extract_apng_frames(file_path, output_prefix="frame"):
    apng = APNG.open(file_path)
    extracted_images = []
    for idx, (png, control) in enumerate(apng.frames):
        # Save to a BytesIO object instead of a file
        img_bytes_io = io.BytesIO()
        png.save(img_bytes_io)
        img_bytes_io.seek(0)
        extracted_images.append(Image.open(img_bytes_io))
    return extracted_images

def pdf_pages_to_images(pdf_path, dpi=None):
    """
    Convert PDF pages to PIL Images
    
    Args:
        pdf_path (str): Path to PDF file
        dpi (int): Resolution for rendering (default from config.PDF_DPI)
        
    Returns:
        list: List of PIL Image objects, one per page
    """
    if dpi is None:
        dpi = config.PDF_DPI
    
    start_time = time.time()
    doc = None
    try:
        doc = fitz.open(pdf_path)
        images = []
        num_pages = len(doc)
        
        logger.info(f"[PDF Conversion] Starting conversion of {num_pages} pages at {dpi} DPI")
        
        for page_num in range(num_pages):
            page = doc[page_num]
            
            # Create matrix for DPI scaling
            mat = fitz.Matrix(dpi/72, dpi/72)
            
            # Render page to pixmap
            pix = page.get_pixmap(matrix=mat)
            
            # Convert to PIL Image
            img_data = pix.tobytes("png")
            pil_image = Image.open(io.BytesIO(img_data))
            images.append(pil_image)
            
            # Free pixmap memory immediately
            pix = None
        
        elapsed_time = time.time() - start_time
        logger.info(f"[PDF Conversion] Completed {num_pages} pages in {elapsed_time:.2f}s ({elapsed_time/num_pages:.2f}s per page)")
        
        return images
    finally:
        if doc:
            doc.close()
        # Force garbage collection after processing
        gc.collect()

router = APIRouter()

def get_ocr_service(service_name: str) -> BaseOCRService:
    try:
        return get_service(service_name)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))

def file_to_images(file: UploadFile) -> List[Image.Image]:
    """
    Converts an uploaded file (PDF or image) into a list of PIL Image objects.
    
    IMPORTANT: All files are processed temporarily and deleted immediately.
    No files are stored persistently.
    """
    images = []
    file_extension = os.path.splitext(file.filename)[1].lower()
    temp_file_path = None

    try:
        temp_file_path = save_upload_to_temp_file(file, file_extension)

        if file_extension == ".pdf":
            images = pdf_pages_to_images(temp_file_path)
        elif file_extension in IMAGE_FORMATS:
            try:
                if is_apng(temp_file_path):
                    extracted_frames = extract_apng_frames(temp_file_path)
                    images.extend(extracted_frames)
                else:
                    # FIXED: Read from the saved temp file, not the consumed file object
                    with Image.open(temp_file_path) as img:
                        # Convert to RGB if needed and create a copy
                        # This ensures the image is fully loaded in memory
                        if img.mode in ('RGBA', 'LA'):
                            # Convert RGBA/LA to RGB
                            rgb_img = Image.new('RGB', img.size, (255, 255, 255))
                            rgb_img.paste(img, mask=img.split()[-1] if img.mode == 'RGBA' else None)
                            images.append(rgb_img)
                        else:
                            images.append(img.convert('RGB') if img.mode != 'RGB' else img.copy())
            except Exception as e:
                sentry_logger.error(
                    'Failed to load image',
                    attributes={
                        'file.name': file.filename,
                        'error.type': type(e).__name__,
                        'error.message': str(e)
                    }
                )
                raise CorruptedImageError(
                    filename=file.filename,
                    error_detail=str(e)
                )

        if not images:
            sentry_logger.error(
                'No images extracted from file',
                attributes={
                    'file.name': file.filename,
                    'error.type': 'NoImagesExtracted'
                }
            )
            raise CorruptedImageError(
                filename=file.filename,
                error_detail="No images could be extracted from the file"
            )

    except OCRException:
        # Re-raise our custom exceptions
        raise
    except Exception as e:
        sentry_logger.error(
            'Unexpected error processing file',
            attributes={
                'file.name': file.filename,
                'error.type': type(e).__name__,
                'error.message': str(e)
            }
        )
        # Catch any unexpected errors
        raise OCRException(
            message=f"Unexpected error processing file: {file.filename}",
            details={"filename": file.filename, "error": str(e)},
            status_code=500
        )
    finally:
        # CRITICAL: Always delete the temporary file
        cleanup_temp_file(temp_file_path, filename=file.filename)

        # Force garbage collection to free memory
        gc.collect()

    return images


@router.post("/ocr/{service_name}", response_model=OCRResponse)
async def ocr(
    service_name: str,
    file: UploadFile = File(...),
    ocr_service: BaseOCRService = Depends(get_ocr_service),
):
    """
    OCR endpoint with enhanced error handling and async batch processing.
    
    Supports both sync (marker) and async (VLLM) services.
    No files are stored persistently - all processing is done in memory.
    """
    import sentry_sdk
    
    # Track file info for monitoring
    file_info = {
        "filename": file.filename,
        "content_type": file.content_type,
        "size": getattr(file, 'size', 'unknown')
    }
    
    # Add Sentry context
    with sentry_sdk.configure_scope() as scope:
        scope.set_tag("ocr.service", service_name)
        scope.set_context("file", file_info)
    
    images = None
    pdf_temp_path = None
    try:
        file_extension = os.path.splitext(file.filename)[1].lower()

        preferred_input = ocr_service.preferred_input_type(file_extension)

        # If the service prefers direct PDF processing for this upload, avoid
        # the sub-optimal PDF -> images -> PDF roundtrip.
        if preferred_input == "pdf":
            save_start = time.time()
            pdf_temp_path = await run_in_threadpool(save_upload_to_temp_file, file, file_extension)
            save_time = time.time() - save_start
            logger.info(f"[File Save] Saved {file.filename} to temp PDF in {save_time:.2f}s")

            process_fn = ocr_service.process_pdf_file

            try:
                process_start = time.time()
                text = await run_in_threadpool_with_gpu_retry(
                    process_fn, pdf_temp_path, service_name=service_name
                )
                process_time = time.time() - process_start
                logger.info(f"[Direct PDF] Processed {file.filename} directly in {process_time:.2f}s")
                results = [text]
            except GPUMemoryError:
                raise
            except Exception as e:
                sentry_logger.error(
                    'OCR processing failed',
                    attributes={
                        'ocr.service': service_name,
                        'error.type': type(e).__name__,
                        'error.message': str(e)
                    }
                )
                # Clear GPU memory on any error
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

                raise OCRProcessingError(
                    service_name=service_name,
                    error_detail=str(e)
                )
        else:
            # Convert file to images (needed for image-based services, including VLLM)
            conversion_start = time.time()
            images = await run_in_threadpool(file_to_images, file)
            conversion_time = time.time() - conversion_start
            logger.info(f"[File Conversion] Converted {file.filename} to {len(images)} images in {conversion_time:.2f}s")

            # Process based on service type
            if isinstance(ocr_service, SGLangOCRService):
                # SGLang services: async batch processing
                try:
                    results = await ocr_service.process_images_async(images)
                except SGLangProcessingError as e:
                    sentry_logger.error(
                        'SGLang OCR processing failed',
                        attributes={
                            'ocr.service': service_name,
                            'error.type': 'SGLangProcessingError',
                            'error.message': str(e)
                        }
                    )
                    raise OCRProcessingError(
                        service_name=service_name,
                        error_detail=f"SGLang processing failed: {str(e)}"
                    )
                except Exception as e:
                    sentry_logger.error(
                        'Unexpected SGLang error',
                        attributes={
                            'ocr.service': service_name,
                            'error.type': type(e).__name__,
                            'error.message': str(e)
                        }
                    )
                    raise OCRProcessingError(
                        service_name=service_name,
                        error_detail=f"Unexpected error: {str(e)}"
                    )
            else:
                # Non-VLLM services: sync processing with threadpool
                try:
                    results = await run_in_threadpool_with_gpu_retry(
                        ocr_service.process_images, images, service_name=service_name
                    )
                except GPUMemoryError:
                    raise
                except Exception as e:
                    sentry_logger.error(
                        'OCR processing failed',
                        attributes={
                            'ocr.service': service_name,
                            'error.type': type(e).__name__,
                            'error.message': str(e)
                        }
                    )
                    # Clear GPU memory on any error
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()

                    raise OCRProcessingError(
                        service_name=service_name,
                        error_detail=str(e)
                    )

        # Join the text from all pages/images
        full_text = "\n\n--- Page Break ---\n\n".join(results)

        # Log successful processing
        logger.info(f"Successfully processed {file.filename} with {service_name}")

        return OCRResponse(text=full_text)

    finally:
        # Ensure complete memory cleanup
        if images:
            for img in images:
                if hasattr(img, 'close'):
                    img.close()
            del images

        # Ensure no persistent storage (direct-PDF path)
        cleanup_temp_file(pdf_temp_path, filename=file.filename)

        gc.collect()

        # Clear GPU memory for non-SGLang services
        if not isinstance(ocr_service, SGLangOCRService) and torch.cuda.is_available():
            torch.cuda.empty_cache()

@router.get("/services")
async def list_services():
    """
    Returns a list of available OCR services.
    """
    services = list(OCR_SERVICES.keys())
    return {"services": services}

@router.get("/health")
async def health_check():
    """
    Health check endpoint to verify service status and GPU memory.
    """
    health_status = {
        "status": "healthy",
        "services": list(OCR_SERVICES.keys())
    }
    
    if torch.cuda.is_available():
        try:
            # Get GPU memory stats
            allocated = torch.cuda.memory_allocated() / 1024**3  # Convert to GB
            reserved = torch.cuda.memory_reserved() / 1024**3
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            
            health_status["gpu"] = {
                "available": True,
                "allocated_gb": round(allocated, 2),
                "reserved_gb": round(reserved, 2),
                "total_gb": round(total, 2),
                "free_gb": round(total - reserved, 2)
            }
        except Exception as e:
            health_status["gpu"] = {
                "available": False,
                "error": str(e)
            }
            sentry_logger.error(
                'GPU health check failed',
                attributes={
                    'error.type': type(e).__name__,
                    'error.message': str(e)
                }
            )
    else:
        health_status["gpu"] = {"available": False}
    return health_status
                