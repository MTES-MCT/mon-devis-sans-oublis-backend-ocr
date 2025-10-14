from fastapi import APIRouter, File, UploadFile, Depends, HTTPException, status
from app.services.ocr import get_service, OCR_SERVICES
from app.models.ocr import OCRResponse
from app.services.ocr.base import BaseOCRService
from app.exceptions import (
    OCRException,
    InvalidFileFormatError,
    CorruptedImageError,
    OCRProcessingError,
    GPUMemoryError
)
from typing import List
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
from fastapi.concurrency import run_in_threadpool
from apng import APNG

logger = logging.getLogger(__name__)

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

def pdf_pages_to_images(pdf_path, dpi=150):
    """
    Convert PDF pages to PIL Images
    
    Args:
        pdf_path (str): Path to PDF file
        dpi (int): Resolution for rendering (default 150)
        
    Returns:
        list: List of PIL Image objects, one per page
    """
    doc = None
    try:
        doc = fitz.open(pdf_path)
        images = []
        
        for page_num in range(len(doc)):
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
    
    # Define supported formats
    SUPPORTED_FORMATS = [".pdf", ".png", ".jpg", ".jpeg", ".bmp", ".gif"]

    if file_extension not in SUPPORTED_FORMATS:
        raise InvalidFileFormatError(file_extension, SUPPORTED_FORMATS)

    try:
        # Create a unique temporary file to avoid conflicts
        # Use tempfile for better security and automatic cleanup
        with tempfile.NamedTemporaryFile(
            mode='wb',
            suffix=file_extension,
            prefix=f"ocr_{uuid.uuid4().hex}_",
            delete=False
        ) as temp_file:
            temp_file_path = temp_file.name
            # Copy uploaded file to temporary location
            shutil.copyfileobj(file.file, temp_file)
        
        if file_extension == ".pdf":
            images = pdf_pages_to_images(temp_file_path, dpi=150)
        elif file_extension in [".png", ".jpg", ".jpeg", ".bmp", ".gif"]:
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
                raise CorruptedImageError(
                    filename=file.filename,
                    error_detail=str(e)
                )
                
        if not images:
            raise CorruptedImageError(
                filename=file.filename,
                error_detail="No images could be extracted from the file"
            )
            
    except OCRException:
        # Re-raise our custom exceptions
        raise
    except Exception as e:
        # Catch any unexpected errors
        raise OCRException(
            message=f"Unexpected error processing file: {file.filename}",
            details={"filename": file.filename, "error": str(e)},
            status_code=500
        )
    finally:
        # CRITICAL: Always delete the temporary file
        # This ensures no files are kept in storage
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
                # Log successful cleanup for audit
                logger.debug(f"Temporary file deleted: {temp_file_path}")
            except OSError as e:
                # Log error but don't fail the request
                logger.warning(f"Failed to delete temp file {temp_file_path}: {e}")
                # Try to schedule cleanup later
                import atexit
                atexit.register(lambda: os.path.exists(temp_file_path) and os.remove(temp_file_path))
        
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
    OCR endpoint with enhanced error handling and guaranteed file cleanup.
    
    No files are stored persistently - all processing is done in memory
    or with temporary files that are immediately deleted.
    """
    import sentry_sdk
    
    # Track file info for monitoring (no actual storage)
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
    try:
        # Convert file to images with proper error handling
        # All files are temporarily processed and immediately deleted
        images = await run_in_threadpool(file_to_images, file)
        
        # Process images with OCR service
        try:
            results = await run_in_threadpool(ocr_service.process_images, images)
        except torch.cuda.OutOfMemoryError:
            # Clear GPU memory and retry once
            logger.warning(f"GPU OOM in {service_name}, attempting retry...")
            torch.cuda.empty_cache()
            gc.collect()
            
            try:
                results = await run_in_threadpool(ocr_service.process_images, images)
            except torch.cuda.OutOfMemoryError:
                # If it fails again, raise custom exception
                raise GPUMemoryError(service_name=service_name, retry_attempted=True)
                
        except Exception as e:
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
        
        # Log successful processing (no file storage)
        logger.info(f"Successfully processed {file.filename} with {service_name}")
        
        return OCRResponse(text=full_text)
        
    finally:
        # Ensure complete memory cleanup
        if images:
            # Clear all image references
            for img in images:
                if hasattr(img, 'close'):
                    img.close()
            del images
        
        # Force garbage collection
        gc.collect()
        
        # Clear GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

@router.get("/services")
async def list_services():
    """
    Returns a list of available OCR services.
    """
    return {"services": list(OCR_SERVICES.keys())}

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
    else:
        health_status["gpu"] = {"available": False}
    
    return health_status
                