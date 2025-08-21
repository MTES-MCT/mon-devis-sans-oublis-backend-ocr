from fastapi import APIRouter, File, UploadFile, Depends, HTTPException, status
from app.services.ocr import get_service, OCR_SERVICES
from app.models.ocr import OCRResponse
from app.services.ocr.base import BaseOCRService
from typing import List
from PIL import Image
import fitz
import io
import os
import shutil
import gc
import torch
from fastapi.concurrency import run_in_threadpool
from apng import APNG

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
    """
    images = []
    file_extension = os.path.splitext(file.filename)[1].lower()
    temp_file_path = None

    try:
        # Save the uploaded file to a temporary location to check for APNG
        temp_file_path = f"/tmp/{file.filename}"
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        if file_extension == ".pdf":
            images = pdf_pages_to_images(temp_file_path, dpi=150)
        elif file_extension in [".png", ".jpg", ".jpeg", ".bmp", ".gif"]:
            if is_apng(temp_file_path):
                extracted_frames = extract_apng_frames(temp_file_path)
                images.extend(extracted_frames)
            else:
                file.file.seek(0)
                images.append(Image.open(file.file))
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported file type: {file_extension}",
            )
    finally:
        # Always clean up the temporary file
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except OSError as e:
                print(f"Error removing temp file: {e}")
        
        # Force garbage collection
        gc.collect()
        
    return images


@router.post("/ocr/{service_name}", response_model=OCRResponse)
async def ocr(
    service_name: str,
    file: UploadFile = File(...),
    ocr_service: BaseOCRService = Depends(get_ocr_service),
):
    """
    This endpoint receives a file (PDF or image), converts it to images,
    processes them using the specified OCR model, and returns the result.
    """
    images = None
    try:
        # Convert file to images
        images = await run_in_threadpool(file_to_images, file)
        
        if not images:
            raise HTTPException(status_code=400, detail="Could not process the uploaded file into images.")

        # Process images with OCR service
        try:
            results = await run_in_threadpool(ocr_service.process_images, images)
        except torch.cuda.OutOfMemoryError:
            # Clear GPU memory and retry once
            print(f"GPU OOM error in {service_name}, clearing cache and retrying...")
            torch.cuda.empty_cache()
            gc.collect()
            
            # Retry with cleared memory
            results = await run_in_threadpool(ocr_service.process_images, images)
        except Exception as e:
            print(f"Error in OCR service {service_name}: {e}")
            # Clear GPU memory on any error
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
            # Return a more informative error
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"OCR processing failed: {str(e)}"
            )
        
        # Join the text from all pages/images into a single string
        full_text = "\n\n--- Page Break ---\n\n".join(results)
        
        return OCRResponse(text=full_text)
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        print(f"Unexpected error in OCR endpoint: {e}")
        # Ensure cleanup on any unexpected error
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred: {str(e)}"
        )
    finally:
        # Clean up images from memory
        if images:
            del images
        gc.collect()
        
        # Final GPU memory cleanup
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
                