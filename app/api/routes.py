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
from fastapi.concurrency import run_in_threadpool

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
    
    if file_extension == ".pdf":
        pdf_bytes = file.file.read()
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        for page in doc:
            pix = page.get_pixmap()
            img_bytes = pix.tobytes("png")
            images.append(Image.open(io.BytesIO(img_bytes)))
        doc.close()
    elif file_extension in [".png", ".jpg", ".jpeg", ".bmp", ".gif"]:
        images.append(Image.open(file.file))
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type: {file_extension}",
        )
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
    images = await run_in_threadpool(file_to_images, file)
    
    if not images:
        raise HTTPException(status_code=400, detail="Could not process the uploaded file into images.")

    results = await run_in_threadpool(ocr_service.process_images, images)
    
    # Join the text from all pages/images into a single string
    full_text = "\n\n--- Page Break ---\n\n".join(results)
    
    return OCRResponse(text=full_text)

@router.get("/services")
async def list_services():
    """
    Returns a list of available OCR services.
    """
    return {"services": list(OCR_SERVICES.keys())}