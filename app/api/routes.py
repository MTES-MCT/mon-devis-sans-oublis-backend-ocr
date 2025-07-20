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
        
        pix = None  # Free memory
    
    doc.close()
    return images

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

    # Save the uploaded file to a temporary location to check for APNG
    temp_file_path = f"/tmp/{file.filename}"
    with open(temp_file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    if file_extension == ".pdf":
        print(f"Processing PDF file: {file.filename}, size: {os.path.getsize(temp_file_path)} bytes")
        images = pdf_pages_to_images(temp_file_path, dpi=150)
        print(f"Converted PDF to {len(images)} images")
    elif file_extension in [".png", ".jpg", ".jpeg", ".bmp", ".gif"]:
        if is_apng(temp_file_path):
            extracted_frames = extract_apng_frames(temp_file_path)
            images.extend(extracted_frames)
        else:
            file.file.seek(0)
            images.append(Image.open(file.file))
        
        # Clean up the temporary file
        os.remove(temp_file_path)
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type: {file_extension}",
        )
    return images


@router.post("/test/marker-pdf", response_model=OCRResponse)
async def test_marker_pdf(
    file: UploadFile = File(...),
):
    """
    Test endpoint that processes PDFs directly with marker without image conversion.
    """
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="This test endpoint only accepts PDF files")
    
    # Save uploaded file to temporary location
    temp_file_path = f"/tmp/test_{file.filename}"
    try:
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        print(f"Saved PDF to {temp_file_path}, size: {os.path.getsize(temp_file_path)} bytes")
        
        # Process directly with marker
        from app.services.ocr.marker import MarkerOCRService
        service = MarkerOCRService()
        text = service.process_pdf_file(temp_file_path)
        
        return OCRResponse(text=text)
        
    except Exception as e:
        print(f"Error in test endpoint: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)


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

    print(f"processing {len(images)} images")
    results = await run_in_threadpool(ocr_service.process_images, images)
    print(f"returning {len(results)} results")
    # Join the text from all pages/images into a single string
    full_text = "\n\n--- Page Break ---\n\n".join(results)
    
    return OCRResponse(text=full_text)


@router.post("/ocr/marker-direct", response_model=OCRResponse)
async def ocr_marker_direct(
    file: UploadFile = File(...),
):
    """
    Special endpoint for marker that processes PDFs directly without converting to images.
    This is more efficient for PDF files.
    """
    file_extension = os.path.splitext(file.filename)[1].lower()
    
    if file_extension != ".pdf":
        # Fall back to regular processing for non-PDF files
        ocr_service = get_ocr_service("marker")
        images = await run_in_threadpool(file_to_images, file)
        if not images:
            raise HTTPException(status_code=400, detail="Could not process the uploaded file into images.")
        results = await run_in_threadpool(ocr_service.process_images, images)
        full_text = "\n\n--- Page Break ---\n\n".join(results)
        return OCRResponse(text=full_text)
    
    # Direct PDF processing
    try:
        from app.services.ocr.marker_optimized import MarkerDirectService
        service = MarkerDirectService()
        
        # Reset file position
        file.file.seek(0)
        
        # Process PDF directly
        text = await run_in_threadpool(service.process_pdf_upload, file)
        return OCRResponse(text=text)
        
    except Exception as e:
        print(f"Error in marker-direct endpoint: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")


@router.get("/services")
async def list_services():
    """
    Returns a list of available OCR services.
    """
    return {"services": list(OCR_SERVICES.keys())}