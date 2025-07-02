from fastapi import APIRouter, File, UploadFile, HTTPException
from app.services.ocr import ocr_factory
from app.models.ocr import OCRResponse
import shutil
import os

router = APIRouter()

@router.post("/ocr/{model_name}", response_model=OCRResponse)
async def ocr(
    model_name: str,
    file: UploadFile = File(...),
):
    """
    This endpoint receives a file (PDF or image), saves it to a temporary location,
    processes it using the specified OCR model, and returns the result.
    
    Available models: nanonets, olmocr
    """
    if model_name not in ["nanonets", "olmocr"]:
        raise HTTPException(status_code=400, detail="Invalid OCR model specified. Available models: nanonets, olmocr")

    temp_dir = "temp_storage"
    os.makedirs(temp_dir, exist_ok=True)
    temp_file_path = os.path.join(temp_dir, file.filename)

    try:
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        result_text = await ocr_factory.process_file(temp_file_path, file.filename, model_name)
        return OCRResponse(text=result_text)
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)