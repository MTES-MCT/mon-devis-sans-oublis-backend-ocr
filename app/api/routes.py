from fastapi import APIRouter, File, UploadFile, Depends
from app.services.ocr.nanonets import ocr_service
from app.models.ocr import OCRResponse
import shutil
import os

router = APIRouter()

@router.post("/ocr", response_model=OCRResponse)
async def ocr(file: UploadFile = File(...)):
    """
    This endpoint receives a file (PDF or image), saves it to a temporary location,
    processes it using an OCR model, and returns the result.
    """
    temp_dir = "temp_storage"
    os.makedirs(temp_dir, exist_ok=True)
    temp_file_path = os.path.join(temp_dir, file.filename)

    try:
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        result_text = await ocr_service.process_file(temp_file_path, file.filename)
        return OCRResponse(text=result_text)
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)