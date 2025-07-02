# download_models.py
from app.services.ocr import OCRFactory

if __name__ == "__main__":
    print("Downloading OCR models...")
    OCRFactory()
    print("OCR models downloaded successfully.")