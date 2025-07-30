# download_models.py
from app.services.ocr import discover_services

if __name__ == "__main__":
    print("Discovering and downloading OCR models...")
    discover_services()
    print("OCR models downloaded successfully.")