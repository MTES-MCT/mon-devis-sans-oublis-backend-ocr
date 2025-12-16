# download_models.py
from app.services.ocr import discover_services, OCR_SERVICES

if __name__ == "__main__":
    print("Discovering OCR services...")
    discover_services()

    print("Warming up enabled OCR services (model download / initialization)...")
    for name, service in OCR_SERVICES.items():
        try:
            service.warmup()
            print(f"Warmup OK: {name}")
        except Exception as e:
            # Fail fast during build if a warmup is expected to work.
            raise RuntimeError(f"Warmup failed for {name}: {e}")

    print("OCR models downloaded successfully.")