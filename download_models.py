# download_models.py
import os

# Ensure HuggingFace uses the /scratch directory for caching
os.environ["HF_HOME"] = "/scratch/huggingface_cache"
os.environ["HUGGINGFACE_HUB_CACHE"] = "/scratch/huggingface_cache/hub"
os.environ["TRANSFORMERS_CACHE"] = "/scratch/huggingface_cache/transformers"
os.environ["HF_DATASETS_CACHE"] = "/scratch/huggingface_cache/datasets"

from app.services.ocr import discover_services, service_names, get_service

if __name__ == "__main__":
    print(f"HuggingFace cache directory: {os.environ.get('HF_HOME', 'Not set')}")
    print(f"Transformers cache: {os.environ.get('TRANSFORMERS_CACHE', 'Not set')}")
    print("Discovering OCR services...")
    
    # Create cache directory if it doesn't exist
    cache_dir = "/scratch/huggingface_cache"
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir, exist_ok=True)
        print(f"Created cache directory: {cache_dir}")
    
    # Discover available service classes without instantiating heavy models
    discover_services()
    services = service_names()
    print(f"Discovered services: {services}")
    
    # Lazily instantiate each service to trigger model downloads into /scratch
    print("Initializing services to pre-download models into /scratch...")
    for name in services:
        try:
            print(f"- Initializing service: {name}")
            svc = get_service(name)
            # Access known attributes to ensure models/processors are loaded
            if hasattr(svc, "model"):
                _ = getattr(svc, "model", None)
            if hasattr(svc, "processor"):
                _ = getattr(svc, "processor", None)
            print(f"  -> {name}: initialized")
        except Exception as e:
            print(f"  !! Failed to initialize {name}: {e}")
    
    print("Pre-download step completed.")