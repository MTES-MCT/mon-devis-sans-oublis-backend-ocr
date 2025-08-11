# download_models.py
import os

# Ensure HuggingFace uses the /scratch directory for caching
os.environ["HF_HOME"] = "/scratch/huggingface_cache"
os.environ["HUGGINGFACE_HUB_CACHE"] = "/scratch/huggingface_cache/hub"
os.environ["TRANSFORMERS_CACHE"] = "/scratch/huggingface_cache/transformers"
os.environ["HF_DATASETS_CACHE"] = "/scratch/huggingface_cache/datasets"

from app.services.ocr import discover_services

if __name__ == "__main__":
    print(f"HuggingFace cache directory: {os.environ.get('HF_HOME', 'Not set')}")
    print(f"Transformers cache: {os.environ.get('TRANSFORMERS_CACHE', 'Not set')}")
    print("Discovering and downloading OCR models...")
    
    # Create cache directory if it doesn't exist
    cache_dir = "/scratch/huggingface_cache"
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir, exist_ok=True)
        print(f"Created cache directory: {cache_dir}")
    
    discover_services()
    print("OCR models downloaded successfully.")