#!/usr/bin/env python3
"""Debug script to test marker with actual PDF processing"""

import os
import sys
import tempfile
import requests
from pathlib import Path

def test_marker_locally():
    """Test marker service locally without Docker"""
    print("Testing marker service locally...")
    
    # First, let's test if we can process a simple PDF
    try:
        from app.services.ocr.marker import MarkerOCRService
        from PIL import Image
        
        print("Creating test images...")
        # Create test images with text
        images = []
        for i in range(2):
            img = Image.new('RGB', (400, 200), color='white')
            images.append(img)
        
        print(f"Created {len(images)} test images")
        
        # Initialize service
        print("Initializing MarkerOCRService...")
        service = MarkerOCRService()
        
        # Process images
        print("Processing images...")
        results = service.process_images(images)
        
        print(f"Results: {results}")
        print(f"Number of results: {len(results)}")
        print(f"Text length: {len(results[0]) if results else 0}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

def test_pdf_direct():
    """Test processing a PDF file directly"""
    print("\n\nTesting direct PDF processing...")
    
    try:
        from marker.converters.pdf import PdfConverter
        from marker.models import create_model_dict
        from marker.output import text_from_rendered
        import PyPDF2
        from reportlab.pdfgen import canvas
        from reportlab.lib.pagesizes import letter
        
        # Create a simple PDF with text
        pdf_path = tempfile.mktemp(suffix='.pdf')
        c = canvas.Canvas(pdf_path, pagesize=letter)
        c.drawString(100, 750, "This is a test PDF")
        c.drawString(100, 700, "Created for testing marker OCR")
        c.showPage()
        c.drawString(100, 750, "This is page 2")
        c.save()
        
        print(f"Created test PDF: {pdf_path}")
        print(f"File size: {os.path.getsize(pdf_path)} bytes")
        
        # Test with marker
        converter = PdfConverter(
            artifact_dict=create_model_dict(),
        )
        
        print("Processing with marker...")
        rendered = converter(pdf_path)
        text, _, _ = text_from_rendered(rendered)
        
        print(f"Extracted text: {text[:200]}...")
        print(f"Total text length: {len(text)}")
        
        # Clean up
        os.unlink(pdf_path)
        
    except ImportError as e:
        print(f"Import error (install reportlab): {e}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Add the app directory to Python path
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    
    test_marker_locally()
    test_pdf_direct()