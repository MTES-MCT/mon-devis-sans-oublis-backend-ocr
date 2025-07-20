#!/usr/bin/env python3
"""Test script to debug marker service issues"""

import sys
import os
import io
import tempfile
from PIL import Image
import img2pdf

def test_img2pdf_conversion():
    """Test img2pdf with different input formats"""
    print("Testing img2pdf conversion...")
    
    # Create a test image
    test_img = Image.new('RGB', (100, 100), color='red')
    
    # Test 1: Convert PIL Image to bytes first
    print("\nTest 1: PIL Image -> bytes -> img2pdf")
    try:
        import io
        img_bytes = io.BytesIO()
        test_img.save(img_bytes, format='JPEG')
        img_bytes.seek(0)
        pdf_bytes = img2pdf.convert([img_bytes.getvalue()])
        print(f"✓ Success! PDF size: {len(pdf_bytes)} bytes")
    except Exception as e:
        print(f"✗ Failed: {e}")
    
    # Test 2: Save to temp file first
    print("\nTest 2: PIL Image -> temp file -> img2pdf")
    try:
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            test_img.save(tmp.name, 'JPEG')
            tmp_path = tmp.name
        
        pdf_bytes = img2pdf.convert([tmp_path])
        print(f"✓ Success! PDF size: {len(pdf_bytes)} bytes")
        os.unlink(tmp_path)
    except Exception as e:
        print(f"✗ Failed: {e}")
    
    # Test 3: Multiple images
    print("\nTest 3: Multiple PIL Images -> bytes -> img2pdf")
    try:
        images_bytes = []
        for i in range(3):
            img = Image.new('RGB', (100, 100), color=['red', 'green', 'blue'][i])
            img_bytes = io.BytesIO()
            img.save(img_bytes, format='JPEG')
            img_bytes.seek(0)
            images_bytes.append(img_bytes.getvalue())
        
        pdf_bytes = img2pdf.convert(images_bytes)
        print(f"✓ Success! PDF size: {len(pdf_bytes)} bytes")
    except Exception as e:
        print(f"✗ Failed: {e}")

def test_marker_converter():
    """Test marker PDF converter"""
    print("\n\nTesting marker converter...")
    
    try:
        from marker.converters.pdf import PdfConverter
        from marker.models import create_model_dict
        from marker.output import text_from_rendered
        
        print("✓ Imports successful")
        
        # Initialize converter
        print("Initializing converter...")
        converter = PdfConverter(
            artifact_dict=create_model_dict(),
        )
        print("✓ Converter initialized")
        
        # Create a test PDF
        test_img = Image.new('RGB', (200, 200), color='white')
        img_bytes = io.BytesIO()
        test_img.save(img_bytes, format='JPEG')
        img_bytes.seek(0)
        pdf_bytes = img2pdf.convert([img_bytes.getvalue()])
        
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
            tmp.write(pdf_bytes)
            tmp_path = tmp.name
        
        print(f"Created test PDF: {tmp_path} ({len(pdf_bytes)} bytes)")
        
        # Process with marker
        print("Processing with marker...")
        rendered = converter(tmp_path)
        text, _, _ = text_from_rendered(rendered)
        print(f"✓ Processing complete. Text length: {len(text)}")
        
        os.unlink(tmp_path)
        
    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_img2pdf_conversion()
    test_marker_converter()