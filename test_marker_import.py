#!/usr/bin/env python3
"""Test script to verify marker-pdf installation and imports"""

import sys
print(f"Python version: {sys.version}")

try:
    import marker
    print(f"✓ marker module found at: {marker.__file__}")
except ImportError as e:
    print(f"✗ Failed to import marker: {e}")
    sys.exit(1)

try:
    from marker.converters.pdf import PdfConverter
    print("✓ Successfully imported PdfConverter from marker.converters.pdf")
except ImportError as e:
    print(f"✗ Failed to import PdfConverter: {e}")
    sys.exit(1)

try:
    from marker.models import create_model_dict
    print("✓ Successfully imported create_model_dict from marker.models")
except ImportError as e:
    print(f"✗ Failed to import create_model_dict: {e}")
    sys.exit(1)

try:
    from marker.output import text_from_rendered
    print("✓ Successfully imported text_from_rendered from marker.output")
except ImportError as e:
    print(f"✗ Failed to import text_from_rendered: {e}")
    sys.exit(1)

print("\n✓ All marker imports successful!")