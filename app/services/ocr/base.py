from abc import ABC, abstractmethod
from typing import List, Literal

from PIL import Image

OCRInputType = Literal["images", "pdf"]


class BaseOCRService(ABC):
    """Base interface for OCR services.

    Services can be either:
    - image-based: implement `process_images()` (default)
    - PDF-native: override `preferred_input_type()` to return "pdf" and implement
      `process_pdf_file()`.
    """

    _service_name = "base"

    def __init__(self) -> None:
        super().__init__()

    def warmup(self) -> None:
        """Optional warmup hook.

        Used during Docker image build to pre-download / initialize models.
        Default is a no-op.
        """
        return

    def preferred_input_type(self, file_extension: str) -> OCRInputType:
        """Return the preferred input type for the given uploaded file extension.

        By default, services expect images (VLLM services, and marker when the
        uploaded file is an image).

        Services that can process PDFs directly (e.g. Marker) can override this
        and return "pdf" when `file_extension == ".pdf"`.
        """
        return "images"

    def process_pdf_file(self, pdf_path: str) -> str:
        """Process a PDF file stored on disk and return extracted text.

        Optional capability. Services that return "pdf" in
        `preferred_input_type()` must override this.
        """
        raise NotImplementedError(
            f"{self._service_name} does not support direct PDF processing"
        )

    @abstractmethod
    def process_images(self, images: List[Image.Image]) -> List[str]:
        """Process a list of PIL images and return a list of extracted text."""
        pass