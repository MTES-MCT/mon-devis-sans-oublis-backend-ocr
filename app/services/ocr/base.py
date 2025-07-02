from abc import ABC, abstractmethod
from typing import List
from PIL import Image

class BaseOCRService(ABC):
    
    _service_name = "base"

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def process_images(self, images: List[Image.Image]) -> List[str]:
        """
        Process a list of PIL images and return a list of extracted text.
        """
        pass