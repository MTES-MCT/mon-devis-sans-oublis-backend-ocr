from abc import ABC, abstractmethod

class BaseOCRService(ABC):

    @abstractmethod
    async def process_file(self, file_path: str, filename: str):
        pass