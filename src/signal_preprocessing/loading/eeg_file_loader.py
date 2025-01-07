from abc import ABC, abstractmethod
from pathlib import Path

class EEGFileLoader(ABC):
    """
    Abstract base class for EEG file loaders.
    """

    def __init__(self, file_path: str):
        self.file_path = Path(file_path)

    @abstractmethod
    def load(self) -> dict:
        """
        Load and process the EEG file.
        """
        pass
