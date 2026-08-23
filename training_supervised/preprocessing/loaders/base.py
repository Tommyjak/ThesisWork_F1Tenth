from abc import ABC, abstractmethod
from pathlib import Path

class Loader(ABC):
    @abstractmethod
    def __init__(self, root: Path):
        self.root = root

    @abstractmethod
    def find_samples(self):
        ...