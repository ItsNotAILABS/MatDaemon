from abc import ABC, abstractmethod
import numpy as np


class BaseBackend(ABC):
    @abstractmethod
    def matmul(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        pass
