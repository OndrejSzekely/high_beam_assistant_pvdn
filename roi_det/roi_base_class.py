import abc
from abc import abstractmethod
import numpy as np
from typing import List, Tuple

class RoiDetBase(abc.ABC):

    @abstractmethod
    def preprocess(self, img: np.ndarray) -> np.ndarray:
        pass

    @staticmethod
    @abstractmethod
    def load_from_yaml():
        pass

    @abstractmethod
    def compute(self, img: np.ndarray) -> List[Tuple[int, int, int, int]]:
       pass

    @property
    def IMG_FORMAT(self):
        raise ValueError("Not defined")