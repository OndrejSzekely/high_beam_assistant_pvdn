import abc
from abc import abstractmethod
import numpy as np
from typing import List, Tuple, Optional

class RoiDetBase(abc.ABC):

    @staticmethod
    @abstractmethod
    def load_from_yaml():
        pass

    @abstractmethod
    def compute(self, img: np.ndarray) -> Tuple[Optional[np.ndarray], List[Tuple[int, int, int, int]]]:
       pass

    @property
    def IMG_FORMAT(self):
        raise ValueError("Not defined")