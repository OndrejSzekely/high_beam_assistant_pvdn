import abc
from abc import abstractmethod
import numpy as np

class BboxDetBase(abc.ABC):

    @abstractmethod
    def compute_and_visualize_bboxes(self, img: np.ndarray) -> np.ndarray:
        pass

    @property
    def IMG_FORMAT(self):
        raise ValueError("Not defined")