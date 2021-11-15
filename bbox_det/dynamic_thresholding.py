import numpy as np
import cv2 as cv
from typing import Final
from bbox_base_class import BboxDetBase
from shared.enums import ImageMode


class DynamicThresholding(BboxDetBase):

    IMG_FORMAT: Final[str] = ImageMode.GRAYSCALE

    def compute_and_visualize_bboxes(self, img: np.ndarray) -> np.ndarray:
        """
        Applies dynamic thresholding described in the paper: https://arxiv.org/pdf/2107.11302.pdf

        Args:
            img (np.ndarray): Grayscale image with shape (cols, rows).

        Returns:

        """
        return img.T