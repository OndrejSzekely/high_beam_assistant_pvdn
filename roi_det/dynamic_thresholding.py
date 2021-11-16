import numpy as np
import cv2 as cv
from typing import Final, List, Tuple
from roi_base_class import RoiDetBase
from shared.enums import ImageMode
from shared.misc import load_yaml


class DynamicThresholding(RoiDetBase):
    """
    Applies dynamic thresholding described in the paper: https://arxiv.org/pdf/2107.11302.pdf
    """

    IMG_FORMAT: Final[str] = ImageMode.GRAYSCALE
    img_processing_scaling_factor: Final[int] = 0.5

    def __init__(
        self,
        blur_kernel_size,
        local_intensity_win,
        dynamic_threshold_factor,
        opening_kernel_h,
        opening_kernel_w,
        dilatation_distance,
    ):
        self.blur_kernel_size = blur_kernel_size
        self.local_intensity_win = local_intensity_win
        self.dynamic_threshold_factor = dynamic_threshold_factor
        self.opening_kernel_h = opening_kernel_h
        self.opening_kernel_w = opening_kernel_w
        self.dilatation_distance = dilatation_distance

    @staticmethod
    def load_from_yaml():
        return DynamicThresholding(
            **load_yaml("../alg_configs/roi_det_dynamic_threshold.yaml")
        )

    def preprocess(self, img: np.ndarray) -> np.ndarray:
        img_preprocesed = cv.resize(
            img,
            (0, 0),
            fx=self.img_processing_scaling_factor,
            fy=self.img_processing_scaling_factor,
        )
        img_preprocesed = cv.GaussianBlur(
            img_preprocesed,
            (
                self.blur_kernel_size,
                self.blur_kernel_size,
            ),
            2,
        )
        img_preprocesed = img_preprocesed.astype(np.float) / 255.0

        # dynamic thresholding
        neighbourhood_means_kernel_norm_const = 1.0 / (
            self.local_intensity_win * self.local_intensity_win
        )
        neighbourhood_means_kernel = (
            np.ones(
                (
                    self.local_intensity_win,
                    self.local_intensity_win,
                )
            )
            * neighbourhood_means_kernel_norm_const
        )
        neighbourhood_means = cv.filter2D(
            src=img_preprocesed, ddepth=-1, kernel=neighbourhood_means_kernel
        )
        deltas = img_preprocesed - neighbourhood_means
        thresholds = neighbourhood_means * (
            1.0 + self.dynamic_threshold_factor * (1.0 - deltas / (1.0 - deltas + 1e-5))
        )
        binary_img = img_preprocesed > thresholds
        binary_img = binary_img.astype(np.uint8)

        # remove blob noise
        binary_img = cv.morphologyEx(
            binary_img,
            cv.MORPH_OPEN,
            np.ones((self.opening_kernel_h, self.opening_kernel_w)),
        )

        # connect disconnected components
        binary_img = cv.dilate(
            binary_img, np.ones((self.dilatation_distance, self.dilatation_distance))
        )

        return binary_img

    def compute(self, img: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Extracts bboxes from preprocessed image.

        Return: List[x_min, y_min, x_max, y_max]
        """
        contours, _ = cv.findContours(img, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
        bboxes = []
        for contour in contours:
            rect = cv.minAreaRect(contour)
            box = cv.boxPoints(rect) * 1 / self.img_processing_scaling_factor
            box = np.int0(box)
            bboxes.append((box[0][0], box[0][1], box[2][0], box[2][1]))
        return bboxes
