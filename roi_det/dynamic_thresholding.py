import numpy as np
import cv2 as cv
from typing import Final, List, Tuple, Optional
from roi_det.roi_base_class import RoiDetBase
from shared.enums import ImageMode
from shared.misc import load_yaml
from roi_det.distance_based_merging import distance_based_merging

class DynamicThresholding(RoiDetBase):
    """
    Applies dynamic thresholding described in the paper: https://arxiv.org/pdf/2107.11302.pdf
    """

    IMG_FORMAT: Final[str] = ImageMode.GRAYSCALE

    def __init__(
        self,
        blur_kernel_size,
        local_intensity_win,
        dynamic_threshold_factor,
        opening_kernel,
        margin_distance,
        minimal_roi_mean_abs_dev,
        processing_scaling_factor
    ):
        self.blur_kernel_size = blur_kernel_size
        self.local_intensity_win = local_intensity_win
        self.dynamic_threshold_factor = dynamic_threshold_factor
        self.opening_kernel = opening_kernel
        self.margin_distance = margin_distance
        self.minimal_roi_mean_abs_dev = minimal_roi_mean_abs_dev
        self.processing_scaling_factor = processing_scaling_factor


    @staticmethod
    def load_from_yaml():
        return DynamicThresholding(
            **load_yaml("../alg_configs/roi_det_dynamic_threshold.yaml")
        )

    def compute(self, img: np.ndarray) -> Tuple[Optional[np.ndarray], List[Tuple[int, int, int, int]]]:
        """
        Extracts bboxes from preprocessed image and prepreocessed image for rendering

        Return: List[x_min, y_min, x_max, y_max]
        """
        img_preprocesed = cv.resize(
            img,
            (0, 0),
            fx=self.processing_scaling_factor,
            fy=self.processing_scaling_factor,
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
        #binary_img = cv.morphologyEx(
        #    binary_img,
        #    cv.MORPH_OPEN,
        #    np.ones((self.opening_kernel, self.opening_kernel)),
        #)

        # extract bboxes
        contours, _ = cv.findContours(binary_img, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
        bboxes = []
        for contour in contours:
            rect = cv.minAreaRect(contour)
            box_scaled = cv.boxPoints(rect)
            box_orig = np.int0(box_scaled * 1 / self.processing_scaling_factor)
            box_scaled = np.int0(box_scaled)
            if (box_orig[2][0] - box_orig[0][0]) * (box_orig[2][1] - box_orig[0][1]) > 0:

                # filter out boxes where mean absolute deviation is smaller then the parameter
                roi = img_preprocesed[box_scaled[0][1]:box_scaled[2][1], box_scaled[0][0]:box_scaled[2][0]]

                roi_mean = np.median(roi)
                roi_mad = np.mean(np.abs(roi - roi_mean))
                if roi_mad > self.minimal_roi_mean_abs_dev:
                    bboxes.append((box_orig[0][0].item(), box_orig[0][1].item(), box_orig[2][0].item(), box_orig[2][1].item()))

        #distance based filtering
        bboxes = distance_based_merging(bboxes, self.margin_distance)
        return (_, bboxes)
