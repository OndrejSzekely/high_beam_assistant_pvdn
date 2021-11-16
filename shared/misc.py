"""
Miscellaneous functions.
"""
import yaml
import numpy as np
import cv2 as cv
from typing import List, Tuple


def load_yaml(file_path: str) -> dict:
    with open(file_path, "r") as stream:
        return yaml.safe_load(stream)


def visualize_bboxes(img: np.ndarray, bboxes: List[Tuple[int, int, int, int]]) -> np.ndarray:
    img_to_draw = np.copy(img)
    if len(img_to_draw.shape) == 2:
        img_to_draw = cv.cvtColor(img_to_draw, cv.COLOR_GRAY2BGR)

    for bbox in bboxes:
        img_to_draw = cv.rectangle(img_to_draw, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0,0,255), 3)

    return img_to_draw