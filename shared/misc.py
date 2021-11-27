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

def convert_bboxes_repr(bboxes: List[Tuple[int, int, int, int]]) -> List[List[Tuple[int, int]]]:
    """
    Converts (x_min, y_min, x_max, y_max) repr into list of points [(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)]
    """
    if len(bboxes) == 0:
        return bboxes
    bboxes_array = np.array(bboxes)
    point_1 = np.expand_dims(np.array([bboxes_array[:,0], bboxes_array[:,1]]).T, 1)
    point_2 = np.expand_dims(np.array([bboxes_array[:, 2], bboxes_array[:, 1]]).T, 1)
    point_3 = np.expand_dims(np.array([bboxes_array[:, 2], bboxes_array[:, 3]]).T, 1)
    point_4 = np.expand_dims(np.array([bboxes_array[:, 0], bboxes_array[:, 3]]).T, 1)
    boxes_points = np.concatenate((point_1, point_2, point_3, point_4), axis=1)
    boxes_points = list(map(lambda box: list(map(tuple, box)), boxes_points))
    return boxes_points