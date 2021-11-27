import glob
from typing import List, Tuple
from os import path
import numpy as np
import cv2 as cv
import math


def get_folder_sequences(sequences_folder_path: str) -> List[str]:
    sequences_list = glob.glob(path.join(sequences_folder_path, "*/"))
    sequences_list = list(filter(lambda path_rec: path.isdir(path_rec), sequences_list))
    return sequences_list


class GetImagesIterator:
    def __init__(self, sequence_path: str):
        self.sequence_path = sequence_path

    def __iter__(self):
        self.images_iter = iter(
            sorted(glob.glob(path.join(self.sequence_path, "*.png")))
        )
        return self

    def __next__(self):
        return cv.imread(next(self.images_iter), cv.IMREAD_COLOR)


def extract_rois(
    img: np.ndarray,
    boxes: List[Tuple[int, int, int, int]],
    resizing_size: int,
    depth: int,
    margin_scaling_factor: int,
) -> List[np.ndarray]:
    images_batch = np.zeros(
        (len(boxes), resizing_size, resizing_size, depth), dtype=np.uint8
    )
    img_w, img_h = img.shape[1], img.shape[0]
    for ind, box in enumerate(boxes):
        bbox_margin_x = int(20 * math.exp(10 / (box[2] - box[0])))
        bbox_margin_y = int(20 * math.exp(10 / (box[3] - box[1])))
        margin = min(bbox_margin_y, bbox_margin_x)
        x_min = max(0, box[0] - margin)
        y_min = max(0, box[1] - margin)
        x_max = min(img_w, box[2] + margin)
        y_max = min(img_h, box[3] + margin)
        roi_crop = img[y_min:y_max, x_min:x_max]
        resized_crop = cv.resize(roi_crop, (resizing_size, resizing_size))
        images_batch[ind, :, :] = resized_crop
    return images_batch

def visualize_inference_res(bboxes: List[Tuple[int, int, int, int]], img: np.ndarray) -> np.ndarray:
    img_height, img_width = img.shape[0], img.shape[1]
    beam_img = np.ones((img_height, img_width, 3), dtype=np.uint8) * 255
    for bbox in bboxes:
        beam_img[:, bbox[0]:bbox[2]] = 0
    return cv.addWeighted(img, 0.8, beam_img, 0.2, 0)
