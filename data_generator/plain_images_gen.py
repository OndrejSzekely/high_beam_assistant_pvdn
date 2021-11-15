"""
Plain image data generator. It takes recursively all images in the given folder hierarchy.
"""

import glob
from os import path
from shared.enums import ImageMode
import cv2 as cv


class PlainImageGen:
    """
    Plain image data generator. It takes recursively all images in the given folder hierarchy.
    """

    def __init__(self, img_format: str, imgs_folder: str, mode: ImageMode) -> None:
        self.imgs_folder = imgs_folder
        self.img_format = img_format
        self._set_image_mode(mode)
        glob_pattern = path.join(self.imgs_folder, "**", f"*.{self.img_format}")
        self.img_iter = glob.glob(glob_pattern)

    def _set_image_mode(self, mode: ImageMode):
        self_mode = None
        if mode == ImageMode.BGR:
            self_mode = cv.IMREREAD_BGR
        elif mode == ImageMode.GRAYSCALE:
            self_mode = cv.IMREAD_GRAYSCALE
        self.mode = self_mode

    def __iter__(self):
        return self

    def __next__(self):
        img_path = next(self.img_iter)
        return cv.imread(img_path, self.mode)
