from shared.enums import PVDNSets
from shared.constants import PVDN_TYPES, PVDN_IMAGES_FOLDER_NAME
from os import path
import glob
import cv2 as cv


class PVDNOriginalImagesSequence:
    def __init__(self, dataset_type: PVDNSets, data_path: str):
        self.dataset_type = dataset_type
        self.data_path = data_path

    def __len__(self):
        imgs_num = 0
        for pvdn_type in PVDN_TYPES:
            type_path = path.join(self.data_path, pvdn_type)
            images_path = path.join(
                type_path, self.dataset_type.value, PVDN_IMAGES_FOLDER_NAME
            )
            imgs_num += len(
                glob.glob(path.join(images_path, "**", "*.png"), recursive=True)
            )
        return imgs_num

    def __getitem__(self, item):
        img_path = path.join(self.data_path, item[0], self.dataset_type.value, "images", item[1], item[2]+".png")
        return cv.imread(img_path, cv.IMREAD_GRAYSCALE)
