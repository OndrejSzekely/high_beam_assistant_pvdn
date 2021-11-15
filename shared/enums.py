import enum
from bbox_det.dynamic_thresholding import DynamicThresholding

class ImageMode(enum.Enum):
    BGR = 1
    GRAYSCALE = 2

class BboxDetMethods(enum.Enum):
    DYNAMIC_THRESHOLDING = DynamicThresholding

    @staticmethod
    def get_choices():
        return [e.name.lower() for e in BboxDetMethods]