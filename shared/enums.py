import enum

class ImageMode(enum.Enum):
    BGR = 1
    GRAYSCALE = 2

class RoiDetMethods(enum.Enum):
    DYNAMIC_THRESHOLDING = "DynamicThresholding"

    @staticmethod
    def get_choices():
        return [e.name.lower() for e in RoiDetMethods]

class PVDNSets(enum.Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"
