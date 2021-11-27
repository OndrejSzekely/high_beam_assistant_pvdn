import sys
from roi_det.dynamic_thresholding import DynamicThresholding
from roi_det.roi_base_class import RoiDetBase
from shared.enums import RoiDetMethods

def get_roi_alg(roi_alg: RoiDetMethods) -> RoiDetBase:
    """
    Instantiates roi alg class from the enum choice.
    """
    roi_alg_class_ = getattr(sys.modules[__name__], roi_alg.value)
    return roi_alg_class_.load_from_yaml()