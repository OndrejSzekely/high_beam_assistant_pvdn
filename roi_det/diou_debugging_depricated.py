from roi_det.diou_nms_depricated import diou_nms_np
import cv2 as cv
from shared.misc import visualize_bboxes
import numpy as np

test_bboxes = [
    [100, 100, 190, 190],
    [70, 140, 120, 190],
    [220, 120, 250, 145]
]

pruned_boxes = diou_nms_np(test_bboxes)

img = np.zeros((300, 300), dtype=np.uint8)
vis_img_orig = visualize_bboxes(img, test_bboxes)
vis_img_pruned = visualize_bboxes(img, pruned_boxes)
cv.imwrite("/digiteq/original_bbox.png", vis_img_orig)
cv.imwrite("/digiteq/original_bbox_pruned.png", vis_img_pruned)