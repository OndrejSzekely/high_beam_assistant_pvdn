from roi_det.distance_based_merging import distance_based_merging
import cv2 as cv
from shared.misc import visualize_bboxes
import numpy as np

test_bboxes = [
    [100, 100, 190, 190],
    [70, 140, 120, 190],
    [220, 120, 250, 145]
]

pruned_boxes = distance_based_merging(test_bboxes, 40)

img = np.zeros((300, 300), dtype=np.uint8)
vis_img_orig = visualize_bboxes(img, test_bboxes)
vis_img_pruned = visualize_bboxes(img, pruned_boxes)
cv.imwrite("/digiteq/original_bbox.png", vis_img_orig)
cv.imwrite("/digiteq/original_bbox_pruned.png", vis_img_pruned)