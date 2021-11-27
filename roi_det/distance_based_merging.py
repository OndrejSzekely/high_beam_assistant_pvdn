from shapely.geometry import Polygon
from shared.misc import convert_bboxes_repr
import numpy as np
result_boxes = []

def _generate_polygons(bboxes):
    list_of_polygons = []
    boxes_points = convert_bboxes_repr(bboxes)
    for box_points in boxes_points:
        list_of_polygons.append(Polygon(box_points))
    return list_of_polygons

def distance_based_merging(bboxes, maximal_distance):
    bboxes = np.array(bboxes)
    if bboxes.shape[0] == 1:
        return [tuple(bboxes[0].tolist())]
    list_of_polygons = _generate_polygons(bboxes)
    result_boxes = []
    while bboxes.shape[0] > 0:
        distances = np.zeros(bboxes.shape[0]-1, dtype=float)
        for ind in range(1, bboxes.shape[0]):
            distances[ind-1] = list_of_polygons[0].distance(list_of_polygons[ind])

        bounding_bbox_right_bottom = np.maximum(bboxes[0, 2:4], bboxes[1:, 2:4])
        bounding_bbox_left_top = np.minimum(bboxes[0, 0:2], bboxes[1:, 0:2])

        valid_mask = distances < maximal_distance
        if sum(valid_mask) > 0:
            bboxes[1:, 0:2][valid_mask] = bounding_bbox_left_top[valid_mask]
            bboxes[1:, 2:4][valid_mask] = bounding_bbox_right_bottom[valid_mask]
            bboxes = bboxes[1:]
            list_of_polygons = _generate_polygons(bboxes)
        else:
            result_boxes.append(tuple(bboxes[0].tolist()))
            bboxes = bboxes[1:]
            list_of_polygons = _generate_polygons(bboxes)
    return result_boxes