import numpy as np

def diou_nms_np(boxes, distance=0.2):
    """Implementing  diou non-maximum suppression in numpy
     Args:
       batch_boxes: detection boxes with shape (N, num, 4) and box format is [x1, y1, x2, y2].
       batch_scores:detection scores with shape (N, num_class).
     Returns:
        a list of numpy array: [boxes, scores, classes, num_valid].
     """
    boxes = np.array(boxes)
    max_box_num = boxes.shape[0]
    result_boxes = []

    #ondra added
    boxes = boxes[np.argsort((boxes[:, 2]-boxes[:, 0])*(boxes[:, 3]-boxes[:, 1]))]

    while boxes.shape[0] > 0:
        # ondra disabled
        #result_boxes.append(boxes[0])

        # computes intersection box size
        inter_wh = np.maximum(np.minimum(boxes[0, 2:4], boxes[1:, 2:4])-np.maximum(boxes[0, 0:2], boxes[1:, 0:2]),0)
        # intersection area
        inter_area = inter_wh[:, 0] * inter_wh[:, 1]

        # computes boxes area
        box1_wh = boxes[0, 2:4] - boxes[0, 0:2]
        box2_wh = boxes[1:, 2:4] - boxes[1:, 0:2]

        #iou_score = inter_area / (box1_wh[0] * box1_wh[1] + box2_wh[:, 0] * box2_wh[:, 1] - inter_area + 1e-7)
        center_dist = np.sum(np.square((boxes[0, 2:4] + boxes[0, 0:2]) / 2 - (boxes[1:, 2:4] + boxes[1:, 0:2]) / 2),
                             axis=-1)
        bounding_rect_wh = np.maximum(boxes[0, 2:4], boxes[1:, 2:4]) - np.minimum(boxes[0, 0:2], boxes[1:, 0:2])
        diagonal_dist = np.sum(np.square(bounding_rect_wh), axis=-1)
        # ondra disabled
        #diou = center_dist / diagonal_dist
        # ondra added
        foo = np.sum(np.square(box1_wh))
        diou = np.sum(np.square(box1_wh)) / diagonal_dist

        # print(diou)
        # ondra disabled
        # valid_mask = diou <= distance
        # boxes = boxes[1:][valid_mask]

        # ondra added
        bounding_bbox_right_bottom = np.maximum(boxes[0, 2:4], boxes[1:, 2:4])
        bounding_bbox_left_top = np.minimum(boxes[0, 0:2], boxes[1:, 0:2])

        # ondra changed to overpass threshold
        #valid_mask = diou <= distance
        valid_mask = diou > distance
        if sum(valid_mask) > 0:
            boxes[1:, 0:2][valid_mask] = bounding_bbox_left_top[valid_mask]
            boxes[1:, 2:4][valid_mask] = bounding_bbox_right_bottom[valid_mask]
            boxes = boxes[1:]
        else:
            result_boxes.append(boxes[0])
            boxes = boxes[1:]

    return result_boxes