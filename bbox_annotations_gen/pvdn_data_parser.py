from shared.enums import PVDNSets
from shared.constants import (
    PVDN_TYPES,
    PVDN_LABELS_FOLDER_NAME,
    PVDN_SEQUENCES_FILE_NAME,
    PVDN_KEYPOINTS_FOLDER_NAME,
)
from os import path
import json
from typing import List, Tuple, Generator
from roi_det.roi_base_class import RoiDetBase
from shapely.geometry import Polygon, Point
from shared.misc import convert_bboxes_repr
from data_generator.pvdn_images_access import PVDNOriginalImagesSequence
import random
import math

def _pvdn_sequences_generator(
    sequence_file_path: str,
) -> Generator[Tuple[str, List[str]], None, None]:
    with open(sequence_file_path, "r") as file:
        sequence_json = json.load(file)
        sequences = sequence_json["sequences"]
        for sequence in sequences:
            sequence_name = sequence["dir"]
            image_names = sequence["image_ids"]
            yield (sequence_name, image_names)

def _pvdn_sequences_samples_generator(
    sequence_file_path: str, fraction_rate: float, random_seed: int
) -> Generator[Tuple[str, List[str]], None, None]:
    with open(sequence_file_path, "r") as file:
        sequence_json = json.load(file)
        sequences = sequence_json["sequences"]
        for sequence in sequences:
            sequence_name = sequence["dir"]
            image_names = sequence["image_ids"]
            random.seed(sequence["id"] + random_seed)
            random.shuffle(image_names)
            samples_num = int(len(image_names)*fraction_rate)
            image_names = image_names[:samples_num]
            yield (sequence_name, image_names)


def _pvdn_get_image_annotations(image_path: str) -> List[Tuple[int, int]]:
    with open(image_path, "r") as file:
        image_annotations = json.load(file)
        keypoints_pos = []
        for annotation in image_annotations["annotations"]:
            keypoints_pos.append((annotation["pos"][0], annotation["pos"][1]))
        return keypoints_pos


def pvdn_annotations_generator(
    dataset_type: PVDNSets, data_path: str
) -> Generator[Tuple[str, str, str, List[Tuple[int, int]]], None, None]:
    for pvdn_type in PVDN_TYPES:
        type_path = path.join(data_path, pvdn_type)
        dataset_path = path.join(type_path, dataset_type.value)
        labels_path = path.join(dataset_path, PVDN_LABELS_FOLDER_NAME)
        sequences_path = path.join(labels_path, PVDN_SEQUENCES_FILE_NAME)
        keypoints_path = path.join(labels_path, PVDN_KEYPOINTS_FOLDER_NAME)
        sequences_gen = _pvdn_sequences_generator(sequences_path)
        for sequence_name, image_names in sequences_gen:
            for image_name in image_names:
                image_name = str(image_name)
                zeros_prefix_num = 6 - len(image_name)
                prefix = "0" * zeros_prefix_num
                image_name = prefix + image_name
                image_annotations = _pvdn_get_image_annotations(
                    path.join(keypoints_path, image_name + ".json")
                )

                yield (pvdn_type, sequence_name, image_name, image_annotations)


def pvdn_samples_annotations_generator(
    dataset_type: PVDNSets, data_path: str, fraction_rate: float, random_seed: int
) -> Generator[Tuple[str, str, str, List[Tuple[int, int]]], None, None]:
    for pvdn_type in PVDN_TYPES:
        type_path = path.join(data_path, pvdn_type)
        dataset_path = path.join(type_path, dataset_type.value)
        labels_path = path.join(dataset_path, PVDN_LABELS_FOLDER_NAME)
        sequences_path = path.join(labels_path, PVDN_SEQUENCES_FILE_NAME)
        keypoints_path = path.join(labels_path, PVDN_KEYPOINTS_FOLDER_NAME)
        sequences_gen = _pvdn_sequences_samples_generator(sequences_path, fraction_rate, random_seed)
        for sequence_name, image_names in sequences_gen:
            for image_name in image_names:
                image_name = str(image_name)
                zeros_prefix_num = 6 - len(image_name)
                prefix = "0" * zeros_prefix_num
                image_name = prefix + image_name
                image_annotations = _pvdn_get_image_annotations(
                    path.join(keypoints_path, image_name + ".json")
                )

                yield (pvdn_type, sequence_name, image_name, image_annotations)


def match_boxes_and_annotations(
    bboxes: List[Tuple[int, int, int, int]], annotations: List[Tuple[int, int]]
) -> Tuple[List[int], int, int, int]:
    tps = 0
    fps = 0
    fns = 0
    q_k_partial = 0
    q_b_partial = 0
    described_keypoints_num = 0
    if not annotations:
        return (
            [0] * len(bboxes),
            tps,
            len(bboxes),
            fns,
            described_keypoints_num,
            q_k_partial,
            q_b_partial,
        )

    bboxes_points = convert_bboxes_repr(bboxes)
    labels = []
    bboxes_geo = [Polygon(bbox) for bbox in bboxes_points]
    keypoints_geo = [Point(annotation) for annotation in annotations]
    for bbox_geo in bboxes_geo:
        intersects = 0
        for keypoint_geo in keypoints_geo:
            intersects += int(bbox_geo.intersects(keypoint_geo))
        label = int(intersects > 0)
        labels.append(label)
        tps += label
        fps += 1 - label
        if label:
            q_k_partial += 1.0 / intersects

    for keypoint_geo in keypoints_geo:
        intersects = 0
        for bbox_geo in bboxes_geo:
            intersects += int(keypoint_geo.intersects(bbox_geo))
        fns += 1 - int(intersects > 0)
        if intersects:
            q_b_partial += 1.0 / intersects
            described_keypoints_num += 1

    return labels, tps, fps, fns, described_keypoints_num, q_k_partial, q_b_partial


class PVDNBboxAnnotationsGenerator:
    def __init__(self, dataset_type: PVDNSets, data_path: str, roi_alg: RoiDetBase):
        self.dataset_type = dataset_type
        self.data_path = data_path
        self.roi_alg = roi_alg
        self.images_operator = PVDNOriginalImagesSequence(dataset_type, data_path)

    def __iter__(self):
        self._annotations_gen = pvdn_annotations_generator(
            self.dataset_type, self.data_path
        )
        return self

    def __next__(self):
        pvdn_type, sequence_name, image_name, image_annotations = next(
            self._annotations_gen
        )
        img = self.images_operator[(pvdn_type, sequence_name, image_name)]
        _, bboxes = self.roi_alg.compute(img)



        # enlarge bboxes annotations
        enlarged_bboxes = []
        for bbox in bboxes:
            try:
                bbox_margin_x = int(20 * math.exp(10 / (bbox[2] - bbox[0])))
                bbox_margin_y = int(20 * math.exp(10 / (bbox[3] - bbox[1])))
            except Exception:
                continue
            margin = min(bbox_margin_y, bbox_margin_x)
            x_min = max(0, bbox[0] - margin)
            y_min = max(0, bbox[1] - margin)
            x_max = min(1280, bbox[2] + margin)
            y_max = min(960, bbox[3] + margin)
            enlarged_bboxes.append((x_min, y_min, x_max, y_max))
        bboxes = enlarged_bboxes



        (
            labels,
            tps,
            fps,
            fns,
            keypoints_num,
            q_k_partial,
            q_b_partial,
        ) = match_boxes_and_annotations(bboxes, image_annotations)
        return (
            pvdn_type,
            sequence_name,
            image_name,
            bboxes,
            labels,
            tps,
            fps,
            fns,
            keypoints_num,
            q_k_partial,
            q_b_partial,
        )


class PVDNBboxFractionSamplesAnnotationsGenerator:
    def __init__(self, dataset_type: PVDNSets, data_path: str, roi_alg: RoiDetBase, fraction_rate: float, random_seed: int):
        self.dataset_type = dataset_type
        self.data_path = data_path
        self.roi_alg = roi_alg
        self.fraction_rate = fraction_rate
        self.random_seed = random_seed
        self.images_operator = PVDNOriginalImagesSequence(dataset_type, data_path)

    def __iter__(self):
        self._annotations_gen = pvdn_samples_annotations_generator(
            self.dataset_type, self.data_path, self.fraction_rate, self.random_seed
        )
        return self

    def __next__(self):
        pvdn_type, sequence_name, image_name, image_annotations = next(
            self._annotations_gen
        )
        img = self.images_operator[(pvdn_type, sequence_name, image_name)]
        _, bboxes = self.roi_alg.compute(img)
        (
            labels,
            tps,
            fps,
            fns,
            keypoints_num,
            q_k_partial,
            q_b_partial,
        ) = match_boxes_and_annotations(bboxes, image_annotations)
        return (
            sequence_name,
            image_name,
            bboxes,
            labels,
            tps,
            fps,
            fns,
            keypoints_num,
            q_k_partial,
            q_b_partial,
        )


class PVDNBboxMetrics:
    def __init__(self):
        self.total_tps = 0
        self.total_fps = 0
        self.total_fns = 0
        self.total_described_keypoints_num = 0
        self.sum_q_k = 0
        self.sum_q_b = 0

    def update(
        self,
        tps,
        fps,
        fns,
        described_keypoints_num,
        q_k_partial,
        q_b_partial,
    ):
        self.total_tps += tps
        self.total_fps += fps
        self.total_fns += fns
        self.total_described_keypoints_num += described_keypoints_num
        self.sum_q_b += q_b_partial
        self.sum_q_k += q_k_partial

    def get_metrics(self):
        precision = self.total_tps / (self.total_tps + self.total_fps + 1e-10)
        recall = self.total_tps / (self.total_tps + self.total_fns + 1e-10)
        f1 = 2.0 * precision * recall / (precision + recall + 1e-10)
        q_k = 1.0 / (self.total_tps + 1e-10) * self.sum_q_k
        q_b = 1.0 / (self.total_described_keypoints_num + 1e-10) * self.sum_q_b
        q = q_k * q_b

        return precision, recall, f1, q_k, q_b, q
