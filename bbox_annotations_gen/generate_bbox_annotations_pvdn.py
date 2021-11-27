import argparse
from shared.enums import RoiDetMethods
from shared.enums import PVDNSets
from shared.dataclasses import MetricsContainer
from pvdn_data_parser import PVDNBboxAnnotationsGenerator, PVDNBboxMetrics
from roi_det.roi_utils import get_roi_alg
from os import path
import os
import json
from time import time


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Generate bbox annotations from PVDN dataset.")
    parser.add_argument(
        "-d",
        "--dataset-folder",
        type=str,
        help="Path to PVDN root folder containing 'day' and 'night' folders.",
    )
    parser.add_argument(
        "-m",
        "--method",
        choices=RoiDetMethods.get_choices(),
        help="ROI detection algorithm.",
    )
    parser.add_argument(
        "-o",
        "--output-folder",
        type=str,
        help="Output folder where JSON annotations will be stored.",
    )

    args = parser.parse_args()
    roi_alg = get_roi_alg(RoiDetMethods[args.method.upper()])

    start_time = time()
    for pvdn_set in PVDNSets:
        json_file_path = path.join(
            args.output_folder, f"{pvdn_set.value}_bboxes_annotation.json"
        )
        if path.exists(json_file_path):
            os.remove(json_file_path)

        pvdn_bbox_metrics_op = PVDNBboxMetrics()
        with open(json_file_path, "w") as json_file:
            json_output_annotations = []

            training_bboxes = PVDNBboxAnnotationsGenerator(
                pvdn_set, args.dataset_folder, roi_alg
            )
            for (
                type,
                sequence_name,
                image_name,
                bboxes,
                labels,
                tps,
                fps,
                fns,
                described_keypoints_num,
                q_k_partial,
                q_b_partial,
            ) in training_bboxes:
                record = {
                    "set_type": pvdn_set.value,
                    "type": type,
                    "sequence_name": sequence_name,
                    "image_name": image_name,
                    "bboxes": bboxes,
                    "labels": labels,
                }
                json_output_annotations.append(record)
                pvdn_bbox_metrics_op.update(
                    tps, fps, fns, described_keypoints_num, q_k_partial, q_b_partial
                )

            metrics = MetricsContainer(*pvdn_bbox_metrics_op.get_metrics())
            print(f"Dataset {pvdn_set.name}:")
            metrics.print_metrics()
            print("\n")

            json.dump(json_output_annotations, json_file)
    end_time = time()
    print(f"Preprocessing duration: {(end_time-start_time)/60:.2f} min")
