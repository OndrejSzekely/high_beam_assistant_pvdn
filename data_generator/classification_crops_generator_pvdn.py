import argparse
from shutil import rmtree
from os import path
import os
import json
from shared.enums import PVDNSets
from data_generator.pvdn_images_access import PVDNOriginalImagesSequence
import cv2 as cv
import math

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Crops extraction for binary classifier of PVDN dataset.")

    parser.add_argument("--data-path", type=str, help="Path to root folder of PVDN dataset.")
    parser.add_argument("--training-json", type=str, help="Path to training dataset path.")
    parser.add_argument("--testing-json", type=str, help="Path to testing dataset path.")
    parser.add_argument("--validation-json", type=str, help="Path to validation dataset path.")
    parser.add_argument("--output-folder", type=str, help="Path to output folder. CONTENT WILL BE REMOVED.")
    parser.add_argument(
        "-s",
        "--bbox-margin-scaling-factor",
        type=int,
        default=0,
        help="Bbox margin scaling factor.",
    )

    args = parser.parse_args()

    if path.exists(args.output_folder):
        rmtree(args.output_folder)
        os.mkdir(args.output_folder)

    json_paths = [args.training_json, args.validation_json, args.testing_json]
    sets = [PVDNSets.TRAIN, PVDNSets.VAL, PVDNSets.TEST]

    for json_path, set_type in zip(json_paths, sets):
        with open(json_path, "r") as json_file:
            images_operator = PVDNOriginalImagesSequence(set_type, args.data_path)
            annotations = json.load(json_file)

            output_path = path.join(args.output_folder, set_type.value)
            os.mkdir(output_path)
            car_path = path.join(output_path, "car")
            artifacts_path = path.join(output_path, "artifacts")
            os.mkdir(car_path)
            os.mkdir(artifacts_path)

            crop_ind = 0
            for annotation in annotations:
                img = images_operator[(annotation["type"], annotation["sequence_name"], annotation["image_name"])]
                img_h, img_w = img.shape[0], img.shape[1]
                for bbox, label in zip(annotation["bboxes"], annotation["labels"]):
                    try:
                        bbox_margin_x = int(1 / math.sqrt(bbox[2]-bbox[0]) * args.bbox_margin_scaling_factor)
                        bbox_margin_y = int(1 / math.sqrt(bbox[3]-bbox[1]) * args.bbox_margin_scaling_factor)
                    except Exception:
                        foo = 5
                    x_min = max(0, bbox[0] - bbox_margin_x)
                    y_min = max(0, bbox[1] - bbox_margin_y)
                    x_max = min(img_w, bbox[2] + bbox_margin_x)
                    y_max = min(img_h, bbox[3] + bbox_margin_y)
                    crop = img[y_min:y_max, x_min:x_max]
                    img_name = f"crop_{annotation['sequence_name']}_{annotation['image_name']}_{crop_ind}.png"
                    if label == 0:
                        crop_path = path.join(artifacts_path, img_name)
                        cv.imwrite(crop_path, crop)
                    else:
                        crop_path = path.join(car_path, img_name)
                        cv.imwrite(crop_path, crop)
                    crop_ind += 1


