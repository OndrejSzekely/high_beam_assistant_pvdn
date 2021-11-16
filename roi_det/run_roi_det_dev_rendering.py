import argparse
from shutil import rmtree
import os
from os import path
import cv2 as cv
from shared.enums import RoiDetMethods
from shared.misc import visualize_bboxes
from roi_utils import get_roi_alg
from data_generator.plain_images_gen import PlainImageGen
from time import time


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Apply and render ROIs detection.")
    parser.add_argument("-i", "--input-folder", type=str, help="Input imgs folder.")
    parser.add_argument("-t", "--image-format", type=str, help="Images file format.")
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
        help="Output folder where renders are stored. FOLDER CONTENT IS REMOVED!",
    )

    args = parser.parse_args()

    rmtree(args.output_folder)
    os.mkdir(args.output_folder)

    roi_alg = get_roi_alg(RoiDetMethods[args.method.upper()])
    image_gen = PlainImageGen(args.image_format, args.input_folder, roi_alg.IMG_FORMAT)

    imgs_num = 0
    cum_time = 0
    for img_name, img in image_gen:
        start_time = time()
        preprocessed_img = roi_alg.preprocess(img)
        bboxes = roi_alg.compute(preprocessed_img)
        end_time = time()
        cum_time += end_time - start_time
        vized_roi_det_img = visualize_bboxes(img, bboxes)
        cv.imwrite(path.join(args.output_folder, img_name + ".jpg"), vized_roi_det_img)
        imgs_num += 1
    print(f"ROIs detection latency per image: {cum_time/imgs_num * 1000:.2f} ms")
    print(f"ROIs detection FPS: {imgs_num / cum_time:.2f} fps")
