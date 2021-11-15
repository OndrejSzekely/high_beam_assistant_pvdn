import argparse
from shutil import rmtree
import os
from shared.enums import BboxDetMethods
from data_generator.plain_images_gen import PlainImageGen


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Apply and render bbox detection.")
    parser.add_argument("-i", "--input-folder", type=str, help="Input imgs folder.")
    parser.add_argument("-t", "--image_format", type=str, help="Images file format.")
    parser.add_argument("-m", "--method", choices=BboxDetMethods.get_choices(), help="Bbox detection algorithm.")
    parser.add_argument("-o", "--output-folder", type=str, help="Output folder where renders are stored. FOLDER CONTENT IS REMOVED!")

    args = parser.parse_args()

    rmtree(args.output_folder)
    os.mkdir(args.output_folder)

    bbox_alg = BboxDetMethods[args.method.upper]()

    image_gen = PlainImageGen(args.image_format, args.output_folder, bbox_alg.IMG_FORMAT)