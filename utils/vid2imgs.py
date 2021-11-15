import argparse
from typing import List
import cv2 as cv
import glob
from os import path
from pymediainfo import MediaInfo


def convert_vid2imgs(
    input_video_path: str, output_folder_path: str, img_names_prefix: str
) -> None:

    video_reader = cv.VideoCapture(input_video_path)

    has_frame, frame = video_reader.read()
    img_num = 0
    while has_frame:
        img_name = f"{img_names_prefix}_{img_num}.png"
        cv.imwrite(path.join(output_folder_path, img_name), frame)
        img_num += 1
        has_frame, frame = video_reader.read()
    video_reader.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Converts video into image frames. Frames are named after the video file name."
    )
    parser.add_argument(
        "-i",
        "--input-video",
        type=str,
        required=True,
        help="Absolute video file path.",
    )
    parser.add_argument(
        "-o", "--output-folder", type=str, required=True, help="Output folder path.",
    )

    args = parser.parse_args()
    _, video_file_name_with_ext = path.split(args.input_video)
    video_file_name, _ = path.splitext(video_file_name_with_ext)
    convert_vid2imgs(args.input_video, args.output_folder, video_file_name)
