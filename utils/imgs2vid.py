import argparse
from typing import List
import cv2 as cv
import glob
from os import path
from pymediainfo import MediaInfo


def convert_imgs2vid(
    image_names_sequence: List[str], output_file_path: str, fps: int
) -> None:
    media_info = MediaInfo.parse(image_names_sequence[0])
    image_track = media_info.image_tracks[0]

    video_writer = cv.VideoWriter(
        output_file_path + ".avi",
        cv.VideoWriter_fourcc("M", "J", "P", "G"),
        fps,
        (image_track.width, image_track.height),
    )

    for img_path in image_names_sequence:
        img = cv.imread(img_path)
        video_writer.write(img)
    video_writer.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Converts image sequence to video based on lexicographical order of image files."
    )
    parser.add_argument(
        "-i",
        "--input_seq",
        type=str,
        required=True,
        help="Folder path containing images.",
    )
    parser.add_argument(
        "-fps", "--fps", type=int, required=False, default=15, help="Output video FPS."
    )
    parser.add_argument(
        "-o",
        "--output_folder",
        type=str,
        required=True,
        help="Absolute output folder path. Generated video file has the same name as input folder name.",
    )

    args = parser.parse_args()
    _, video_file_name = path.split(args.input_seq)
    output_file_name = path.join(args.output_folder, video_file_name)
    img_paths = glob.glob(path.join(args.input_seq, "*.*"))
    convert_imgs2vid(sorted(img_paths, key=str.lower), output_file_name, args.fps)
