import argparse
from inference.inference_utils import *
from time import time
from roi_det.dynamic_thresholding import DynamicThresholding
import tensorflow as tf
from roi_det.distance_based_merging import distance_based_merging

ROI_INFERENCE_RESOLUTION = 128
ROI_INFERENCE_DEPTH = 3
MARGIN_SCALING_FACTOR = 90
BBOXES_MERGING_DISTANCE = 100

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Video sequences inference.")
    parser.add_argument("--sequences-folder", type=str, help="Runs inference across all sequences in the PVDN folder.")
    parser.add_argument("--output-folder", type=str, help="Output folder where rendered sequences are stored.")
    parser.add_argument("--model-path", type=str, help="ROI TF classification model path.")

    args = parser.parse_args()
    _, model_prefix = path.split(args.model_path)

    roi_alg = DynamicThresholding.load_from_yaml()
    roi_inference_alg = tf.keras.models.load_model(args.model_path)

    sequences = get_folder_sequences(args.sequences_folder)
    img_count = 0
    cum_inference_time = 0
    for sequence_path in sequences:
        _, sequence_name = path.split(sequence_path[:-1])
        video_writer = cv.VideoWriter(
            path.join(args.output_folder, f"{sequence_name}_{model_prefix}.avi"),
            cv.VideoWriter_fourcc("M", "J", "P", "G"),
            15,
            (1280, 960),
        )
        for img in GetImagesIterator(sequence_path):
            start_time = time()
            _, bboxes = roi_alg.compute(img[:,:,0])
            if len(bboxes) > 0:
                inference_batch = extract_rois(img, bboxes, ROI_INFERENCE_RESOLUTION, ROI_INFERENCE_DEPTH, MARGIN_SCALING_FACTOR)
                preprocessed_batch = tf.keras.applications.xception.preprocess_input(inference_batch)
                inference_results = np.squeeze(roi_inference_alg.predict(preprocessed_batch))
                car_indices = inference_results > 0.5
                car_indices = [car_indices] if type(car_indices) is not np.ndarray else car_indices
                car_boxes = np.array(bboxes)[car_indices]

                car_boxes_joined = []
                for car_box in car_boxes:
                    car_boxes_joined.append((car_box[0], 0, car_box[2], 960))
                    
                merged_bboxes = distance_based_merging(car_boxes_joined, BBOXES_MERGING_DISTANCE)
            else:
                merged_bboxes = []
            end_time = time()
            rendered_img = visualize_inference_res(merged_bboxes, img)
            video_writer.write(rendered_img)
            cum_inference_time = end_time - start_time
            img_count += 1
        video_writer.release()
    print(f"Latency: {cum_inference_time/img_count * 1000:.2f} ms")
    print(f"FPS: {img_count / cum_inference_time:.2f} fps")
