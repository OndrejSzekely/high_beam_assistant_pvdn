import argparse
from shared.enums import RoiDetMethods, PVDNSets
from shared.dataclasses import ParamsContainer, MetricsContainer
from bbox_annotations_gen.pvdn_data_parser import PVDNBboxAnnotationsGenerator, PVDNBboxFractionSamplesAnnotationsGenerator, PVDNBboxMetrics
from hyperopt import fmin, atpe, hp, STATUS_OK, Trials
from roi_det.dynamic_thresholding import DynamicThresholding

def dynamic_threhold_f(metrics: MetricsContainer) -> float:
    return metrics.f1

def get_optimization_logging(computation_func):
    def optimization_logging(params):
        global best_metrics
        global best_params

        metrics = computation_func(**params)
        if dynamic_threhold_f(metrics) > dynamic_threhold_f(best_metrics):
            best_metrics = metrics
            best_params = ParamsContainer(**params)
            print(f"New best! f: {dynamic_threhold_f(metrics)}")

        return {'loss': -dynamic_threhold_f(metrics), 'status': STATUS_OK}
    return optimization_logging

def get_optimizaiton_func_dynamic_threshold(args):
    def optimizaiton_func_dynamic_threshold(**params):
        bbox_generator = DynamicThresholding(**params)
        pvdn_bbox_metrics_op = PVDNBboxMetrics()
        training_bboxes = PVDNBboxFractionSamplesAnnotationsGenerator(
            PVDNSets.TRAIN, args.dataset_folder, bbox_generator, args.samples_rate, args.random_seed
        )
        for (
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
            pvdn_bbox_metrics_op.update(
                tps, fps, fns, described_keypoints_num, q_k_partial, q_b_partial
            )
        metrics = MetricsContainer(*pvdn_bbox_metrics_op.get_metrics())
        return metrics
    return optimizaiton_func_dynamic_threshold


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Hyper-parameters search of bbox generation of PVDN dataset using Dynamic Thresholding."
    )
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
        "-r",
        "--samples-rate",
        type=float,
        help="Percentage of image samples used for HPO.",
    )
    parser.add_argument(
        "-e",
        "--evals-num",
        type=int,
        help="Number of HPO evaluations.",
    )
    parser.add_argument(
        "-s",
        "--random-seed",
        type=int,
        default=456,
        help="Random generator seed.",
    )


    args = parser.parse_args()

    best_metrics = MetricsContainer(0, 0, 0, 0, 0, 0)
    best_params = None

    parameters_space = {
        "blur_kernel_size": hp.choice('blur_kernel_size', [1, 3, 5, 9, 13, 17, 21, 25, 29]),
        "local_intensity_win": hp.choice('local_intensity_win', [1, 3, 5, 9, 13, 17, 21, 25, 29]),
        "dynamic_threshold_factor": hp.uniform('dynamic_threshold_factor', 0.4, 0.401),
        "opening_kernel": hp.choice('opening_kernel', [1, 3, 5, 9, 13, 17, 21]),
        "margin_distance": hp.uniform('margin_distance', 0, 30),
        "minimal_roi_mean_abs_dev": hp.uniform('minimal_roi_mean_abs_dev', 0.01, 0.0101),
        "processing_scaling_factor": hp.uniform('processing_scaling_factor', 0.3, 0.6),
    }
    computation_func = get_optimizaiton_func_dynamic_threshold(args)
    logging_fun = get_optimization_logging(computation_func)

    trials = Trials()
    best = fmin(logging_fun, parameters_space, algo=atpe.suggest, max_evals=args.evals_num, trials=trials)


    print("\n\n\n")
    print("COMPLETED")
    print("BEST METRICS")
    best_metrics.print_metrics()
    print("BEST PARAMS")
    best_params.print_params()


