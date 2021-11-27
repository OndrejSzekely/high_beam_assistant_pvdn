from dataclasses import dataclass

@dataclass
class ParamsContainer:
    blur_kernel_size: int
    local_intensity_win: int
    dynamic_threshold_factor: float
    opening_kernel: int
    margin_distance: float
    minimal_roi_mean_abs_dev: float
    processing_scaling_factor: float

    def print_params(self):
        print(f"\t{'blur_kernel_size:':>30} {self.blur_kernel_size}")
        print(f"\t{'local_intensity_win:':>30} {self.local_intensity_win}")
        print(f"\t{'dynamic_threshold_factor:':>30} {self.dynamic_threshold_factor}")
        print(f"\t{'opening_kernel:':>30} {self.opening_kernel}")
        print(f"\t{'margin_distance:':>30} {self.margin_distance}")
        print(f"\t{'minimal_roi_mean_abs_dev:':>30} {self.minimal_roi_mean_abs_dev}")
        print(f"\t{'processing_scaling_factor:':>30} {self.processing_scaling_factor}")

@dataclass
class MetricsContainer:
    precision: float
    recall: float
    f1: float
    q_k: float
    q_b: float
    q: float

    def print_metrics(self):
        print(f"\t{'Precision:':>20} {self.precision:>12.2f}")
        print(f"\t{'Recall:':>20} {self.recall:>12.2f}")
        print(f"\t{'F1 Score:':>20} {self.f1:>12.2f}")
        print(f"\t{'KP/Bbox Quality:':>20} {self.q_k:>12.2f}")
        print(f"\t{'Bbox/KP Quality:':>20} {self.q_b:>12.2f}")
        print(f"\t{'Bbox Quality:':>20} {self.q:>12.2f}")
