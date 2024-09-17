import numpy as np
from numba import njit
from dataclasses import dataclass
from typing import List
from shared_types import FilterResult, calculate_processing_details

SPECKLE_FORMULA_CONFIG = {
    "title": "Speckle Contrast Calculation",
    "main_formula": r"I_{{{x},{y}}} = {original_value:.3f} \quad \rightarrow \quad SC_{{{x},{y}}} = \frac{{\sigma_{{{x},{y}}}}}{{\mu_{{{x},{y}}}}} = \frac{{{std:.3f}}}{{{mean:.3f}}} = {sc:.3f}",
    "explanation": r"This formula shows the transition from the original pixel intensity $I_{{{x},{y}}}$ to the Speckle Contrast (SC) for the same pixel position.",
    "additional_formulas": [
        {
            "title": "Neighborhood Analysis",
            "formula": r"\text{{Kernel Size: }} {kernel_size} \times {kernel_size}"
                       r"\quad\quad\text{{Centered at pixel: }}({x}, {y})"
                       r"\\\\"
                       "{kernel_matrix}",
            "explanation": r"Analysis of a ${kernel_size}\times{kernel_size}$ neighborhood centered at pixel $({x},{y})$. The matrix shows pixel values, with the central value (in bold) being the processed pixel."
        },
        {
            "title": "Mean Calculation", 
            "formula": r"\mu_{{{x},{y}}} = \frac{{1}}{{N}} \sum_{{i,j \in K_{{{x},{y}}}}} I_{{i,j}} = \frac{{1}}{{{kernel_size}^2}} \sum_{{i,j \in K_{{{x},{y}}}}} I_{{i,j}} = {mean:.3f}",
            "explanation": r"Mean ($\mu$) calculation: average intensity of all pixels in the kernel $K$ centered at $({x},{y})$. $N = {kernel_size}^2 = {total_pixels}$."
        },
        {
            "title": "Standard Deviation Calculation",
            "formula": r"\sigma_{{{x},{y}}} = \sqrt{{\frac{{1}}{{N}} \sum_{{i,j \in K_{{{x},{y}}}}} (I_{{i,j}} - \mu_{{{x},{y}}})^2}} = \sqrt{{\frac{{1}}{{{kernel_size}^2}} \sum_{{i,j \in K_{{{x},{y}}}}} (I_{{i,j}} - {mean:.3f})^2}} = {std:.3f}",
            "explanation": r"Standard deviation ($\sigma$) calculation: measure of intensity spread around the mean for all pixels in the kernel $K$ centered at $({x},{y})$."
        },
        {
            "title": "Speckle Contrast Calculation",
            "formula": r"SC_{{{x},{y}}} = \frac{{\sigma_{{{x},{y}}}}}{{\mu_{{{x},{y}}}}} = \frac{{{std:.3f}}}{{{mean:.3f}}} = {sc:.3f}",
            "explanation": r"Speckle Contrast (SC): ratio of standard deviation to mean intensity within the kernel centered at $({x},{y})$."
        }
    ]
}

@njit
def calculate_speckle_stats(local_window):
    local_mean = np.mean(local_window)
    local_std = np.std(local_window)
    speckle_contrast = local_std / local_mean if local_mean != 0 else 0
    return local_mean, local_std, speckle_contrast

@njit
def apply_speckle_contrast(image, kernel_size, pixels_to_process, height, width, start_x, start_y):
    mean_filter = np.zeros((height, width), dtype=np.float32)
    std_dev_filter = np.zeros((height, width), dtype=np.float32)
    sc_filter = np.zeros((height, width), dtype=np.float32)
    half_kernel = kernel_size // 2
    valid_width = width - kernel_size + 1

    last_processed_x, last_processed_y = start_x, start_y

    for pixel in range(pixels_to_process):
        row = start_y + pixel // valid_width
        col = start_x + pixel % valid_width
        if row < height and col < width:
            local_window = image[max(0, row-half_kernel):min(height, row+half_kernel+1),
                                 max(0, col-half_kernel):min(width, col+half_kernel+1)]
            mean_filter[row, col], std_dev_filter[row, col], sc_filter[row, col] = calculate_speckle_stats(local_window)
            last_processed_x, last_processed_y = col, row

    return mean_filter, std_dev_filter, sc_filter, last_processed_x, last_processed_y

def process_speckle(image, kernel_size, pixels_to_process):
    try:
        processing_info = calculate_processing_details(image, kernel_size, pixels_to_process)
        
        mean_filter, std_dev_filter, sc_filter, last_processed_x, last_processed_y = apply_speckle_contrast(
            image, kernel_size, processing_info.pixels_to_process, 
            processing_info.image_height, processing_info.image_width, 
            processing_info.start_x, processing_info.start_y
        )

        start_x, start_y = processing_info.start_x, processing_info.start_y

        return SpeckleResult(
            mean_filter=mean_filter,
            std_dev_filter=std_dev_filter,
            speckle_contrast_filter=sc_filter,
            start_pixel_mean=mean_filter[start_y, start_x],
            start_pixel_std_dev=std_dev_filter[start_y, start_x],
            start_pixel_speckle_contrast=sc_filter[start_y, start_x],
            processing_coord=(start_x, start_y),
            processing_end_coord=(last_processed_x, last_processed_y),
            kernel_size=kernel_size,
            pixels_processed=processing_info.pixels_to_process,
            image_dimensions=(processing_info.image_height, processing_info.image_width)
        )
    except Exception as e:
        print(f"Error in process_speckle: {str(e)}")
        return None

@dataclass
class SpeckleResult(FilterResult):
    mean_filter: np.ndarray
    std_dev_filter: np.ndarray
    speckle_contrast_filter: np.ndarray
    start_pixel_mean: float
    start_pixel_std_dev: float
    start_pixel_speckle_contrast: float

    @classmethod
    def get_filter_options(cls) -> List[str]:
        return ["Mean Filter", "Std Dev Filter", "Speckle Contrast"]

    def get_filter_data(self) -> dict:
        return {
            "Mean Filter": self.mean_filter,
            "Std Dev Filter": self.std_dev_filter,
            "Speckle Contrast": self.speckle_contrast_filter
        }