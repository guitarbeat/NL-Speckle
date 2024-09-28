"""
This module provides functions for calculating speckle contrast in images.

Functions:
- calculate_speckle_contrast(local_std, local_mean): Calculate the speckle contrast.
- apply_speckle_contrast(image, kernel_size, pixels_to_process, processing_origin):
    Apply speckle contrast to an image.
- process_speckle(image, kernel_size, pixels_to_process, start_pixel): Process speckle contrast for an image.
"""

# Imports
import numpy as np
from typing import Tuple, Optional, Callable, Type, List, Dict
from dataclasses import dataclass
from src.classes import BaseResult, ResultCombinationError
from src.utils import validate_input, calculate_processing_end


@dataclass
class SpeckleResult(BaseResult):
    mean_filter: np.ndarray
    std_dev_filter: np.ndarray
    speckle_contrast_filter: np.ndarray

    @staticmethod
    def get_filter_options() -> List[str]:
        return ["Mean Filter", "Std Dev Filter", "Speckle Contrast"]

    @property
    def filter_data(self) -> Dict[str, np.ndarray]:
        return {
            "Mean Filter": self.mean_filter,
            "Std Dev Filter": self.std_dev_filter,
            "Speckle Contrast": self.speckle_contrast_filter,
        }

    @classmethod
    def combine(
        class_: Type["SpeckleResult"], results: List["SpeckleResult"]
    ) -> "SpeckleResult":
        if not results:
            raise ResultCombinationError("No Speckle results provided for combination")

        try:
            combined_arrays: Dict[str, np.ndarray] = {
                attr: np.maximum.reduce([getattr(r, attr) for r in results])
                for attr in ["mean_filter", "std_dev_filter", "speckle_contrast_filter"]
            }
        except ValueError as e:
            raise ResultCombinationError(f"Error combining Speckle results: {str(e)}")

        return class_(
            **combined_arrays,
            **BaseResult.combine(results).__dict__,
        )

    @classmethod
    def merge(
        class_: Type["SpeckleResult"],
        new_result: "SpeckleResult",
        existing_result: "SpeckleResult",
    ) -> "SpeckleResult":
        merged_arrays: Dict[str, np.ndarray] = {
            attr: np.maximum(getattr(new_result, attr), getattr(existing_result, attr))
            for attr in ["mean_filter", "std_dev_filter", "speckle_contrast_filter"]
        }

        return class_(
            **merged_arrays,
            processing_end_coord=max(
                new_result.processing_end_coord, existing_result.processing_end_coord
            ),
            kernel_size=new_result.kernel_size,
            pixels_processed=max(
                new_result.pixels_processed, existing_result.pixels_processed
            ),
            image_dimensions=new_result.image_dimensions,
        )


def process_speckle_contrast(status, image: np.ndarray, kernel_size: int, pixel_count: int,
                             update_progress: Callable) -> SpeckleResult:
    validate_input(image, kernel_size, pixel_count)
    height, width = image.shape
    half_kernel = kernel_size // 2
    _valid_height, _valid_width = height - kernel_size + 1, width - kernel_size + 1

    status.write("📊 Calculating local statistics (mean, standard deviation) for each pixel neighborhood")
    status.write(f"🔍 Using a {kernel_size}x{kernel_size} kernel to analyze {pixel_count} pixels") 
    try:
        mean_image, std_dev_image, speckle_contrast_image = apply_speckle_contrast(
            image, kernel_size, pixel_count, (half_kernel, half_kernel), update_progress)
    except Exception as e:
        raise ResultCombinationError(f"Error in apply_speckle_contrast: {str(e)}") from e

    processing_end = calculate_processing_end(width, height, kernel_size, pixel_count)
    status.write("📈 Computing speckle contrast (σ/μ) from local statistics")

    return SpeckleResult(mean_filter=mean_image, std_dev_filter=std_dev_image, 
                         speckle_contrast_filter=speckle_contrast_image,
                         processing_end_coord=processing_end, kernel_size=kernel_size,
                         pixels_processed=pixel_count, image_dimensions=(height, width))

def apply_speckle_contrast(
    image: np.ndarray, 
    kernel_size: int,
    pixels_to_process: int,
    processing_origin: Tuple[int, int],
    progress_callback: Optional[Callable[[float], None]] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    height, width = image.shape
    mean_filter = np.zeros_like(image) 
    std_dev_filter = np.zeros_like(image)
    sc_filter = np.zeros_like(image)
    valid_width = width - kernel_size + 1

    half_kernel = kernel_size // 2
    update_interval = max(1, pixels_to_process // 1000)

    for pixel in range(pixels_to_process):
        row = processing_origin[1] + pixel // valid_width
        col = processing_origin[0] + pixel % valid_width

        if row < height and col < width:
            row_start, row_end = max(0, row - half_kernel), min(height, row + half_kernel + 1)
            col_start, col_end = max(0, col - half_kernel), min(width, col + half_kernel + 1)

            local_window = image[row_start:row_end, col_start:col_end]
            local_mean = np.nanmean(local_window) 
            local_std = np.nanstd(local_window)
            sc = calculate_speckle_contrast(local_std, local_mean)

            mean_filter[row, col] = local_mean
            std_dev_filter[row, col] = local_std
            sc_filter[row, col] = sc

            if progress_callback and pixel % update_interval == 0:
                progress_callback((pixel + 1) / pixels_to_process)

    if progress_callback:
        progress_callback(1.0)

    return mean_filter, std_dev_filter, sc_filter


# Moved process_speckle_pixel to the top level of the module
def process_speckle_pixel(args, image, kernel_size):
    pixel, processing_origin, height, width, valid_width = args
    row = processing_origin[1] + pixel // valid_width
    col = processing_origin[0] + pixel % valid_width
    if row < height and col < width:
        half_kernel = kernel_size // 2
        row_start = max(0, row - half_kernel)
        row_end = min(height, row + half_kernel + 1)
        col_start = max(0, col - half_kernel)
        col_end = min(width, col + half_kernel + 1)

        local_window = image[row_start:row_end, col_start:col_end]
        local_mean = np.nanmean(local_window)
        local_std = np.nanstd(local_window)
        sc = calculate_speckle_contrast(local_std, local_mean)

        return row, col, local_mean, local_std, sc
    return None


# Helper functions


def calculate_speckle_contrast(local_std, local_mean):
    """
    Speckle Contrast (SC): ratio of standard deviation to mean intensity within
    the kernel centered at (x, y). Formula: SC_{x,y} = σ_{x,y} / μ_{x,y}
    """
    return local_std / local_mean if local_mean != 0 else 0


# Main processing functions

