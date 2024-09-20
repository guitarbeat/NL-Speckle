"""
This module provides functions for calculating speckle contrast in images.

Functions:
- calculate_mean(local_window): Calculate the mean intensity within a local window.
- calculate_speckle_contrast(local_std, local_mean): Calculate the speckle contrast.
- apply_speckle_contrast(image, kernel_size, pixels_to_process, start_point): Apply speckle contrast to an image.
"""

from dataclasses import dataclass
from typing import List

import numpy as np
from numba import njit

from src.utils import (
    FilterResult,
    Point,
    ProcessingDetails,
    calculate_processing_details,
)


# --- Mean Calculation ---
@njit
def calculate_mean(local_window):
    """
    Mean (μ) calculation: average intensity of all pixels in the kernel K centered at (x, y).
    Formula: μ_{x,y} = (1 / N) * Σ_{i,j ∈ K_{x,y}} I_{i,j} = (1 / kernel_size^2) * Σ_{i,j ∈ K_{x,y}} I_{i,j}
    """
    return np.mean(local_window)


# --- Standard Deviation Calculation ---


# --- Speckle Contrast Calculation ---
@njit
def calculate_speckle_contrast(local_std, local_mean):
    """
    Speckle Contrast (SC): ratio of standard deviation to mean intensity within the kernel centered at (x, y).
    Formula: SC_{x,y} = σ_{x,y} / μ_{x,y}
    """
    return local_std / local_mean if local_mean != 0 else 0


# --- Apply Speckle Contrast ---
@njit
def apply_speckle_contrast(image, kernel_size, pixels_to_process, start_point: Point):
    height, width = image.shape
    mean_filter = np.zeros((height, width), dtype=np.float32)
    std_dev_filter = np.zeros((height, width), dtype=np.float32)
    sc_filter = np.zeros((height, width), dtype=np.float32)
    half_kernel = kernel_size // 2
    valid_width = width - kernel_size + 1

    # Process each pixel in parallel
    for pixel in range(pixels_to_process):
        row = start_point.y + pixel // valid_width
        col = start_point.x + pixel % valid_width
        if row < height and col < width:
            local_window = image[
                max(0, row - half_kernel) : min(height, row + half_kernel + 1),
                max(0, col - half_kernel) : min(width, col + half_kernel + 1),
            ]

            # Calculate mean, std deviation, and speckle contrast
            mean_filter[row, col] = calculate_mean(local_window)
            std_dev_filter[row, col] = np.std(local_window)
            sc_filter[row, col] = calculate_speckle_contrast(
                std_dev_filter[row, col], mean_filter[row, col]
            )

    return mean_filter, std_dev_filter, sc_filter


# --- Main Processing Function ---
def process_speckle(image, kernel_size, pixels_to_process):
    """
    Processes a speckle image to calculate speckle contrast and related metrics.
    """
    try:
        # Get processing details
        processing_info: ProcessingDetails = calculate_processing_details(
            image, kernel_size, pixels_to_process
        )

        # Apply speckle contrast calculations
        mean_filter, std_dev_filter, sc_filter = apply_speckle_contrast(
            image,
            kernel_size,
            processing_info.pixels_to_process,
            processing_info.start_point,
        )

        # Return results encapsulated in SpeckleResult
        return SpeckleResult(
            mean_filter=mean_filter,
            std_dev_filter=std_dev_filter,
            speckle_contrast_filter=sc_filter,
            start_pixel_mean=mean_filter[
                processing_info.start_point.y, processing_info.start_point.x
            ],
            start_pixel_std_dev=std_dev_filter[
                processing_info.start_point.y, processing_info.start_point.x
            ],
            start_pixel_speckle_contrast=sc_filter[
                processing_info.start_point.y, processing_info.start_point.x
            ],
            processing_coord=processing_info.start_point,
            processing_end_coord=processing_info.end_point,
            kernel_size=kernel_size,
            pixels_processed=processing_info.pixels_to_process,
            image_dimensions=processing_info.image_dimensions,
        )
    except ValueError as ve:
        print(f"ValueError in process_speckle: {str(ve)}")
        return None
    except TypeError as te:
        print(f"TypeError in process_speckle: {str(te)}")
        return None


# --- Data Class for Results ---
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
            "Speckle Contrast": self.speckle_contrast_filter,
        }
