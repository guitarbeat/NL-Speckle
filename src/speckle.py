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
from multiprocessing import Pool, cpu_count
from functools import partial
from typing import  Tuple


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


def apply_speckle_contrast(
    image: np.ndarray,
    kernel_size: int,
    pixels_to_process: int,
    processing_origin: Tuple[int, int],
):
    """Applies speckle contrast to the given image using parallel processing."""
    if not isinstance(image, np.ndarray):
        raise TypeError(f"Expected numpy array, got {type(image)}")
    if image.ndim != 2:
        raise ValueError(f"Expected 2D array, got {image.ndim}D array")

    height, width = image.shape
    mean_filter = np.zeros((height, width), dtype=np.float32)
    std_dev_filter = np.zeros((height, width), dtype=np.float32)
    sc_filter = np.zeros((height, width), dtype=np.float32)
    valid_width = width - kernel_size + 1

    # Prepare arguments for parallel processing
    args_list = [
        (pixel, processing_origin, height, width, valid_width)
        for pixel in range(pixels_to_process)
    ]

    # Use partial to fix image and kernel_size arguments
    process_pixel_partial = partial(
        process_speckle_pixel, image=image, kernel_size=kernel_size
    )

    # Determine optimal chunk size and number of processes
    chunk_size = max(1, pixels_to_process // (cpu_count() * 4))
    num_processes = min(cpu_count(), pixels_to_process // chunk_size)

    # Process using Pool.map with dill serialization
    with Pool(processes=num_processes) as pool:
        try:
            results = pool.map(process_pixel_partial, args_list, chunksize=chunk_size)

            for result in results:
                if result:
                    row, col, local_mean, local_std, sc = result
                    mean_filter[row, col] = local_mean
                    std_dev_filter[row, col] = local_std
                    sc_filter[row, col] = sc

        except Exception as e:
            # Log the error and re-raise
            print(f"Error in apply_speckle_contrast: {str(e)}")
            raise

    return mean_filter, std_dev_filter, sc_filter

