"""
This module provides functions for calculating speckle contrast in images.

Functions:
- calculate_speckle_contrast(local_std, local_mean): Calculate the speckle contrast.
- apply_speckle_contrast(image, kernel_size, pixels_to_process, processing_origin):
    Apply speckle contrast to an image.
- process_speckle(image, kernel_size, pixels_to_process, start_pixel): Process speckle contrast for an image.
"""

# Imports
import streamlit as st
import numpy as np
from multiprocessing import Pool, cpu_count
from functools import partial
from itertools import islice
from dataclasses import dataclass
from typing import List, Dict, Tuple
from src.utils import BaseResult

# Helper functions
def calculate_speckle_contrast(local_std, local_mean):
    """
    Speckle Contrast (SC): ratio of standard deviation to mean intensity within
    the kernel centered at (x, y). Formula: SC_{x,y} = σ_{x,y} / μ_{x,y}
    """
    return local_std / local_mean if local_mean != 0 else 0

def process_pixel(args, image, kernel_size):
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

# Main processing functions
def apply_speckle_contrast(image: np.ndarray, kernel_size: int, pixels_to_process: int, processing_origin: Tuple[int, int]):
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
    args_list = ((pixel, processing_origin, height, width, valid_width) 
                 for pixel in range(pixels_to_process))

    # Use partial to fix image and kernel_size arguments
    process_pixel_partial = partial(process_pixel, image=image, kernel_size=kernel_size)

    # Determine optimal chunk size and number of processes
    chunk_size = max(1, pixels_to_process // (cpu_count() * 4))  # Adjust based on your specific use case
    num_processes = min(cpu_count(), pixels_to_process // chunk_size)

    # Process in chunks
    with Pool(processes=num_processes) as pool:
        try:
            for i, results in enumerate(pool.imap(process_pixel_partial, args_list, chunksize=chunk_size)):
                if results:
                    row, col, local_mean, local_std, sc = results
                    mean_filter[row, col] = local_mean
                    std_dev_filter[row, col] = local_std
                    sc_filter[row, col] = sc
                
                # You can implement a callback here for progress reporting
                if (i + 1) % chunk_size == 0:
                    progress = (i + 1) / pixels_to_process
                    # Report progress (e.g., through a callback function)
        except Exception as e:
            # Log the error and re-raise
            print(f"Error in apply_speckle_contrast: {str(e)}")
            raise

    return mean_filter, std_dev_filter, sc_filter

def process_speckle(image, kernel_size, pixels_to_process, start_pixel=0):

    if not isinstance(image, np.ndarray):
        image = np.array(image)

    height, width = image.shape
    half_kernel = kernel_size // 2
    valid_height, valid_width = height - kernel_size + 1, width - kernel_size + 1
    pixels_to_process = min(pixels_to_process, valid_height * valid_width)

    # Calculate starting coordinates
    start_y, start_x = divmod(start_pixel, valid_width)
    start_y += half_kernel
    start_x += half_kernel

    mean_filter, std_dev_filter, sc_filter = apply_speckle_contrast(
        image, 
        kernel_size, 
        pixels_to_process, 
        (start_x, start_y)
    )

    # Calculate processing end coordinates
    end_pixel = start_pixel + pixels_to_process
    end_y, end_x = divmod(end_pixel - 1, valid_width)
    end_y, end_x = end_y + half_kernel, end_x + half_kernel
    processing_end = (min(end_x, width - 1), min(end_y, height - 1))

    return SpeckleResult(
        mean_filter=mean_filter,
        std_dev_filter=std_dev_filter,
        speckle_contrast_filter=sc_filter,
        processing_end_coord=processing_end,
        kernel_size=kernel_size,
        pixels_processed=pixels_to_process,
        image_dimensions=(height, width),
    )


# Result class
@dataclass
class SpeckleResult(BaseResult):
    mean_filter: np.ndarray
    std_dev_filter: np.ndarray
    speckle_contrast_filter: np.ndarray

    @staticmethod
    def get_filter_options() -> List[str]:
        return ["Mean Filter", "Std Dev Filter", "Speckle Contrast"]

    def get_filter_data(self) -> Dict[str, np.ndarray]:
        return {
            "Mean Filter": self.mean_filter,
            "Std Dev Filter": self.std_dev_filter,
            "Speckle Contrast": self.speckle_contrast_filter,
        }

    @classmethod
    def combine(cls, results: List["SpeckleResult"]) -> "SpeckleResult":
        if not results:
            raise ValueError("No results to combine")

        combined_arrays = {
            attr: np.maximum.reduce([getattr(r, attr) for r in results])
            for attr in ['mean_filter', 'std_dev_filter', 'speckle_contrast_filter']
        }

        return cls(
            **combined_arrays,
            **BaseResult.combine(results).__dict__,
        )

