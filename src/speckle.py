"""
This module provides functions for calculating speckle contrast in images.

Functions:
- calculate_speckle_contrast(local_std, local_mean): Calculate the speckle contrast.
- apply_speckle_contrast(image, kernel_size, pixels_to_process, processing_origin):
    Apply speckle contrast to an image.
"""

from dataclasses import dataclass
from typing import List, Tuple, Dict
import streamlit as st
import numpy as np
from multiprocessing import Pool
from functools import partial
import logging
from itertools import islice

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

@st.cache_resource()
def apply_speckle_contrast(image, kernel_size, pixels_to_process, processing_origin):
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

    # Process in chunks
    chunk_size = 10000  # Adjust this value based on your memory constraints
    with Pool() as pool:
        for i in range(0, pixels_to_process, chunk_size):
            chunk = list(islice(args_list, chunk_size))
            try:
                results = pool.map(process_pixel_partial, chunk)
                for result in results:
                    if result:
                        row, col, local_mean, local_std, sc = result
                        mean_filter[row, col] = local_mean
                        std_dev_filter[row, col] = local_std
                        sc_filter[row, col] = sc
                logger.info(f"Processed {i + len(chunk)} pixels out of {pixels_to_process}")
            except Exception as e:
                logger.error(f"Error processing chunk starting at pixel {i}: {str(e)}")

    return mean_filter, std_dev_filter, sc_filter

def process_speckle(image, kernel_size, pixels_to_process):
    try:
        if not isinstance(image, np.ndarray):
            image = np.array(image)

        height, width = image.shape
        half_kernel = kernel_size // 2
        valid_height, valid_width = height - kernel_size + 1, width - kernel_size + 1
        pixels_to_process = min(pixels_to_process, valid_height * valid_width)

        mean_filter, std_dev_filter, sc_filter = apply_speckle_contrast(
            image, 
            kernel_size, 
            pixels_to_process, 
            (half_kernel, half_kernel)
        )

        # Calculate processing end coordinates
        end_y, end_x = divmod(pixels_to_process - 1, valid_width)
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
    except Exception as e:
        st.error(f"Error in process_speckle: {type(e).__name__}: {str(e)}")
        raise

# --- Data Class for Results ---
@dataclass
class SpeckleResult:
    """Represents the result of a speckle filter, containing mean and standard
    deviation filters."""

    mean_filter: np.ndarray
    std_dev_filter: np.ndarray
    speckle_contrast_filter: np.ndarray
    processing_end_coord: Tuple[int, int]
    kernel_size: int
    pixels_processed: int
    image_dimensions: Tuple[int, int]

    @staticmethod
    def get_filter_options() -> List[str]:
        return ["Mean Filter", "Std Dev Filter", "Speckle Contrast"]

    def get_filter_data(self) -> Dict[str, np.ndarray]:
        return {
            "Mean Filter": self.mean_filter,
            "Std Dev Filter": self.std_dev_filter,
            "Speckle Contrast": self.speckle_contrast_filter,
        }

    def get_last_processed_coordinates(self) -> Tuple[int, int]:
        """Get the last processed pixel coordinates."""
        return self.processing_end_coord

    @classmethod
    def combine(cls, results):
        if not results:
            raise ValueError("No results to combine")

        combined_mean = np.maximum.reduce([r.mean_filter for r in results])
        combined_std = np.maximum.reduce([r.std_dev_filter for r in results])
        combined_sc = np.maximum.reduce([r.speckle_contrast_filter for r in results])

        total_pixels = sum(r.pixels_processed for r in results)
        max_coord = max(r.processing_end_coord for r in results)

        return cls(
            mean_filter=combined_mean,
            std_dev_filter=combined_std,
            speckle_contrast_filter=combined_sc,
            processing_end_coord=max_coord,
            kernel_size=results[0].kernel_size,
            pixels_processed=total_pixels,
            image_dimensions=results[0].image_dimensions
        )
