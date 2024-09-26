"""
This module provides functions for calculating speckle contrast in images.

Functions: - calculate_mean(local_window): Calculate the mean intensity within a
local window. - calculate_speckle_contrast(local_std, local_mean): Calculate the
speckle contrast. - apply_speckle_contrast(image, kernel_size,
pixels_to_process, processing_origin):
    Apply speckle contrast to an image.
"""

from dataclasses import dataclass
from typing import List
import streamlit as st
import numpy as np


from src.processing import FilterResult, ProcessingDetails, calculate_processing_details
from src.decor import timeit
# from src.sat import calculate_summed_area_table, get_area_sum

def calculate_speckle_contrast(local_std, local_mean):
    """
    Speckle Contrast (SC): ratio of standard deviation to mean intensity within
    the kernel centered at (x, y). Formula: SC_{x,y} = σ_{x,y} / μ_{x,y}
    """
    return local_std / local_mean if local_mean != 0 else 0

@timeit
@st.cache_resource()
def apply_speckle_contrast(image, kernel_size, pixels_to_process, processing_origin):
    """Applies speckle contrast to the given image."""
    if not isinstance(image, np.ndarray):
        raise TypeError(f"Expected numpy array, got {type(image)}")
    if image.ndim != 2:
        raise ValueError(f"Expected 2D array, got {image.ndim}D array")
    
    height, width = image.shape
    mean_filter = np.zeros((height, width), dtype=np.float32)
    std_dev_filter = np.zeros((height, width), dtype=np.float32)
    sc_filter = np.zeros((height, width), dtype=np.float32)
    half_kernel = kernel_size // 2
    valid_width = width - kernel_size + 1

    # with st.sidebar.status(f"Applying speckle contrast on {st.session_state.image_file}", expanded=True) as status:

              
    for pixel in range(pixels_to_process):
        row = processing_origin[1] + pixel // valid_width
        col = processing_origin[0] + pixel % valid_width
        if row < height and col < width:
            row_start = max(0, row - half_kernel)
            row_end = min(height, row + half_kernel + 1)
            col_start = max(0, col - half_kernel)
            col_end = min(width, col + half_kernel + 1)
            

       
            local_window = image[row_start:row_end, col_start:col_end]
            local_mean = np.mean(local_window)
            local_std = np.std(local_window)

            mean_filter[row, col] = local_mean
            std_dev_filter[row, col] = local_std
            sc_filter[row, col] = calculate_speckle_contrast(
                local_std, local_mean)
                
        #     status.update(label=f"We've processed {pixel} pixels so far. {pixels_to_process - pixel} pixels left.")     
        # status.update(label=f"Speckle contrast calculation on {st.session_state.image_file} complete!", state="complete")

    return mean_filter, std_dev_filter, sc_filter

def process_speckle(image, kernel_size, pixels_to_process):
    """
    Processes a speckle image to calculate speckle contrast and related metrics.
    """
    # use_sat = st.session_state.get("use_sat")
    # st.write(f"Use Summed Area Tables Value is {use_sat}")
    try:
        if not isinstance(image, np.ndarray):
            image = np.array(image)

        processing_info: ProcessingDetails = calculate_processing_details(
            image, kernel_size, pixels_to_process
        )

        start_y, start_x = processing_info.processing_origin

        mean_filter, std_dev_filter, sc_filter = apply_speckle_contrast(
            image, kernel_size, pixels_to_process, (start_x, start_y), 
        )
        
        return SpeckleResult(
            mean_filter=mean_filter,
            std_dev_filter=std_dev_filter,
            speckle_contrast_filter=sc_filter,
            start_pixel_mean=mean_filter[start_y, start_x],
            start_pixel_std_dev=std_dev_filter[start_y, start_x],
            start_pixel_speckle_contrast=sc_filter[start_y, start_x],
            processing_end_coord=processing_info.processing_end,
            kernel_size=kernel_size,
            pixels_processed=processing_info.pixels_to_process,
            image_dimensions=processing_info.image_dimensions,
        )
    except Exception as e:
        print(f"Error in process_speckle: {type(e).__name__}: {str(e)}")
        import traceback

        traceback.print_exc()
        return None


# --- Data Class for Results ---
@dataclass
class SpeckleResult(FilterResult):
    """Represents the result of a speckle filter, containing mean and standard
    deviation filters."""

    mean_filter: np.ndarray
    std_dev_filter: np.ndarray
    speckle_contrast_filter: np.ndarray
    start_pixel_mean: float
    start_pixel_std_dev: float
    start_pixel_speckle_contrast: float

    @staticmethod
    def get_filter_options() -> List[str]:
        return ["Mean Filter", "Std Dev Filter", "Speckle Contrast"]

    def get_filter_data(self) -> dict:
        return {
            "Mean Filter": self.mean_filter,
            "Std Dev Filter": self.std_dev_filter,
            "Speckle Contrast": self.speckle_contrast_filter,
        }
