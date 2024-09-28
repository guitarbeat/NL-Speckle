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

