import numpy as np
from typing import Tuple, Dict, Any
from numba import njit

import streamlit as st
from image_processing import calculate_processing_details

SPECKLE_FORMULA_CONFIG = {
    "main_formula": r"I_{{{x},{y}}} = {original_value:.3f} \quad \rightarrow \quad SC_{{{x},{y}}} = \frac{{\sigma_{{{x},{y}}}}}{{\mu_{{{x},{y}}}}} = \frac{{{std:.3f}}}{{{mean:.3f}}} = {sc:.3f}",
    "explanation": "This formula shows the transition from the original pixel intensity I({x},{y}) to the Speckle Contrast (SC) for the same pixel position.",
    "variables": {},  # To be filled dynamically
    "additional_formulas": [
        {
            "title": "Neighborhood Analysis",
            "formula": r"\text{{Kernel Size: }} {kernel_size} \times {kernel_size}"
                       r"\quad\quad\text{{Centered at pixel: }}({x}, {y})"
                       r"\\\\"
                       "{kernel_matrix}",
            "explanation": "We analyze a {kernel_size}x{kernel_size} neighborhood centered around the pixel ({x},{y}). This matrix shows the pixel values in the neighborhood. The central value (in bold) corresponds to the pixel being processed."
        },
        {
            "title": "Mean Calculation", 
            "formula": r"\mu_{{{x},{y}}} = \frac{{1}}{{N}} \sum_{{(i,j) \in K_{{{x},{y}}}}} I_{{i,j}} = \frac{{1}}{{{total_pixels}}} \sum_{{(i,j) \in K_{{{x},{y}}}}} I_{{i,j}} = {mean:.3f}",
            "explanation": "The mean (μ) is calculated as the average intensity of all pixels in the kernel K centered at ({x},{y}), where N is the total number of pixels in the kernel (N = {kernel_size}^2 = {total_pixels})."
        },
        {
            "title": "Standard Deviation Calculation",
            "formula": r"\sigma_{{{x},{y}}} = \sqrt{{\frac{{1}}{{N}} \sum_{{(i,j) \in K_{{{x},{y}}}}} (I_{{i,j}} - \mu_{{{x},{y}}})^2}} = \sqrt{{\frac{{1}}{{{total_pixels}}} \sum_{{(i,j) \in K_{{{x},{y}}}}} (I_{{i,j}} - {mean:.3f})^2}} = {std:.3f}",
            "explanation": "The standard deviation (σ) is calculated using all pixels in the kernel K centered at ({x},{y}), measuring the spread of intensities around the mean."
        },
        {
            "title": "Speckle Contrast Calculation",
            "formula": r"SC_{{{x},{y}}} = \frac{{\sigma_{{{x},{y}}}}}{{\mu_{{{x},{y}}}}} = \frac{{{std:.3f}}}{{{mean:.3f}}} = {sc:.3f}",
            "explanation": "The Speckle Contrast (SC) is the ratio of the standard deviation to the mean intensity within the kernel centered at ({x},{y})."
        }
    ]
}

@njit
def calculate_speckle_stats(local_window: np.ndarray) -> Tuple[float, float, float]:
    """Calculate mean, standard deviation, and speckle contrast for a local window."""
    local_mean = np.mean(local_window)
    local_std = np.std(local_window)
    speckle_contrast = local_std / local_mean if local_mean != 0 else 0
    return local_mean, local_std, speckle_contrast

@njit
def process_pixel(row: int, col: int, image: np.ndarray, mean_filter: np.ndarray, 
                  std_dev_filter: np.ndarray, sc_filter: np.ndarray, kernel_size: int) -> None:
    """Process a single pixel for speckle contrast calculation."""
    half_kernel = kernel_size // 2
    local_window = image[row-half_kernel:row+half_kernel+1, col-half_kernel:col+half_kernel+1]
    local_mean, local_std, speckle_contrast = calculate_speckle_stats(local_window)

    mean_filter[row, col] = local_mean
    std_dev_filter[row, col] = local_std
    sc_filter[row, col] = speckle_contrast

@njit
def apply_speckle_contrast(image: np.ndarray, kernel_size: int, pixels_to_process: int, 
                           height: int, width: int, first_x: int, first_y: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply speckle contrast calculation to the image."""
    mean_filter = np.zeros((height, width), dtype=np.float32)
    std_dev_filter = np.zeros((height, width), dtype=np.float32)
    sc_filter = np.zeros((height, width), dtype=np.float32)

    for pixel in range(pixels_to_process):
        row = first_y + pixel // (width - kernel_size + 1)
        col = first_x + pixel % (width - kernel_size + 1)
        process_pixel(row, col, image, mean_filter, std_dev_filter, sc_filter, kernel_size)

    return mean_filter, std_dev_filter, sc_filter

@st.cache_data(persist=True)
def process_speckle(image: np.ndarray, kernel_size: int, max_pixels: int) -> Dict[str, Any]:
    """Process the image using Speckle Contrast calculation."""
    details = calculate_processing_details(image, kernel_size, max_pixels)
    
    mean_filter, std_dev_filter, sc_filter = apply_speckle_contrast(
        image, kernel_size, details['pixels_to_process'], details['height'], details['width'], 
        details['first_x'], details['first_y']
    )

    return {
        'mean_filter': mean_filter,
        'std_dev_filter': std_dev_filter,
        'speckle_contrast_filter': sc_filter,
        'first_pixel': (details['first_x'], details['first_y']),
        'first_pixel_stats': {
            'mean': mean_filter[details['first_y'], details['first_x']],
            'std_dev': std_dev_filter[details['first_y'], details['first_x']],
            'speckle_contrast': sc_filter[details['first_y'], details['first_x']]
        },
        'additional_info': {
            'kernel_size': kernel_size,
            'pixels_processed': details['pixels_to_process'],
            'image_dimensions': (details['height'], details['width'])
        }
    }