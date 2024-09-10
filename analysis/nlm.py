import numpy as np
from typing import Optional, Tuple, Dict, Any
from numba import njit

import streamlit as st
from image_processing import calculate_processing_details

NLM_FORMULA_CONFIG = {
    "main_formula": r"I_{{{x},{y}}} = {original_value:.3f} \quad \rightarrow \quad NLM_{{{x},{y}}} = \frac{{1}}{{W_{{{x},{y}}}}} \sum_{{(i,j) \in \Omega_{{{x},{y}}}}} I_{{i,j}} \cdot w_{{{x},{y}}}(i,j) = {nlm_value:.3f}",
    "explanation": "This formula shows the transition from the original pixel intensity I({x},{y}) to the denoised value NLM({x},{y}) using the Non-Local Means (NLM) algorithm.",
    "variables": {},  # To be filled dynamically
    "additional_formulas": [
        {
            "title": "Neighborhood Analysis",
            "formula": r"\text{{Patch Size: }} {kernel_size} \times {kernel_size}"
                       r"\quad\quad\text{{Centered at pixel: }}({x}, {y})"
                       r"\\\\"
                       "{kernel_matrix}",
            "explanation": "We analyze a {kernel_size}x{kernel_size} patch centered around the pixel ({x},{y}). This matrix shows the pixel values in the patch. The central value (in bold) corresponds to the pixel being denoised."
        },
        {
            "title": "Weight Calculation",
            "formula": r"w_{{{x},{y}}}(i,j) = \exp\left(-\frac{{\|P_{{{x},{y}}} - P_{{i,j}}\|^2}}{{h^2}}\right)",
            "explanation": r"""
            This formula determines the weight of each pixel (i,j) when denoising pixel (x,y):
            - w_{{{x},{y}}}(i,j): Weight assigned to pixel (i,j) when denoising (x,y) 
            - P_{{{x},{y}}} and P_{{i,j}} are patches centered at (x,y) and (i,j)
            - \|P_{{{x},{y}}} - P_{{i,j}}\|^2 measures the squared difference between patches
            - h = {filter_strength} controls the smoothing strength
            - More similar patches result in higher weights
            """
        },
        {
            "title": "Normalization Factor",
            "formula": r"W_{{{x},{y}}} = \sum_{{(i,j) \in \Omega_{{{x},{y}}}}} w_{{{x},{y}}}(i,j)", 
            "explanation": "We sum all weights for pixel (x,y). This ensures the final weighted average preserves the overall image brightness."
        },
        {
            "title": "Search Window",
            "formula": r"\Omega_{{{x},{y}}} = \begin{{cases}} \text{{Full Image}} & \text{{if search_size = 'full'}} \\ {search_size} \times {search_size} \text{{ window}} & \text{{otherwise}} \end{{cases}}",
            "explanation": "The search window Ω_{{{x},{y}}} is where we look for similar patches. {search_window_description}"
        },
        {   
            "title": "NLM Calculation",
            "formula": r"NLM_{{{x},{y}}} = \frac{{1}}{{W_{{{x},{y}}}}} \sum_{{(i,j) \in \Omega_{{{x},{y}}}}} I_{{i,j}} \cdot w_{{{x},{y}}}(i,j) = {nlm_value:.3f}",
            "explanation": "The final NLM value for pixel (x,y) is a weighted average of pixels in the search window, normalized by the sum of weights."
        }
    ]
}

@njit
def calculate_weight(center_patch: np.ndarray, comparison_patch: np.ndarray, filter_strength: float) -> float:
    """Calculate the weight for a patch comparison."""
    distance = np.sum((center_patch - comparison_patch)**2)
    return np.exp(-distance / (filter_strength ** 2))

@njit
def process_pixel(center_row: int, center_col: int, image: np.ndarray, denoised_image: np.ndarray, 
                  weight_sum_map: np.ndarray, kernel_size: int, search_size: Optional[int], 
                  filter_strength: float, height: int, width: int) -> None:
    """Process a single pixel for NLM denoising."""
    half_kernel = kernel_size // 2
    center_patch = image[center_row-half_kernel:center_row+half_kernel+1, center_col-half_kernel:center_col+half_kernel+1]
    
    denoised_value = 0.0
    weight_sum = 0.0

    search_y_start, search_y_end = get_search_range(center_row, height, search_size)
    search_x_start, search_x_end = get_search_range(center_col, width, search_size)

    for i in range(search_y_start, search_y_end):
        for j in range(search_x_start, search_x_end):
            if not is_valid_pixel(i, j, half_kernel, height, width):
                continue
            
            comparison_patch = image[i-half_kernel:i+half_kernel+1, j-half_kernel:j+half_kernel+1]
            weight = calculate_weight(center_patch, comparison_patch, filter_strength)
            
            denoised_value += image[i, j] * weight
            weight_sum += weight
            weight_sum_map[i, j] += weight

    denoised_image[center_row, center_col] = denoised_value / weight_sum if weight_sum > 0 else image[center_row, center_col]

@njit
def get_search_range(center: int, dimension: int, search_size: Optional[int]) -> Tuple[int, int]:
    """Calculate the search range for a given dimension."""
    if search_size is None:
        return 0, dimension
    else:
        start = max(0, center - search_size // 2)
        end = min(dimension, center + search_size // 2 + 1)
        return start, end

@njit
def is_valid_pixel(i: int, j: int, half_kernel: int, height: int, width: int) -> bool:
    """Check if a pixel is valid for processing."""
    return (half_kernel <= j < width - half_kernel) and (half_kernel <= i < height - half_kernel)

@st.cache_data(persist=True)
def process_nlm(image: np.ndarray, kernel_size: int, max_pixels: int, search_window_size: int, 
                filter_strength: float) -> Dict[str, Any]:
    """Process the image using Non-Local Means denoising."""
    details = calculate_processing_details(image, kernel_size, max_pixels)
    
    denoised_image, weight_sum_map = apply_nlm(image, kernel_size, search_window_size, filter_strength, 
                                               details['pixels_to_process'], details['height'], details['width'], 
                                               details['first_x'], details['first_y'])
    
    max_weight = np.max(weight_sum_map)
    normalized_weight_map = weight_sum_map / max_weight if max_weight > 0 else weight_sum_map

    return {
        'processed_image': denoised_image,
        'normalized_weight_map': normalized_weight_map,
        'first_pixel': (details['first_x'], details['first_y']),
        'max_weight': weight_sum_map[details['first_y'], details['first_x']],
        'additional_info': {
            'kernel_size': kernel_size,
            'pixels_processed': details['pixels_to_process'],
            'image_dimensions': (details['height'], details['width']),
            'search_window_size': search_window_size,
            'filter_strength': filter_strength
        }
    }

@njit(parallel=True)
def apply_nlm(image: np.ndarray, kernel_size: int, search_size: Optional[int], filter_strength: float, 
              pixels_to_process: int, height: int, width: int, first_x: int, first_y: int) -> Tuple[np.ndarray, np.ndarray]:
    """Apply Non-Local Means denoising to the image."""
    denoised_image = np.zeros((height, width), dtype=np.float32)
    weight_sum_map = np.zeros((height, width), dtype=np.float32)

    for pixel in range(pixels_to_process):
        row = first_y + pixel // (width - kernel_size + 1)
        col = first_x + pixel % (width - kernel_size + 1)
        process_pixel(row, col, image, denoised_image, weight_sum_map, kernel_size, search_size, filter_strength, height, width)

    return denoised_image, weight_sum_map