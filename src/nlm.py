"""
This module provides the implementation of Non-Local Means (NLM) denoising
algorithm functions.
"""

from dataclasses import dataclass
from typing import List, Tuple
import streamlit as st
import numpy as np

from src.processing import FilterResult, ProcessingDetails, calculate_processing_details

# --- Patch Calculation Functions ---


def gaussian_kernel(x: float, sigma: float) -> float:
    """
The gaussian_kernel function computes the Gaussian kernel value for a given distance and filtering parameter (sigma). This is used to compute the weights based on patch similarity.
    """
    return np.exp(-x**2 / (2 * sigma**2))


def patch_distance(patch1: np.ndarray, patch2: np.ndarray) -> float:
    """
The patch_distance function computes the Euclidean distance between two patches. This distance is used to determine the similarity between patches.
    """
    return np.sqrt(np.sum((patch1 - patch2)**2))


def get_patch(image: np.ndarray, row: int, col: int, patch_size: int) -> np.ndarray:
    """
    Extracts a square patch from the image centered at the given row and column.
    """
    half_patch = patch_size // 2
    return image[row-half_patch:row+half_patch+1, col-half_patch:col+half_patch+1]


# --- NLM Calculation Function ---

def calculate_nlm_value(
    row: int,
    col: int,
    image: np.ndarray,
    patch_size: int,
    search_window: int,
    h: float,
    use_full_image: bool, # Add this parameter
) -> Tuple[float, float]:
    """
    Calculates the NLM denoised value for a pixel in the image.
    """
    height, width = image.shape
    pad_size = search_window // 2
    padded_image = np.pad(image, pad_size, mode='reflect')
    
    center_patch = get_patch(padded_image, row+pad_size, col+pad_size, patch_size)
    
    weighted_sum = 0.0
    normalizing_factor = 0.0
    similarity_map = np.zeros_like(image)
    
    if use_full_image:
        row_range = range(height)
        col_range = range(width)
    else:
        row_range = range(max(0, row - pad_size), min(height, row + pad_size + 1))
        col_range = range(max(0, col - pad_size), min(width, col + pad_size + 1))

    for k in row_range:
        for neighbor_col in col_range:
            if k == row and neighbor_col == col:
                continue
            
            neighbor_patch = get_patch(padded_image, k+pad_size, neighbor_col+pad_size, patch_size)
            distance = patch_distance(center_patch, neighbor_patch)
            similarity_map[k, neighbor_col] = distance

            weight = gaussian_kernel(distance, h)
            
            weighted_sum += weight * padded_image[k+pad_size, neighbor_col+pad_size]
            normalizing_factor += weight
    
    nlm_value = weighted_sum / normalizing_factor if normalizing_factor > 0 else image[row, col]
    
    return nlm_value, normalizing_factor,similarity_map

# --- NLM Application Function ---


@st.cache_resource 
def apply_nlm(
    image: np.ndarray,
    patch_size: int,
    search_window: int,
    h: float,
    pixels_to_process: int,
    processing_origin: Tuple[int, int]
) -> np.ndarray:
    """
    Applies the Non-Local Means algorithm to the entire image.

    Args:
        image (np.ndarray): The input image.
        patch_size (int): The size of the patch used for similarity comparison.
        search_window (int): The size of the search window around each pixel.
        h (float): The filtering parameter controlling the decay of the weights.
        pixels_to_process (int): The number of pixels to process.
        processing_origin (Tuple[int, int]): The starting point for processing.

    Returns:
        np.ndarray: The denoised image and total weights for normalization.
    """
    height, width = image.shape
    valid_width = width - patch_size + 1

    nonlocal_means = np.zeros_like(image)
    total_weights = np.zeros_like(image)
    last_similarity_map = None

    use_full_image =  st.session_state.get("use_full_image")

    for pixel in range(pixels_to_process):
        row = processing_origin[1] + pixel // valid_width
        col = processing_origin[0] + pixel % valid_width

        if row < height and col < width:
            nlm_value, weight, similarity_map = calculate_nlm_value(
                row, col, image, patch_size, search_window, h,
                use_full_image
            )
            nonlocal_means[row, col] = nlm_value
            total_weights[row, col] = weight
            last_similarity_map = similarity_map

    return nonlocal_means, total_weights, last_similarity_map


# --- Main Processing Function ---


def process_nlm(
    image: np.ndarray,
    kernel_size: int,
    pixels_to_process: int,
    search_window_size: int = 21,
    filter_strength: float = 10.0,
) -> "NLMResult":
    """
    Main function to execute the NLM denoising on the input image.

    Args:
        image (np.ndarray): The input image.
        patch_size (int): The size of the patch used for similarity comparison.
        pixels_to_process (int): The number of pixels to process.
        search_window (int, optional): The size of the search window around each pixel. Defaults to 21.
        h (float, optional): The filtering parameter controlling the decay of the weights. Defaults to 10.0.

    Returns:
        NLMResult: The result of the NLM denoising.
    """
    try:
        processing_info: ProcessingDetails = calculate_processing_details(
            image, kernel_size, pixels_to_process
        )

        # Ensure image is a NumPy array before converting
        nonlocal_means, total_weights, last_similarity_map = apply_nlm(
            # Ensure this is a float32 array
            np.asarray(image, dtype=np.float32),
            kernel_size,
            search_window_size,
            filter_strength,
            processing_info.pixels_to_process,
            processing_info.processing_origin,  # Pass the config here
        )
        return NLMResult(
            nonlocal_means=nonlocal_means,
            normalization_factors=total_weights,
            processing_end_coord=processing_info.processing_end,
            kernel_size=kernel_size,
            pixels_processed=processing_info.pixels_to_process,
            image_dimensions=processing_info.image_dimensions,
            search_window_size=search_window_size,
            filter_strength=filter_strength,
            last_similarity_map=last_similarity_map,
        )
    except Exception:
        st.error("An error occurred during NLM processing.")
        raise


# --- Data Class for NLM Results ---


@dataclass
class NLMResult(FilterResult):
    """
    Data class to hold the result of the NLM denoising algorithm.
    """

    nonlocal_means: np.ndarray
    normalization_factors: np.ndarray
    search_window_size: int
    filter_strength: float
    last_similarity_map: List[np.ndarray]

    @staticmethod
    def get_filter_options() -> List[str]:
        """
        Returns the available filter options for NLM results.

        Returns:
            List[str]: The list of available filter options.
        """
        return ["Non-Local Means", "Normalization Factors", "Last Similarity Map"]

    def get_filter_data(self) -> dict:
        """
        Provides the NLM result data as a dictionary.

        Returns:
            dict: The result data.
        """
        return {
            "Non-Local Means": self.nonlocal_means,
            "Normalization Factors": self.normalization_factors,
            "Last Similarity Map": self.last_similarity_map,
        }