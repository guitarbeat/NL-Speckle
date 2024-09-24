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


def calculate_patch_difference(
    center_patch: np.ndarray, comparison_patch: np.ndarray
) -> float:
    """
    Computes the squared Euclidean distance between two image patches.

    Args:
        center_patch (np.ndarray): The reference patch at the center.
        comparison_patch (np.ndarray): The patch to compare against the center.

    Returns:
        float: The sum of squared differences between the patches.
    """
    return np.sum((center_patch - comparison_patch) ** 2)


def calculate_weight(patch_difference: float, filter_strength: float) -> float:
    """
    Calculates the weight for a patch comparison based on patch difference and
    filter strength.
    """
    return np.exp(-patch_difference / (filter_strength**2))


def get_patch(image: np.ndarray, row: int, col: int, half_kernel: int) -> np.ndarray:
    # sourcery skip: use-itertools-product
    """
    Extracts a square patch from the image centered at the given row and column.
    """
    height, width = image.shape
    patch = np.zeros(
        (2 * half_kernel + 1, 2 * half_kernel + 1), dtype=image.dtype)

    for i in range(-half_kernel, half_kernel + 1):
        for j in range(-half_kernel, half_kernel + 1):
            patch_row = row + i
            patch_col = col + j
            if 0 <= patch_row < height and 0 <= patch_col < width:
                patch[i + half_kernel, j +
                      half_kernel] = image[patch_row, patch_col]

    return patch


# --- NLM Calculation Function ---


def calculate_nlm_value(
    row: int,
    col: int,
    image: np.ndarray,
    kernel_size: int,
    search_window_size: int,
    filter_strength: float,
) -> Tuple[float, float]:  # sourcery skip: use-itertools-product
    """
    Calculates the NLM denoised value for a pixel in the image.
    """
    if kernel_size is None:
        raise ValueError("kernel_size cannot be None")

    height, width = image.shape
    half_kernel = kernel_size // 2
    half_search = search_window_size // 2

    center_patch = get_patch(image, row, col, half_kernel)

    total_weight = 0.0
    weighted_sum = 0.0
    
    for i in range(max(0, row - half_search), min(height, row + half_search + 1)):
        for j in range(max(0, col - half_search), min(width, col + half_search + 1)):
            if i == row and j == col:
                continue

            comparison_patch = get_patch(image, i, j, half_kernel)
            patch_difference = calculate_patch_difference(
                center_patch, comparison_patch
            )
            weight = calculate_weight(patch_difference, filter_strength)

            total_weight += weight
            weighted_sum += weight * image[i, j]

    nlm_value = weighted_sum / \
        total_weight if total_weight > 0 else image[row, col]

    return nlm_value, total_weight


# --- NLM Application Function ---


@st.cache_data
def apply_nlm(
    image: np.ndarray,
    kernel_size: int,
    search_window_size: int,
    filter_strength: float,
    pixels_to_process: int,
    processing_origin: Tuple[int, int],
) -> np.ndarray:
    """
    Applies the Non-Local Means algorithm to the entire image.

    Args:
        image (np.ndarray): The input image. kernel_size (int): The size of the
        patch kernel. search_window_size (int): The size of the search window.
        filter_strength (float): The filter strength for patch comparison.
        pixels_to_process (int): The number of pixels to process. processing_origin
        (Point): The starting point for processing.

    Returns:
        np.ndarray: The denoised image and total weights for normalization.
    """
    height, width = image.shape
    valid_width = width - kernel_size + 1

    nonlocal_means = np.zeros_like(image)
    total_weights = np.zeros_like(image)

    for pixel in range(pixels_to_process):
        row = processing_origin[1] + pixel // valid_width  # Use indexing
        col = processing_origin[0] + pixel % valid_width  # Use indexing

        if row < height and col < width:
            nlm_value, weight = calculate_nlm_value(
                row, col, image, kernel_size, search_window_size, filter_strength
            )
            nonlocal_means[row, col] = nlm_value
            total_weights[row, col] = weight

    return nonlocal_means, total_weights


# --- Main Processing Function ---

def process_nlm(
    image: np.ndarray,
    kernel_size: int,
    pixels_to_process: int,
    search_window_size: int = 7,
    filter_strength: float = 0.1,
) -> "NLMResult":
    """
    Main function to execute the NLM denoising on the input image.

    Args:
        image (np.ndarray): The input image. kernel_size (int): The size of the
        patch kernel. pixels_to_process (int): The number of pixels to process.
        search_window_size (int, optional): The size of the search window.
        Defaults to 7. filter_strength (float, optional): The filter strength.
        Defaults to 0.1.

    Returns:
        NLMResult: The result of the NLM denoising.
    """
    try:
        processing_info: ProcessingDetails = calculate_processing_details(
            image, kernel_size, pixels_to_process
        )

        # Ensure image is a NumPy array before converting
        nonlocal_means, total_weights = apply_nlm(
            # Ensure this is a float32 array
            np.asarray(image, dtype=np.float32),
            kernel_size,
            search_window_size,
            filter_strength,
            processing_info.pixels_to_process,
            processing_info.processing_origin,
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

    @staticmethod
    def get_filter_options() -> List[str]:
        """
        Returns the available filter options for NLM results.

        Returns:
            List[str]: The list of available filter options.
        """
        return ["Non-Local Means", "Normalization Factors"]

    def get_filter_data(self) -> dict:
        """
        Provides the NLM result data as a dictionary.

        Returns:
            dict: The result data.
        """
        return {
            "Non-Local Means": self.nonlocal_means,
            "Normalization Factors": self.normalization_factors,
        }
