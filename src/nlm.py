"""
This module provides the implementation of Non-Local Means (NLM) denoising
algorithm functions.
"""

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import streamlit as st

from src.processing import FilterResult, ProcessingDetails, calculate_processing_details
from memory_profiler import profile
@profile



# --- Patch Calculation Functions ---

def calculate_weight(P_diff_squared_xy_ij: float, h: float) -> float:
    # sourcery skip: inline-immediately-returned-variable
    """
    Calculates the weight w_{x,y}(i,j) based on patch similarity.
    
    Args:
        P_diff_squared_xy_ij (float): The distance between patches |P_{x,y} - P_{i,j}|^2
        h (float): The filtering parameter h controlling the decay of weights
    
    Returns:
        float: The weight w_{x,y}(i,j)
    """
    weight_xy_ij = np.exp(-(P_diff_squared_xy_ij) / (h**2))
    return weight_xy_ij

def calculate_patch_distance(P_xy: np.ndarray, P_ij: np.ndarray) -> float:
    # sourcery skip: inline-immediately-returned-variable
    """
    Computes the squared Euclidean distance between two patches |P_{x,y} - P_{i,j}|^2.
    
    Args:
        P_xy (np.ndarray): The patch P_{x,y}
        P_ij (np.ndarray): The patch P_{i,j}
    
    Returns:
        float: The squared Euclidean distance between the patches
    """
    P_diff_squared_xy_ij = (np.sum((P_xy - P_ij) ** 2))
    return P_diff_squared_xy_ij

def extract_patch(image: np.ndarray, x: int, y: int, patch_size: int) -> np.ndarray:
    # sourcery skip: inline-immediately-returned-variable
    """
    Extracts a square patch P_{x,y} from the image centered at (x, y).
    
    Args:
        image (np.ndarray): The input image I
        x (int): The center x-coordinate of the patch
        y (int): The center y-coordinate of the patch
        patch_size (int): The size of the patch
    
    Returns:
        np.ndarray: The extracted patch P_{x,y}
    """
    half_patch = patch_size // 2
    P_xy = image[x - half_patch : x + half_patch + 1, y - half_patch : y + half_patch + 1]
    return P_xy

# --- NLM Calculation Function ---

def calculate_nlm_value(
    x: int,
    y: int,
    image: np.ndarray,
    patch_size: int,
    search_window: int,
    h: float,
    use_full_image: bool,
) -> Tuple[float, float, np.ndarray]:
    """
    Calculates the NLM denoised value NLM_{x,y} for a pixel in the image.
    
    This function implements the main NLM formula:
    NLM_{x,y} = (1 / C_{x,y}) * sum_{(i,j) in Ω_{x,y}} I_{i,j} * w_{x,y}(i,j)
    
    Args:
        x (int): The x-coordinate of the current pixel
        y (int): The y-coordinate of the current pixel
        image (np.ndarray): The input image I
        patch_size (int): The size of patches for comparison
        search_window (int): The size of the search window Ω_{x,y}
        h (float): The filtering parameter h
        use_full_image (bool): Whether to use the full image as Ω_{x,y}
    
    Returns:
        Tuple[float, float, np.ndarray]: NLM_{x,y}, C_{x,y}, and similarity map
    """
    height, width = image.shape
    half_search = search_window // 2 
    padded_image = np.pad(image, half_search, mode="reflect") # this is padding the image with values from the border

    P_xy = extract_patch(padded_image, x + half_search, y + half_search, patch_size) 


    weighted_intensity_sum_xy = 0.0 # sum_{(i,j) in Ω_{x,y}} I_{i,j} * w_{x,y}(i,j)
    weighted_intensity_squared_sum_xy = 0.0 # for non-local std
    C_xy = 0.0 # C_{x,y}
    similarity_map = np.zeros_like(image)

    # Determine the range of pixels to process
    if use_full_image:
        x_range = range(height)
        y_range = range(width)
    else:
        x_range = range(max(0, x - half_search), min(height, x + half_search + 1))
        y_range = range(max(0, y - half_search), min(width, y + half_search + 1))

    # Iterate through the determined range of pixels
    for i in x_range:
        for j in y_range:
            if i == x and j == y:
                continue

            # Get the neighboring patch
            # neighbor_y is the y-coordinate of the neighbor in the search window
            # k is the x-coordinate of the neighbor in the search window
            P_ij = extract_patch(padded_image, i + half_search, j + half_search, patch_size)
            
            # Calculate the distance between patches
            P_diff_squared_xy_ij = calculate_patch_distance(P_xy, P_ij)
            
            # Calculate the weight based on patch similarity
            # w_{x,y}(i,j) = exp(-d_{x,y}(i,j) / h^2)
            weight_xy_ij = calculate_weight(P_diff_squared_xy_ij, h)
            similarity_map[i, j] = weight_xy_ij
         
            # Update the weighted sum and normalizing factor
            neighbor_pixel = padded_image[i + half_search, j + half_search]

            weighted_intensity_sum_xy += weight_xy_ij * neighbor_pixel
            weighted_intensity_squared_sum_xy += weight_xy_ij * (neighbor_pixel ** 2)
            C_xy += weight_xy_ij

    # Calculate the final denoised value
    NLM_xy = (
        weighted_intensity_sum_xy / C_xy if C_xy > 0 else image[x, y]
    )
    # Calculate non-local standard deviation
    if C_xy > 0:
        variance_xy = (weighted_intensity_squared_sum_xy / C_xy) - (NLM_xy ** 2)
        NLstd_xy = np.sqrt(max(0, variance_xy))  # max to avoid negative values due to floating-point errors
        # Calculate non-local speckle contrast
        NLSC_xy = NLstd_xy / NLM_xy if NLM_xy > 0 else 0
    else:
        NLstd_xy = 0
        NLSC_xy = 0

    return NLM_xy, C_xy, similarity_map, NLstd_xy, NLSC_xy

# --- NLM Application Function ---

@profile
@st.cache_resource
def apply_nlm_to_image(
    image: np.ndarray,
    patch_size: int,
    search_window_size: int,
    h: float,
    pixels_to_process: int,
    processing_origin: Tuple[int, int],
) -> np.ndarray:
    """
    Applies the Non-Local Means algorithm to the entire image I.
    
    This function calculates NLM_{x,y} for multiple pixels in the image.
    
    Args:
        image (np.ndarray): The input image I
        patch_size (int): The size of the patch P_{x,y}
        search_window_size (int): The size of the search window Ω_{x,y}
        h (float): The filtering parameter h
        pixels_to_process (int): The number of pixels to process
        processing_origin (Tuple[int, int]): The starting point for processing

    Returns:
        np.ndarray: The NLM image, normalization factors C_{x,y}, and last similarity map
    """
    height, width = image.shape
    valid_width = width - patch_size + 1

    NLM_image = np.zeros_like(image)
    C_xy_image = np.zeros_like(image)
    NLstd_image = np.zeros_like(image)
    NLSC_xy_image = np.zeros_like(image)
    
    use_full_image = st.session_state.get("use_full_image")

    for pixel in range(pixels_to_process):
        x = processing_origin[1] + pixel // valid_width
        y = processing_origin[0] + pixel % valid_width

        if x < height and y < width:
            NLM_xy, C_xy, similarity_map, NLstd_xy,NLSC_xy = calculate_nlm_value(
                x, y, image, patch_size, search_window_size, h, use_full_image
            )
            NLM_image[x, y] = NLM_xy
            NLstd_image[x, y] = NLstd_xy
            NLSC_xy_image[x, y] = NLSC_xy
            C_xy_image[x, y] = C_xy
            last_similarity_map = similarity_map
    
    
    return NLM_image, NLstd_image,NLSC_xy_image, C_xy_image, last_similarity_map

# --- Main Processing Function ---

def process_nlm(
    image: np.ndarray,
    kernel_size: int,
    pixels_to_process: int,
    search_window_size: int = 21,
    h: float = 10.0,
) -> "NLMResult":
    """
    Main function to execute the NLM denoising on the input image I.
    
    This function implements the complete NLM algorithm:
    1. Define patches P_{x,y} and search windows Ω_{x,y}
    2. Calculate weights w_{x,y}(i,j) based on patch similarity
    3. Compute NLM_{x,y} for each pixel
    
    Args:
        image (np.ndarray): The input image I
        kernel_size (int): The size of the patch P_{x,y}
        pixels_to_process (int): The number of pixels to process
        search_window_size (int, optional): The size of the search window Ω_{x,y}. Defaults to 21
        h (float, optional): The filtering parameter h. Defaults to 10.0

    Returns:
        NLMResult: The result of the NLM denoising
    """
    try:
        processing_info: ProcessingDetails = calculate_processing_details(
            image, kernel_size, pixels_to_process
        )

        # Apply NLM denoising
        NLM_image,NLstd_image,NLSC_xy_image, C_xy_image, last_similarity_map = apply_nlm_to_image(
            np.asarray(image, dtype=np.float32),
            kernel_size,
            search_window_size,
            h,
            processing_info.pixels_to_process,
            processing_info.processing_origin,
        )
        
        # Return the results
        return NLMResult(
            nonlocal_means=NLM_image,
            normalization_factors=C_xy_image,
            nonlocal_std=NLstd_image,
            nonlocal_speckle = NLSC_xy_image,
            processing_end_coord=processing_info.processing_end,
            kernel_size=kernel_size,
            pixels_processed=processing_info.pixels_to_process,
            image_dimensions=processing_info.image_dimensions,
            search_window_size=search_window_size,
            filter_strength=h,
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
    This class extends FilterResult to include NLM-specific attributes.
    """

    nonlocal_means: np.ndarray
    normalization_factors: np.ndarray
    search_window_size: int
    filter_strength: float
    last_similarity_map: List[np.ndarray]
    nonlocal_std: np.ndarray
    nonlocal_speckle: np.ndarray

    @staticmethod
    def get_filter_options() -> List[str]:
        """
        Returns the available filter options for NLM results.

        Returns:
            List[str]: The list of available filter options
        """
        return ["Non-Local Means", "Normalization Factors", "Last Similarity Map", "Non-Local Standard Deviation", "Non-Local Speckle"]

    def get_filter_data(self) -> dict:
        """
        Provides the NLM result data as a dictionary.

        Returns:
            dict: The result data
        """
        return {
            "Non-Local Means": self.nonlocal_means,
            "Normalization Factors": self.normalization_factors,
            "Last Similarity Map": self.last_similarity_map,
            "Non-Local Standard Deviation": self.nonlocal_std,
            "Non-Local Speckle": self.nonlocal_speckle
        }