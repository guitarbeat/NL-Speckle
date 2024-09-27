"""
This module provides the implementation of Non-Local Means (NLM) denoising
algorithm functions.
"""

from typing import Tuple
import numpy as np
import streamlit as st
from dataclasses import dataclass
from typing import List, Dict
from src.utils import BaseResult
from functools import lru_cache
from multiprocessing import Pool

# --- Patch Calculation Functions ---

@lru_cache(maxsize=None)
def calculate_weight(P_diff_squared_xy_ij: float, h: float) -> float:
    weight_xy_ij = np.exp(-(P_diff_squared_xy_ij) / (h**2))
    return weight_xy_ij

def calculate_patch_distance(P_xy: np.ndarray, P_ij: np.ndarray) -> float:
    P_diff_squared_xy_ij = (np.sum((P_xy - P_ij) ** 2))
    return P_diff_squared_xy_ij

def extract_patch(image: np.ndarray, x: int, y: int, patch_size: int) -> np.ndarray:
    half_patch = patch_size // 2
    P_xy = image[x - half_patch : x + half_patch + 1, y - half_patch : y + half_patch + 1]
    return P_xy

# --- NLM Calculation Function ---

def calculate_nlm(weighted_intensity_sum: float, C: float, original_pixel_value: float) -> float:
    """
    Calculate the Non-Local Means value for a pixel.

    Args:
        weighted_intensity_sum (float): Sum of weighted intensities
        C (float): Normalization factor
        original_pixel_value (float): Original pixel value

    Returns:
        float: Non-Local Means value
    """
    return weighted_intensity_sum / C if C > 0 else original_pixel_value

def calculate_nlstd(weighted_intensity_sum: float, weighted_intensity_squared_sum: float, C: float) -> float:
    """
    Calculate the Non-Local Standard Deviation.

    Args:
        weighted_intensity_sum (float): Sum of weighted intensities
        weighted_intensity_squared_sum (float): Sum of weighted squared intensities
        C (float): Normalization factor

    Returns:
        float: Non-Local Standard Deviation
    """
    if C > 0:
        mean = weighted_intensity_sum / C
        variance = (weighted_intensity_squared_sum / C) - (mean ** 2)
        return np.sqrt(max(0, variance))
    return 0

def calculate_nlsc(nlstd: float, nlm: float) -> float:
    """
    Calculate the Non-Local Speckle Contrast.

    Args:
        nlstd (float): Non-Local Standard Deviation
        nlm (float): Non-Local Means value

    Returns:
        float: Non-Local Speckle Contrast
    """
    return nlstd / nlm if nlm > 0 else 0

# --- NLM Application Function ---

def calculate_c_xy(image: np.ndarray, x: int, y: int, patch_size: int, search_window_size: int, h: float, use_full_image: bool) -> Tuple[float, float, float, np.ndarray]:
    height, width = image.shape
    half_patch = patch_size // 2
    half_search = search_window_size // 2
    similarity_map = np.zeros_like(image)

    P_xy = extract_patch(image, x, y, patch_size)
    weighted_intensity_sum_xy = 0.0
    weighted_intensity_squared_sum_xy = 0.0
    C_xy = 0.0

    # Determine the range of pixels to process
    if use_full_image:
        x_range = range(half_patch, height - half_patch)
        y_range = range(half_patch, width - half_patch)
    else:
        x_range = range(max(half_patch, x - half_search), min(height - half_patch, x + half_search + 1))
        y_range = range(max(half_patch, y - half_search), min(width - half_patch, y + half_search + 1))

    for i in x_range:
        for j in y_range:
            if i == x and j == y:
                continue

            # Ensure we can extract a valid patch for comparison
            if i - half_patch < 0 or i + half_patch >= height or j - half_patch < 0 or j + half_patch >= width:
                continue

            P_ij = extract_patch(image, i, j, patch_size)
            P_diff_squared_xy_ij = calculate_patch_distance(P_xy, P_ij)
            weight_xy_ij = calculate_weight(P_diff_squared_xy_ij, h)
            similarity_map[i, j] = weight_xy_ij
            neighbor_pixel = image[i, j]
            weighted_intensity_sum_xy += weight_xy_ij * neighbor_pixel
            weighted_intensity_squared_sum_xy += weight_xy_ij * (neighbor_pixel ** 2)
            C_xy += weight_xy_ij

    return C_xy, weighted_intensity_sum_xy, weighted_intensity_squared_sum_xy, similarity_map

def process_pixel(args):
    x, y, image, patch_size, search_window_size, h, use_full_image = args
    
    height, width = image.shape
    half_patch = patch_size // 2
    
    # Ensure we can extract a valid patch
    if x - half_patch < 0 or x + half_patch >= height or y - half_patch < 0 or y + half_patch >= width:
        return x, y, image[x, y], 0, 0, 0, np.zeros_like(image)

    C_xy, weighted_intensity_sum_xy, weighted_intensity_squared_sum_xy, similarity_map = calculate_c_xy(
        image, x, y, patch_size, search_window_size, h, use_full_image
    )

    if C_xy > 0:
        NLM_xy = calculate_nlm(weighted_intensity_sum_xy, C_xy, image[x, y])
        NLstd_xy = calculate_nlstd(weighted_intensity_sum_xy, weighted_intensity_squared_sum_xy, C_xy)
        NLSC_xy = calculate_nlsc(NLstd_xy, NLM_xy)
    else:
        NLM_xy = image[x, y]
        NLstd_xy = 0
        NLSC_xy = 0

    return x, y, NLM_xy, C_xy, NLstd_xy, NLSC_xy, similarity_map

@st.cache_resource()
def apply_nlm_to_image(
    image: np.ndarray,
    patch_size: int,
    search_window_size: int,
    h: float,
    pixels_to_process: int,
    processing_origin: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    height, width = image.shape
    valid_width = width - patch_size + 1

    NLM_image = np.zeros_like(image)
    C_xy_image = np.zeros_like(image)
    NLstd_image = np.zeros_like(image)
    NLSC_xy_image = np.zeros_like(image)
    
    use_full_image = st.session_state.get("use_full_image")

    # Prepare arguments for parallel processing
    args_list = []
    for pixel in range(pixels_to_process):
        x = processing_origin[1] + pixel // valid_width
        y = processing_origin[0] + pixel % valid_width
        if x < height and y < width:
            args_list.append((x, y, image, patch_size, search_window_size, h, use_full_image))

    # Process in chunks
    chunk_size = 1000  # Adjust this value based on your memory constraints
    progress_bar = st.progress(0)
    status_text = st.empty()
    with Pool() as pool:
        for i in range(0, len(args_list), chunk_size):
            chunk = args_list[i:i+chunk_size]
            try:
                results = pool.map(process_pixel, chunk)
                for x, y, NLM_xy, C_xy, NLstd_xy, NLSC_xy, similarity_map in results:
                    NLM_image[x, y] = NLM_xy
                    NLstd_image[x, y] = NLstd_xy
                    NLSC_xy_image[x, y] = NLSC_xy
                    C_xy_image[x, y] = C_xy
                    last_similarity_map = similarity_map  # This will be overwritten in each iteration
                progress = (i + len(chunk)) / len(args_list)
                progress_bar.progress(progress)
                status_text.text(f"Processed {i + len(chunk)} pixels out of {len(args_list)}")
            except Exception as e:
                st.error(f"Error processing chunk starting at pixel {i}: {str(e)}")

    progress_bar.empty()
    status_text.empty()

    return NLM_image, NLstd_image, NLSC_xy_image, C_xy_image, last_similarity_map

def process_nlm(
    image: np.ndarray,
    kernel_size: int,
    pixels_to_process: int,
    search_window_size: int = 21,
    h: float = 10.0,
    start_pixel: int = 0
) -> "NLMResult":
    try:
        height, width = image.shape
        half_kernel = kernel_size // 2
        valid_height, valid_width = height - kernel_size + 1, width - kernel_size + 1
        total_valid_pixels = valid_height * valid_width
        
        # Ensure we don't process beyond the valid pixels
        end_pixel = min(start_pixel + pixels_to_process, total_valid_pixels)
        pixels_to_process = end_pixel - start_pixel

        # Calculate starting coordinates
        start_y, start_x = divmod(start_pixel, valid_width)
        start_y += half_kernel
        start_x += half_kernel

        NLM_image, NLstd_image, NLSC_xy_image, C_xy_image, last_similarity_map = apply_nlm_to_image(
            np.asarray(image, dtype=np.float32),
            kernel_size,
            search_window_size,
            h,
            pixels_to_process,
            (start_x, start_y)
        )

        # Calculate processing end coordinates
        end_y, end_x = divmod(end_pixel - 1, valid_width)
        end_y, end_x = end_y + half_kernel, end_x + half_kernel
        processing_end = (min(end_x, width - 1), min(end_y, height - 1))

        return NLMResult(
            nonlocal_means=NLM_image,
            normalization_factors=C_xy_image,
            nonlocal_std=NLstd_image,
            nonlocal_speckle=NLSC_xy_image,
            processing_end_coord=processing_end,
            kernel_size=kernel_size,
            pixels_processed=pixels_to_process,
            image_dimensions=(height, width),
            search_window_size=search_window_size,
            filter_strength=h,
            last_similarity_map=last_similarity_map,
        )
    except Exception as e:
        st.error(f"An error occurred during NLM processing: {str(e)}")
        raise

# --- Data Class for NLM Results ---

@dataclass
class NLMResult(BaseResult):
    nonlocal_means: np.ndarray
    normalization_factors: np.ndarray
    nonlocal_std: np.ndarray
    nonlocal_speckle: np.ndarray
    search_window_size: int
    filter_strength: float
    last_similarity_map: np.ndarray

    @staticmethod
    def get_filter_options() -> List[str]:
        return [
            "Non-Local Means",
            "Normalization Factors",
            "Last Similarity Map",
            "Non-Local Standard Deviation",
            "Non-Local Speckle",
        ]

    def get_filter_data(self) -> Dict[str, np.ndarray]:
        return {
            "Non-Local Means": self.nonlocal_means,
            "Normalization Factors": self.normalization_factors,
            "Last Similarity Map": self.last_similarity_map,
            "Non-Local Standard Deviation": self.nonlocal_std,
            "Non-Local Speckle": self.nonlocal_speckle,
        }

    @classmethod
    def combine(cls, results: List["NLMResult"]) -> "NLMResult":
        if not results:
            return cls.empty_result()

        combined_arrays = {
            attr: np.maximum.reduce([getattr(r, attr) for r in results])
            for attr in ['nonlocal_means', 'normalization_factors', 'nonlocal_std', 'nonlocal_speckle']
        }

        return cls(
            **combined_arrays,
            **BaseResult.combine(results).__dict__,
            search_window_size=results[0].search_window_size,
            filter_strength=results[0].filter_strength,
            last_similarity_map=results[-1].last_similarity_map,
        )

    @classmethod
    def empty_result(cls) -> "NLMResult":
        return cls(
            nonlocal_means=np.array([]),
            normalization_factors=np.array([]),
            nonlocal_std=np.array([]),
            nonlocal_speckle=np.array([]),
            processing_end_coord=(0, 0),
            kernel_size=0,
            pixels_processed=0,
            image_dimensions=(0, 0),
            search_window_size=0,
            filter_strength=0,
            last_similarity_map=np.array([]),
        )
