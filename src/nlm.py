# import itertools
import numpy as np
from numba import njit, prange
from dataclasses import dataclass
from typing import List, Tuple
from src.utils import FilterResult, calculate_processing_details, ProcessingDetails, Point
import logging

# --- Logger Setup ---
logger = logging.getLogger(__name__)

# --- Patch Calculation Functions ---

@njit
def calculate_patch_difference(center_patch: np.ndarray, comparison_patch: np.ndarray) -> float:
    return np.sum((center_patch - comparison_patch) ** 2)

@njit
def calculate_weight(patch_difference: float, filter_strength: float) -> float:
    return np.exp(-patch_difference / (filter_strength ** 2))

@njit
def get_patch(image: np.ndarray, row: int, col: int, half_kernel: int) -> np.ndarray:
    # sourcery skip: use-itertools-product
    height, width = image.shape
    patch = np.zeros((2 * half_kernel + 1, 2 * half_kernel + 1), dtype=image.dtype)

    for i in range(-half_kernel, half_kernel + 1):
        for j in range(-half_kernel, half_kernel + 1):
            patch_row = row + i
            patch_col = col + j
            if 0 <= patch_row < height and 0 <= patch_col < width:
                patch[i + half_kernel, j + half_kernel] = image[patch_row, patch_col]

    return patch

# --- NLM Calculation Function ---
@njit
def calculate_nlm_value(row: int, col: int, image: np.ndarray, kernel_size: int, search_window_size: int, filter_strength: float) -> Tuple[float, float]:
    # sourcery skip: use-itertools-product
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
            patch_difference = calculate_patch_difference(center_patch, comparison_patch)
            weight = calculate_weight(patch_difference, filter_strength)

            total_weight += weight
            weighted_sum += weight * image[i, j]

    nlm_value = weighted_sum / total_weight if total_weight > 0 else image[row, col]

    return nlm_value, total_weight
# --- NLM Application Function ---

@njit(parallel=True)
def apply_nlm(image: np.ndarray, kernel_size: int, search_window_size: int, filter_strength: float, pixels_to_process: int, start_point: Point) -> np.ndarray:
    height, width = image.shape
    valid_width = width - kernel_size + 1
    
    nonlocal_means = np.zeros_like(image)
    total_weights = np.zeros_like(image)

    for pixel in prange(pixels_to_process):
        row = start_point.y + pixel // valid_width
        col = start_point.x + pixel % valid_width
        
        if row < height and col < width:
            nlm_value, weight = calculate_nlm_value(row, col, image, kernel_size, search_window_size, filter_strength)
            nonlocal_means[row, col] = nlm_value
            total_weights[row, col] = weight
    
    return nonlocal_means, total_weights

# --- Main Processing Function ---

def process_nlm(image: np.ndarray, kernel_size: int, pixels_to_process: int, 
                search_window_size: int = 7, filter_strength: float = 0.1) -> 'NLMResult':
    try:
        print(f"Received search_window_size: {search_window_size}")
        print(f"Received filter_strength: {filter_strength}")
        
        # Input validation
        if kernel_size is None or kernel_size <= 0:
            raise ValueError(f"kernel_size must be a positive integer. Received: {kernel_size} (type: {type(kernel_size).__name__})")
        if search_window_size is None or search_window_size <= 0:
            raise ValueError(f"search_window_size must be a positive integer. Received: {search_window_size} (type: {type(search_window_size).__name__})")
        if filter_strength <= 0:
            raise ValueError(f"filter_strength must be a positive float. Received: {filter_strength} (type: {type(filter_strength).__name__})")


        processing_info: ProcessingDetails = calculate_processing_details(image, kernel_size, pixels_to_process)

        nonlocal_means, total_weights = apply_nlm(
            image.astype(np.float32), kernel_size, search_window_size, filter_strength,
            processing_info.pixels_to_process, processing_info.start_point
        )
        
        return NLMResult(
            nonlocal_means=nonlocal_means,
            normalization_factors=total_weights,
            processing_coord=processing_info.start_point,  
            processing_end_coord=processing_info.end_point, 
            kernel_size=kernel_size,
            pixels_processed=processing_info.pixels_to_process,
            image_dimensions=processing_info.image_dimensions,
            search_window_size=search_window_size,
            filter_strength=filter_strength
        )
    except Exception as e:
        logger.error(f"Error in process_nlm: {type(e).__name__}: {e}", exc_info=True)
        raise

# --- Data Class for NLM Results ---

@dataclass
class NLMResult(FilterResult):
    nonlocal_means: np.ndarray
    normalization_factors: np.ndarray
    search_window_size: int
    filter_strength: float

    @classmethod
    def get_filter_options(cls) -> List[str]:
        return ["Non-Local Means", "Normalization Factors"]
    
    def get_filter_data(self) -> dict:
        return {
            "Non-Local Means": self.nonlocal_means,
            "Normalization Factors": self.normalization_factors
        }
