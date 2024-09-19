import itertools
import numpy as np
from numba import njit, prange
from dataclasses import dataclass
from typing import List, Tuple
from src.utils import FilterResult, calculate_processing_details, ProcessingDetails, Point
import logging

logger = logging.getLogger(__name__)

#------------------------------------------------------------------------------
# Constants
#------------------------------------------------------------------------------
DEFAULT_SEARCH_WINDOW_SIZE = 51
DEFAULT_FILTER_STRENGTH = 0.1


#------------------------------------------------------------------------------
# NLM Formula Configuration
#------------------------------------------------------------------------------


#------------------------------------------------------------------------------
# Core NLM Functions
#------------------------------------------------------------------------------

@njit
def calculate_patch_difference(center_patch: np.ndarray, comparison_patch: np.ndarray) -> float:
    """
    Calculate the squared difference between two patches.
    Formula: ||P_{(x,y)} - P_{(i,j)}||^2 

    """
    return np.sum((center_patch - comparison_patch) ** 2)

@njit
def calculate_weight(patch_difference: float, filter_strength: float) -> float: # Maybe change to "calculate_similarity"?
    """
    Calculate the weight between two patches based on their squared difference.
    Formula: w_{(x,y)}(i,j) = exp(-||P_{(x,y)} - P_{(i,j)}||^2 / h^2)
    
    This function uses a Gaussian weighting function to assign higher weights to similar patches.
    - This gives us a similarity measure between patches, with higher weights for more similar patches.
    """
    return np.exp(-patch_difference / (filter_strength ** 2))

@njit
def get_patch(image: np.ndarray, row: int, col: int, half_kernel: int) -> np.ndarray:
    """Extract a patch P_{(x,y)} from the image centered at the given coordinates (x, y)."""
    height, width = image.shape
    patch = np.zeros((2 * half_kernel + 1, 2 * half_kernel + 1), dtype=np.float32)

    for i, j in itertools.product(range(-half_kernel, half_kernel + 1), range(-half_kernel, half_kernel + 1)):
        if 0 <= row + i < height and 0 <= col + j < width:
            patch[i + half_kernel, j + half_kernel] = image[row + i, col + j]

    return patch

@njit
def calculate_nlm_value(row: int, col: int, image: np.ndarray, kernel_size: int, search_window_size: int, filter_strength: float) -> Tuple[float, float]:
    """
    Calculate the NLM value for a single pixel (x, y).
    Formula: NLM_{(x,y)} = (1 / W_{(x,y)}) * Σ_{i,j ∈ Ω_{(x,y)}} I_{i,j} * w_{(x,y)}(i,j)
    """
    height, width = image.shape
    half_kernel = kernel_size // 2
    half_search = search_window_size // 2

    center_patch = get_patch(image, row, col, half_kernel)

    total_weight = 0.0
    weighted_sum = 0.0

    for i, j in itertools.product(range(max(0, row - half_search), min(height, row + half_search + 1)), range(max(0, col - half_search), min(width, col + half_search + 1))):
        if i == row and j == col: # Skip the center pixel
            continue

        comparison_patch = get_patch(image, i, j, half_kernel) # This is the patch centered at the pixel in the search window
        patch_difference = calculate_patch_difference(center_patch, comparison_patch) # This is the squared difference between the two patches
        weight = calculate_weight(patch_difference, filter_strength) # This is the weight based on the squared difference, similarity score

        total_weight += weight # This is the sum of all weights for the pixel (x, y) as a normalization factor
        weighted_sum += weight * image[i, j] # This is the weighted sum of pixel intensities for the pixel (x, y)

    nlm_value = weighted_sum / total_weight if total_weight > 0 else image[row, col] # This is the final NLM value for the pixel (x, y)

    return nlm_value, total_weight

@njit(parallel=True)
def apply_nlm(image: np.ndarray, kernel_size: int, search_window_size: int, filter_strength: float, pixels_to_process: int, start_point: Point) -> np.ndarray:
    """Apply the NLM algorithm to the image."""
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

def process_nlm(image: np.ndarray, kernel_size: int, pixels_to_process: int, search_window_size: int = None, filter_strength: float = None) -> 'NLMResult':
    """Process the image using the NLM algorithm."""
    search_window_size = search_window_size or min(DEFAULT_SEARCH_WINDOW_SIZE, min(image.shape))
    filter_strength = filter_strength or DEFAULT_FILTER_STRENGTH
    
    try:
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
