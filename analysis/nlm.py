import numpy as np
from numba import njit, prange
from dataclasses import dataclass
from typing import List 
from shared_types import FilterResult, calculate_processing_details
import logging

logger = logging.getLogger(__name__)

#------------------------------------------------------------------------------
# Constants
#------------------------------------------------------------------------------

DEFAULT_SEARCH_WINDOW_SIZE = 21
DEFAULT_FILTER_STRENGTH = 10.0

#------------------------------------------------------------------------------
# NLM Formula Configuration
#------------------------------------------------------------------------------

NLM_FORMULA_CONFIG = {
    "title": "Non-Local Means (NLM) Denoising",
    "variables": {
        "h": "{filter_strength}",
        "patch_size": "{kernel_size}",
        "search_size": "{search_window_size}"
    },
    "main_formula": r"I_{{{x},{y}}} = {original_value:.3f} \quad \rightarrow \quad NLM_{{{x},{y}}} = \frac{{1}}{{W_{{{x},{y}}}}} \sum_{{i,j \in \Omega_{{{x},{y}}}}} I_{{i,j}} \cdot w_{{{x},{y}}}(i,j) = {nlm_value:.3f}",
    "explanation": r"""
    The Non-Local Means (NLM) algorithm denoises each pixel by replacing it with a weighted average of pixels from the entire image (or a large search window). The weights are determined by the similarity of small patches around each pixel:
    
    1. Patch Comparison: For each pixel $(x,y)$, compare the patch $P_{{{x},{y}}}$ to patches $P_{{i,j}}$ around other pixels $(i,j)$.
    2. Weight Calculation: Based on the patch similarity, calculate a weight $w_{{{x},{y}}}(i,j)$ for each comparison.  
    3. Weighted Average: Use these weights to compute the NLM value $NLM_{{{x},{y}}}$, a weighted average of pixel intensities $I_{{i,j}}$.
    
    This process transitions the original pixel intensity $I_{{{x},{y}}}$ to the denoised value $NLM_{{{x},{y}}}$.
    """,
    "additional_formulas": [
        {
            "title": "Neighborhood Analysis",
            "formula": r"\text{{Patch Size: }} {patch_size} \times {patch_size}"
                       r"\quad\quad\text{{Centered at: }}({x}, {y})" 
                       r"\\\\"
                       "{kernel_matrix}", 
            "explanation": r"Analysis of a ${patch_size}\times{patch_size}$ patch $P_{{{x},{y}}}$ centered at $(x,y)$ for patch comparison. Matrix shows pixel values, with the central value (bold) being the denoised pixel."
        },
        {
            "title": "Weight Calculation", 
            "formula": r"w_{{{x},{y}}}(i,j) = e^{{-\frac{{\|P_{{{x},{y}}} - P_{{i,j}}\|^2}}{{h^2}}}}",
            "explanation": r"""
            Weight calculation for pixel $(i,j)$ when denoising $(x,y)$ based on patch similarity, using a Gaussian weighting function:
            - $w_{{({x},{y})}}(i,j)$: Weight for pixel $(i,j)$
            - $P_{{({x},{y})}}$, $P_{{(i,j)}}$: Patches centered at $(x,y)$ and $(i,j)$
            - $\|P_{{({x},{y})}} - P_{{(i,j)}}\|^2$: Squared difference between patches
            - $h = {h}$: Smoothing parameter
            - Similar patches yield higher weights
            """
        },
        {
            "title": "Normalization Factor",
            "formula": r"W_{{{x},{y}}} = \sum_{{i,j \in \Omega_{{{x},{y}}}}} w_{{{x},{y}}}(i,j)", 
            "explanation": r"Sum of all weights for pixel $(x,y)$, ensuring the final weighted average preserves overall image brightness."
        },
        {
            "title": "Search Window",
            "formula": r"\Omega_{{{x},{y}}} = \begin{{cases}} \text{{Full Image}} & \text{{if search\_size = 'full'}} \\ {search_window_size} \times {search_window_size} \text{{ window}} & \text{{otherwise}} \end{{cases}}",
            "explanation": r"Search window $\Omega_{{({x},{y})}}$ for finding similar patches. {search_window_description}"
        },
        {   
            "title": "NLM Calculation",
            "formula": r"NLM_{{{x},{y}}} = \frac{{1}}{{W_{{{x},{y}}}}} \sum_{{i,j \in \Omega_{{{x},{y}}}}} I_{{i,j}} \cdot w_{{{x},{y}}}(i,j) = {nlm_value:.3f}",
            "explanation": r"Final NLM value for pixel $(x,y)$: weighted average of pixel intensities $I_{{i,j}}$ in the search window, normalized by the sum of weights $W_{{{x},{y}}}$."
        }
    ]
}
#------------------------------------------------------------------------------
# Core NLM Functions
#------------------------------------------------------------------------------

@njit
def calculate_weight(center_patch, comparison_patch, filter_strength):
    patch_diff = np.sum((center_patch - comparison_patch)**2)
    return np.exp(-patch_diff / (filter_strength ** 2))

@njit
def calculate_range(center, dimension, window_size):
    if window_size is None or window_size >= dimension:
        return 0, dimension
    start = max(0, center - window_size // 2)
    end = min(dimension, center + window_size // 2 + 1)
    return start, end

@njit
def get_search_ranges(center_row, center_col, height, width, search_window_size):
    y_start, y_end = calculate_range(center_row, height, search_window_size)
    x_start, x_end = calculate_range(center_col, width, search_window_size)
    return y_start, y_end, x_start, x_end

@njit
def is_pixel_within_bounds(i, j, half_kernel, height, width):
    return (half_kernel <= i < height - half_kernel) and (half_kernel <= j < width - half_kernel)

@njit
def calculate_pixel_position(pixel, valid_width, start_x, start_y):
    row, col = divmod(pixel, valid_width)
    return np.int32(start_y + row), np.int32(start_x + col)

@njit
def get_patch(image, row, col, half_kernel):
    height, width = image.shape
    patch = np.zeros((2*half_kernel+1, 2*half_kernel+1), dtype=np.float32)
    for i in range(-half_kernel, half_kernel+1):
        for j in range(-half_kernel, half_kernel+1):
            if 0 <= row+i < height and 0 <= col+j < width:
                patch[i+half_kernel, j+half_kernel] = image[row+i, col+j]
    return patch

@njit
def process_pixel(row, col, image, kernel_size, search_window_size, filter_strength):
    height, width = image.shape
    half_kernel = kernel_size // 2
    half_search = search_window_size // 2
    
    center_patch = get_patch(image, row, col, half_kernel)
    
    total_weight = 0.0
    weighted_sum = 0.0
    max_similarity = 0.0

    for i in range(max(0, row - half_search), min(height, row + half_search + 1)):
        for j in range(max(0, col - half_search), min(width, col + half_search + 1)):
            if i == row and j == col:
                continue
            
            current_patch = get_patch(image, i, j, half_kernel)
            weight = calculate_weight(center_patch, current_patch, filter_strength)
            
            total_weight += weight
            weighted_sum += weight * image[i, j]
            
            max_similarity = max(max_similarity, weight)

    denoised_value = weighted_sum / total_weight if total_weight > 0 else image[row, col]
    return denoised_value, max_similarity


@njit(parallel=True)
def apply_nlm(image, kernel_size, search_window_size, filter_strength, pixels_to_process, start_x, start_y):
    height, width = np.int32(image.shape[0]), np.int32(image.shape[1])
    valid_width = np.int32(width - kernel_size + 1)
    denoised_image = np.zeros_like(image)
    similarity_map = np.zeros((height, width), dtype=np.float32)

    for pixel in prange(pixels_to_process):
        row, col = calculate_pixel_position(np.int32(pixel), valid_width, np.int32(start_x), np.int32(start_y))
        denoised_value, local_similarity = process_pixel(row, col, image, np.int32(kernel_size), np.int32(search_window_size), filter_strength)
        denoised_image[row, col] = denoised_value
        similarity_map[row, col] = local_similarity

    return denoised_image, similarity_map

@njit
def process_pixel_value(row, col, image, kernel_size, search_window_size, filter_strength):
    denoised_value, _ = process_pixel(row, col, image, kernel_size, search_window_size, filter_strength)
    return denoised_value

def process_nlm(image: np.ndarray, kernel_size: int, pixels_to_process: int, 
                search_window_size: int = None, filter_strength: float = None) -> 'NLMResult':
    search_window_size = search_window_size or min(DEFAULT_SEARCH_WINDOW_SIZE, min(image.shape))  
    filter_strength = filter_strength or DEFAULT_FILTER_STRENGTH
    
    try:
        processing_info = calculate_processing_details(image, kernel_size, pixels_to_process)
        height, width = image.shape
        
        denoised_image, last_weight_map = apply_nlm(
            image.astype(np.float32), kernel_size, search_window_size, filter_strength,
            processing_info.pixels_to_process, processing_info.start_x, processing_info.start_y
        )
        
        return NLMResult(
            denoised_image=denoised_image,
            last_weight_map=last_weight_map,
            processing_coord=(processing_info.start_x, processing_info.start_y),
            processing_end_coord=(processing_info.end_x, processing_info.end_y),
            kernel_size=kernel_size,
            pixels_processed=processing_info.pixels_to_process,
            image_dimensions=(height, width),
            search_window_size=search_window_size,
            filter_strength=filter_strength
        )
    except Exception as e:
        logger.error(f"Error in process_nlm: {type(e).__name__}: {e}", exc_info=True)
        raise

@dataclass
class NLMResult(FilterResult):
    denoised_image: np.ndarray
    last_weight_map: np.ndarray
    search_window_size: int
    filter_strength: float

    @classmethod
    def get_filter_options(cls) -> List[str]:
        return ["NL-Means Image", "Weight Map"]

    def get_filter_data(self) -> dict:
        return {
            "NL-Means Image": self.denoised_image,
            "Weight Map": self.last_weight_map
        }