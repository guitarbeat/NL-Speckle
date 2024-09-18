import numpy as np
from numba import njit, prange
from dataclasses import dataclass
from typing import List, Tuple
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
def calculate_weight(center_patch: np.ndarray, comparison_patch: np.ndarray, filter_strength: float) -> float:
    """Calculate the weight between two patches based on their similarity."""
    patch_diff = np.sum((center_patch - comparison_patch) ** 2)
    return np.exp(-patch_diff / (filter_strength ** 2))

@njit
def get_patch(image: np.ndarray, row: int, col: int, half_kernel: int) -> np.ndarray:
    """Extract a patch from the image centered at the given coordinates."""
    height, width = image.shape
    patch = np.zeros((2 * half_kernel + 1, 2 * half_kernel + 1), dtype=np.float32)
    
    for i in range(-half_kernel, half_kernel + 1):
        for j in range(-half_kernel, half_kernel + 1):
            if 0 <= row + i < height and 0 <= col + j < width:
                patch[i + half_kernel, j + half_kernel] = image[row + i, col + j]
    
    return patch

@njit
def process_pixel(row: int, col: int, image: np.ndarray, kernel_size: int, search_window_size: int, filter_strength: float) -> Tuple[float, float]:
    """Process a single pixel using the NLM algorithm."""
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
            
            current_patch = get_patch(image, i, j, half_kernel)
            weight = calculate_weight(center_patch, current_patch, filter_strength)
            
            total_weight += weight
            weighted_sum += weight * image[i, j]
    
    denoised_value = weighted_sum / total_weight if total_weight > 0 else image[row, col]
    return denoised_value, total_weight

@njit(parallel=True)
def apply_nlm(image: np.ndarray, kernel_size: int, search_window_size: int, filter_strength: float, pixels_to_process: int, start_x: int, start_y: int) -> np.ndarray:
    """Apply the NLM algorithm to the image."""
    height, width = image.shape
    valid_width = width - kernel_size + 1
    denoised_image = np.zeros_like(image)
    total_weights = np.zeros_like(image)
    
    for pixel in prange(pixels_to_process):
        row = start_y + pixel // valid_width
        col = start_x + pixel % valid_width
        
        if row < height and col < width:
            denoised_value, weight = process_pixel(row, col, image, kernel_size, search_window_size, filter_strength)
            denoised_image[row, col] = denoised_value
            total_weights[row, col] = weight
    
    return denoised_image, total_weights

def process_nlm(image: np.ndarray, kernel_size: int, pixels_to_process: int, search_window_size: int = None, filter_strength: float = None) -> 'NLMResult':
    """Process the image using the NLM algorithm."""
    search_window_size = search_window_size or min(DEFAULT_SEARCH_WINDOW_SIZE, min(image.shape))  
    filter_strength = filter_strength or DEFAULT_FILTER_STRENGTH
    
    try:
        processing_info = calculate_processing_details(image, kernel_size, pixels_to_process)
        height, width = image.shape
        
        denoised_image, total_weights = apply_nlm(
            image.astype(np.float32), kernel_size, search_window_size, filter_strength,
            processing_info.pixels_to_process, processing_info.start_x, processing_info.start_y
        )
        
        return NLMResult(
            denoised_image=denoised_image,
            total_weights=total_weights,
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
    total_weights: np.ndarray
    search_window_size: int
    filter_strength: float

    @classmethod
    def get_filter_options(cls) -> List[str]:
        return ["Non-Local Means", "NLM Weights"]
    
    def get_filter_data(self) -> dict:
        return {
            "Non-Local Means": self.denoised_image,
            "NLM Weights": self.total_weights
        }