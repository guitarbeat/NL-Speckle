import numpy as np
from numba import njit, prange
from dataclasses import dataclass
from typing import List 
from shared_types import FilterResult, calculate_processing_details
import logging

logger = logging.getLogger(__name__)

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
    """
    Calculate the weight between two patches based on their similarity.
    
    Args:
        center_patch: The patch centered on the pixel being denoised
        comparison_patch: Another patch being compared to the center patch
        filter_strength: Controls the decay of the exponential function
    
    Returns:
        The calculated weight (higher for more similar patches)
    """
    distance = np.sum((center_patch - comparison_patch)**2)
    return np.exp(-distance / (filter_strength ** 2))

@njit
def get_search_range(center, dimension, search_window_size):
    """
    Determine the search range for a given dimension.
    
    Args:
        center: The center coordinate
        dimension: The size of the image in this dimension
        search_window_size: The size of the search window
    
    Returns:
        A tuple of (start, end) indices for the search range
    """
    if search_window_size is None or search_window_size >= dimension:
        return 0, dimension
    start = max(0, center - search_window_size // 2)
    end = min(dimension, center + search_window_size // 2 + 1)
    return start, end

@njit
def is_valid_pixel(i, j, half_kernel, height, width):
    """
    Check if a pixel is valid (i.e., not too close to the image border).
    
    Args:
        i, j: Pixel coordinates
        half_kernel: Half the size of the kernel
        height, width: Image dimensions
    
    Returns:
        Boolean indicating if the pixel is valid
    """
    return (half_kernel <= j < width - half_kernel) and (half_kernel <= i < height - half_kernel)

@njit
def process_pixel(center_row, center_col, image, denoised_image, kernel_size, search_window_size, 
                  filter_strength, height, width):
    """
    Process a single pixel using the NLM algorithm.
    
    Args:
        center_row, center_col: Coordinates of the pixel being processed
        image: Input image
        denoised_image: Output denoised image
        kernel_size: Size of the patch
        search_window_size: Size of the search window
        filter_strength: Strength of the filter
        height, width: Image dimensions
    
    Returns:
        Tuple of (denoised_value, pixel_weight_map)
    """
    half_kernel = kernel_size // 2
    center_patch = image[center_row-half_kernel:center_row+half_kernel+1, center_col-half_kernel:center_col+half_kernel+1]
    
    denoised_value = np.float32(0.0)
    weight_sum = np.float32(0.0)
    pixel_weight_map = np.zeros((height, width), dtype=np.float32)

    search_y_start, search_y_end = get_search_range(center_row, height, search_window_size)
    search_x_start, search_x_end = get_search_range(center_col, width, search_window_size)

    for i in range(search_y_start, search_y_end):
        for j in range(search_x_start, search_x_end):
            if not is_valid_pixel(i, j, half_kernel, height, width):
                continue
            
            comparison_patch = image[i-half_kernel:i+half_kernel+1, j-half_kernel:j+half_kernel+1]
            weight = calculate_weight(center_patch, comparison_patch, filter_strength)
            
            denoised_value += image[i, j] * weight
            weight_sum += weight
            pixel_weight_map[i, j] = weight

    denoised_value = denoised_value / weight_sum if weight_sum > 0 else image[center_row, center_col]
    denoised_image[center_row, center_col] = denoised_value
    return denoised_value, pixel_weight_map

#------------------------------------------------------------------------------
# Main NLM Processing Functions
#------------------------------------------------------------------------------

def process_nlm(image, kernel_size, pixels_to_process, search_window_size, filter_strength):
    """
    Main function to process an image using the NLM algorithm.
    
    Args:
        image: Input image
        kernel_size: Size of the patch
        pixels_to_process: Number of pixels to process
        search_window_size: Size of the search window
        filter_strength: Strength of the filter
    
    Returns:
        NLMResult object containing the denoised image and other metadata
    """
    try:
        processing_info = calculate_processing_details(image, kernel_size, pixels_to_process)
        
        height, width = processing_info.image_height, processing_info.image_width
        denoised_image = np.zeros((height, width), dtype=np.float32)
        
        end_pixel_weight_map = apply_nlm(
            image.astype(np.float32), denoised_image, kernel_size, search_window_size, np.float32(filter_strength), 
            processing_info.pixels_to_process, height, width, 
            processing_info.start_x, processing_info.start_y
        )
        
        start_x, start_y = processing_info.start_x, processing_info.start_y
        end_x, end_y = processing_info.end_x, processing_info.end_y
        normalized_weight_map = end_pixel_weight_map / np.max(end_pixel_weight_map) if np.max(end_pixel_weight_map) > 0 else end_pixel_weight_map
        difference_map = np.abs(denoised_image - image.astype(np.float32))

        return NLMResult(
            denoised_image=denoised_image,
            weight_map_for_end_pixel=normalized_weight_map,
            difference_map=difference_map,
            processing_coord=(start_x, start_y),
            processing_end_coord=(end_x, end_y),
            kernel_size=kernel_size,
            pixels_processed=processing_info.pixels_to_process,
            image_dimensions=(height, width),
            search_window_size=search_window_size,
            filter_strength=filter_strength
        )
    except Exception as e:
        logger.error(f"Error in process_nlm: {e}", exc_info=True)
        raise

@njit(parallel=True)
def apply_nlm(image, denoised_image, kernel_size, search_window_size, filter_strength, pixels_to_process, height, width, start_x, start_y):
    """
    Apply the NLM algorithm to the image using parallel processing.
    
    Args:
        image: Input image
        denoised_image: Output denoised image
        kernel_size: Size of the patch
        search_window_size: Size of the search window
        filter_strength: Strength of the filter
        pixels_to_process: Number of pixels to process
        height, width: Image dimensions
        start_x, start_y: Starting coordinates for processing
    
    Returns:
        Weight map for the last processed pixel
    """
    valid_width = width - kernel_size + 1
    end_pixel_weight_map = np.zeros((height, width), dtype=np.float32)
    for pixel in prange(pixels_to_process):
        row = start_y + pixel // valid_width
        col = start_x + pixel % valid_width
        denoised_value, weight_map = process_pixel(row, col, image, denoised_image, kernel_size, search_window_size, filter_strength, height, width)
        denoised_image[row, col] = denoised_value
        if pixel == pixels_to_process - 1:
            end_pixel_weight_map = weight_map
    return end_pixel_weight_map

#------------------------------------------------------------------------------
# Result Class
#------------------------------------------------------------------------------

@dataclass
class NLMResult(FilterResult):
    denoised_image: np.ndarray
    weight_map_for_end_pixel: np.ndarray
    difference_map: np.ndarray
    search_window_size: int
    filter_strength: float

    @classmethod
    def get_filter_options(cls) -> List[str]:
        return ["NL-Means Image", "Weight Distribution", "Difference Image"]

    def get_filter_data(self) -> dict:
        return {
            "NL-Means Image": self.denoised_image,
            "Weight Distribution": self.weight_map_for_end_pixel,
            "Difference Image": self.difference_map
        }