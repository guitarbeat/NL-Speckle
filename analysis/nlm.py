import numpy as np
from numba import njit, prange
from utils import calculate_processing_details
from dataclasses import dataclass

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

# Core NLM Functions

@njit
def calculate_weight(center_patch, comparison_patch, filter_strength):
    """Calculate the weight for a patch comparison."""
    distance = np.sum((center_patch - comparison_patch)**2)
    return np.exp(-distance / (filter_strength ** 2))

@njit
def process_pixel(center_row, center_col, image, denoised_image, weight_map, kernel_size, search_window_size, 
                  filter_strength, height, width):
    """Process a single pixel for NLM denoising."""
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

    denoised_image[center_row, center_col] = denoised_value / weight_sum if weight_sum > 0 else image[center_row, center_col]
    weight_map[center_row, center_col] = pixel_weight_map

    # Comments explaining the weight map:
    # The weight_map stores similarity scores for each pixel compared to the center pixel.
    # - A weight close to 1 indicates high similarity (patches are very alike).
    # - A weight close to 0 indicates low similarity (patches are very different).
    # - The center pixel always has the highest similarity to itself (weight = 1).
    # - Weights are relative and depend on the filter_strength parameter.
    # - Higher filter_strength results in more uniform weights, lower leads to more distinct differences.

@njit
def get_search_range(center, dimension, search_window_size):
    """Calculate the search range for a given dimension."""
    if search_window_size is None or search_window_size >= dimension:
        return 0, dimension
    else:
        start = max(0, center - search_window_size // 2)
        end = min(dimension, center + search_window_size // 2 + 1)
        return start, end

@njit
def is_valid_pixel(i, j, half_kernel, height, width):
    """Check if a pixel is valid for processing."""
    return (half_kernel <= j < width - half_kernel) and (half_kernel <= i < height - half_kernel)

# Main NLM Processing Functions

# @st.cache_data(persist=True, show_spinner="Retrieving cached data...")
def process_nlm(image, kernel_size, max_pixels, search_window_size, filter_strength):
    """Process the image using Non-Local Means denoising."""
    processing_info = calculate_processing_details(image, kernel_size, max_pixels)
    
    # Pre-allocate arrays
    height, width = processing_info.image_height, processing_info.image_width
    denoised_image = np.zeros((height, width), dtype=np.float32)
    weight_map = np.zeros((height, width, height, width), dtype=np.float32)
    
    apply_nlm(
        image.astype(np.float32), denoised_image, weight_map, kernel_size, search_window_size, np.float32(filter_strength), 
        processing_info.pixels_to_process, height, width, 
        processing_info.start_x, processing_info.start_y
    )
    
    end_x, end_y = processing_info.end_x, processing_info.end_y
    end_pixel_weight_map = weight_map[end_y, end_x]
    normalized_weight_map = end_pixel_weight_map / np.max(end_pixel_weight_map) if np.max(end_pixel_weight_map) > 0 else end_pixel_weight_map
    difference_map = np.abs(denoised_image - image.astype(np.float32))

    return NLMResult(
        denoised_image=denoised_image,
        weight_map_for_end_pixel=normalized_weight_map,
        difference_map=difference_map,
        processing_end_coord=(end_x, end_y),
        end_pixel_max_weight=np.max(end_pixel_weight_map),
        kernel_size=kernel_size,
        pixels_processed=processing_info.pixels_to_process,
        image_dimensions=(height, width),
        search_window_size=search_window_size,
        filter_strength=filter_strength
    )


@njit(parallel=True)
def apply_nlm(image, denoised_image, weight_map, kernel_size, search_window_size, filter_strength, pixels_to_process, height, width, start_x, start_y):
    """Apply Non-Local Means denoising to the image."""
    valid_width = width - kernel_size + 1
    for pixel in prange(pixels_to_process):  # Use prange for parallelization
        row = start_y + pixel // valid_width
        col = start_x + pixel % valid_width
        process_pixel(row, col, image, denoised_image, weight_map, kernel_size, search_window_size, filter_strength, height, width)


@dataclass
class NLMResult:
    denoised_image: np.ndarray
    weight_map_for_end_pixel: np.ndarray
    difference_map: np.ndarray
    processing_end_coord: tuple[int, int]
    end_pixel_max_weight: float
    kernel_size: int
    pixels_processed: int
    image_dimensions: tuple[int, int]
    search_window_size: int | None
    filter_strength: float