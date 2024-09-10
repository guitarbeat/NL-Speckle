import numpy as np
from typing import Optional, Tuple, Dict, Any, List
from numba import njit
from utils import generate_kernel_matrix, display_formula_section, display_additional_formulas
import streamlit as st
from utils import calculate_processing_details,visualize_image

NLM_FORMULA_CONFIG = {
    "main_formula": r"I_{{{x},{y}}} = {original_value:.3f} \quad \rightarrow \quad NLM_{{{x},{y}}} = \frac{{1}}{{W_{{{x},{y}}}}} \sum_{{(i,j) \in \Omega_{{{x},{y}}}}} I_{{i,j}} \cdot w_{{{x},{y}}}(i,j) = {nlm_value:.3f}",
    "explanation": "This formula shows the transition from the original pixel intensity I({x},{y}) to the denoised value NLM({x},{y}) using the Non-Local Means (NLM) algorithm.",
    "variables": {},  # To be filled dynamically
    "additional_formulas": [
        {
            "title": "Neighborhood Analysis",
            "formula": r"\text{{Patch Size: }} {kernel_size} \times {kernel_size}"
                       r"\quad\quad\text{{Centered at pixel: }}({x}, {y})"
                       r"\\\\"
                       "{kernel_matrix}",
            "explanation": "We analyze a {kernel_size}x{kernel_size} patch centered around the pixel ({x},{y}). This matrix shows the pixel values in the patch. The central value (in bold) corresponds to the pixel being denoised."
        },
        {
            "title": "Weight Calculation",
            "formula": r"w_{{{x},{y}}}(i,j) = \exp\left(-\frac{{\|P_{{{x},{y}}} - P_{{i,j}}\|^2}}{{h^2}}\right)",
            "explanation": r"""
            This formula determines the weight of each pixel (i,j) when denoising pixel (x,y):
            - w_{{{x},{y}}}(i,j): Weight assigned to pixel (i,j) when denoising (x,y) 
            - P_{{{x},{y}}} and P_{{i,j}} are patches centered at (x,y) and (i,j)
            - \|P_{{{x},{y}}} - P_{{i,j}}\|^2 measures the squared difference between patches
            - h = {filter_strength} controls the smoothing strength
            - More similar patches result in higher weights
            """
        },
        {
            "title": "Normalization Factor",
            "formula": r"W_{{{x},{y}}} = \sum_{{(i,j) \in \Omega_{{{x},{y}}}}} w_{{{x},{y}}}(i,j)", 
            "explanation": "We sum all weights for pixel (x,y). This ensures the final weighted average preserves the overall image brightness."
        },
        {
            "title": "Search Window",
            "formula": r"\Omega_{{{x},{y}}} = \begin{{cases}} \text{{Full Image}} & \text{{if search_size = 'full'}} \\ {search_size} \times {search_size} \text{{ window}} & \text{{otherwise}} \end{{cases}}",
            "explanation": "The search window Ω_{{{x},{y}}} is where we look for similar patches. {search_window_description}"
        },
        {   
            "title": "NLM Calculation",
            "formula": r"NLM_{{{x},{y}}} = \frac{{1}}{{W_{{{x},{y}}}}} \sum_{{(i,j) \in \Omega_{{{x},{y}}}}} I_{{i,j}} \cdot w_{{{x},{y}}}(i,j) = {nlm_value:.3f}",
            "explanation": "The final NLM value for pixel (x,y) is a weighted average of pixels in the search window, normalized by the sum of weights."
        }
    ]
}

@njit
def calculate_weight(center_patch: np.ndarray, comparison_patch: np.ndarray, filter_strength: float) -> float:
    """Calculate the weight for a patch comparison."""
    distance = np.sum((center_patch - comparison_patch)**2)
    return np.exp(-distance / (filter_strength ** 2))

@njit
def process_pixel(center_row: int, center_col: int, image: np.ndarray, denoised_image: np.ndarray, 
                  weight_sum_map: np.ndarray, kernel_size: int, search_size: Optional[int], 
                  filter_strength: float, height: int, width: int) -> None:
    """Process a single pixel for NLM denoising."""
    half_kernel = kernel_size // 2
    center_patch = image[center_row-half_kernel:center_row+half_kernel+1, center_col-half_kernel:center_col+half_kernel+1]
    
    denoised_value = 0.0
    weight_sum = 0.0

    search_y_start, search_y_end = get_search_range(center_row, height, search_size)
    search_x_start, search_x_end = get_search_range(center_col, width, search_size)

    for i in range(search_y_start, search_y_end):
        for j in range(search_x_start, search_x_end):
            if not is_valid_pixel(i, j, half_kernel, height, width):
                continue
            
            comparison_patch = image[i-half_kernel:i+half_kernel+1, j-half_kernel:j+half_kernel+1]
            weight = calculate_weight(center_patch, comparison_patch, filter_strength)
            
            denoised_value += image[i, j] * weight
            weight_sum += weight
            weight_sum_map[i, j] += weight

    denoised_image[center_row, center_col] = denoised_value / weight_sum if weight_sum > 0 else image[center_row, center_col]

@njit
def get_search_range(center: int, dimension: int, search_size: Optional[int]) -> Tuple[int, int]:
    """Calculate the search range for a given dimension."""
    if search_size is None:
        return 0, dimension
    else:
        start = max(0, center - search_size // 2)
        end = min(dimension, center + search_size // 2 + 1)
        return start, end

@njit
def is_valid_pixel(i: int, j: int, half_kernel: int, height: int, width: int) -> bool:
    """Check if a pixel is valid for processing."""
    return (half_kernel <= j < width - half_kernel) and (half_kernel <= i < height - half_kernel)

@st.cache_data(persist=True)
def process_nlm(image: np.ndarray, kernel_size: int, max_pixels: int, search_window_size: int, 
                filter_strength: float) -> Dict[str, Any]:
    """Process the image using Non-Local Means denoising."""
    details = calculate_processing_details(image, kernel_size, max_pixels)
    
    denoised_image, weight_sum_map = apply_nlm(image, kernel_size, search_window_size, filter_strength, 
                                               details['pixels_to_process'], details['height'], details['width'], 
                                               details['first_x'], details['first_y'])
    
    max_weight = np.max(weight_sum_map)
    normalized_weight_map = weight_sum_map / max_weight if max_weight > 0 else weight_sum_map

    return {
        'processed_image': denoised_image,
        'normalized_weight_map': normalized_weight_map,
        'first_pixel': (details['first_x'], details['first_y']),
        'max_weight': weight_sum_map[details['first_y'], details['first_x']],
        'additional_info': {
            'kernel_size': kernel_size,
            'pixels_processed': details['pixels_to_process'],
            'image_dimensions': (details['height'], details['width']),
            'search_window_size': search_window_size,
            'filter_strength': filter_strength
        }
    }

@njit(parallel=True)
def apply_nlm(image: np.ndarray, kernel_size: int, search_size: Optional[int], filter_strength: float, 
              pixels_to_process: int, height: int, width: int, first_x: int, first_y: int) -> Tuple[np.ndarray, np.ndarray]:
    """Apply Non-Local Means denoising to the image."""
    denoised_image = np.zeros((height, width), dtype=np.float32)
    weight_sum_map = np.zeros((height, width), dtype=np.float32)

    for pixel in range(pixels_to_process):
        row = first_y + pixel // (width - kernel_size + 1)
        col = first_x + pixel % (width - kernel_size + 1)
        process_pixel(row, col, image, denoised_image, weight_sum_map, kernel_size, search_size, filter_strength, height, width)

    return denoised_image, weight_sum_map

def prepare_nlm_variables(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    variables = kwargs.copy()
    
    if 'input_x' not in variables or 'input_y' not in variables:
        kernel_size = variables.get('kernel_size', 3)
        variables['input_x'] = variables['x'] - kernel_size // 2
        variables['input_y'] = variables['y'] - kernel_size // 2

    if 'kernel_size' in variables and 'kernel_matrix' in variables:
        variables['kernel_matrix'] = generate_kernel_matrix(variables['kernel_size'], variables['kernel_matrix'])

    search_size = variables.get('search_size')
    variables['search_window_description'] = (
        "We search the entire image for similar pixels." if search_size == "full" 
        else f"A search window of size {search_size}x{search_size} centered around the target pixel."
    )

    return variables

def display_nlm_formula(formula_placeholder: Any, **kwargs):
    with formula_placeholder.container():
        variables = prepare_nlm_variables(kwargs)
        display_formula_section(NLM_FORMULA_CONFIG, variables, 'main')
        display_additional_formulas(NLM_FORMULA_CONFIG, variables)



#------ Display ------#

def prepare_nlm_filter_options_and_params(
    results: Dict[str, Any], 
    first_pixel: Tuple[int, int], 
    filter_strength: float, 
    search_window_size: int
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    first_x, first_y = first_pixel
    
    return {
        "NL-Means Image": results['processed_image'],
        "Weight Map": results['normalized_weight_map'],
        "Difference Map": np.abs(results['processed_image'] - results['additional_info']['image_dimensions'][0])
    }, {
        "filter_strength": filter_strength,
        "search_size": search_window_size,
        "total_pixels": results['additional_info']['pixels_processed'],
        "nlm_value": results['processed_image'][first_y, first_x]
    }

def visualize_nlm_results(
    image_np: np.ndarray,
    results: Dict[str, Any],
    placeholders: Dict[str, Any],
    params: Dict[str, Any],
    first_pixel: Tuple[int, int],
    kernel_size: int,
    kernel_matrix: List[List[float]],
    original_value: float
):
    first_x, first_y = first_pixel
    vmin, vmax = np.min(image_np), np.max(image_np)
    show_full_processed = params['show_full_processed']
    cmap = params['analysis_params']['cmap']
    search_window_size = params['analysis_params'].get('search_window_size')
    filter_strength = params['analysis_params'].get('filter_strength')

    visualize_image(image_np, placeholders['original_image'], first_x, first_y, kernel_size, cmap, 
                    show_full_processed, vmin, vmax, "Original Image", "nlm", search_window_size)
    
    if not show_full_processed:
        visualize_image(image_np, placeholders['zoomed_original_image'], first_x, first_y, kernel_size, 
                        cmap, show_full_processed, vmin, vmax, "Zoomed-In Original Image", zoom=True)

    filter_options, specific_params = prepare_nlm_filter_options_and_params(
        results, (first_x, first_y), filter_strength, search_window_size
    )
    
    for filter_name, filter_data in filter_options.items():
        key = filter_name.lower().replace(" ", "_")
        if key in placeholders:
            visualize_image(filter_data, placeholders[key], first_x, first_y, kernel_size, cmap, 
                            show_full_processed, np.min(filter_data), np.max(filter_data), filter_name)
            
            if not show_full_processed:
                zoomed_key = f'zoomed_{key}'
                if zoomed_key in placeholders:
                    visualize_image(filter_data, placeholders[zoomed_key], first_x, first_y, kernel_size, 
                                    cmap, show_full_processed, np.min(filter_data), np.max(filter_data), 
                                    f"Zoomed-In {filter_name}", zoom=True)

    specific_params.update({
        'x': first_x, 'y': first_y, 'input_x': first_x, 'input_y': first_y,
        'kernel_size': kernel_size, 'kernel_matrix': kernel_matrix, 'original_value': original_value
    })

    display_nlm_formula(placeholders['formula'], **specific_params)