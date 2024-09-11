import numpy as np
from typing import Tuple, Dict, Any, List
from numba import njit
from utils import generate_kernel_matrix, display_formula_section, display_additional_formulas, calculate_processing_details,visualize_image
from cache_manager import cached_db



SPECKLE_FORMULA_CONFIG = {
    "main_formula": r"I_{{{x},{y}}} = {original_value:.3f} \quad \rightarrow \quad SC_{{{x},{y}}} = \frac{{\sigma_{{{x},{y}}}}}{{\mu_{{{x},{y}}}}} = \frac{{{std:.3f}}}{{{mean:.3f}}} = {sc:.3f}",
    "explanation": r"This formula shows the transition from the original pixel intensity $I_{{{x},{y}}}$ to the Speckle Contrast (SC) for the same pixel position.",
    "additional_formulas": [
        {
            "title": "Neighborhood Analysis",
            "formula": r"\text{{Kernel Size: }} {kernel_size} \times {kernel_size}"
                       r"\quad\quad\text{{Centered at pixel: }}({x}, {y})"
                       r"\\\\"
                       "{kernel_matrix}",
            "explanation": r"Analysis of a ${kernel_size}\times{kernel_size}$ neighborhood centered at pixel $({x},{y})$. The matrix shows pixel values, with the central value (in bold) being the processed pixel."
        },
        {
            "title": "Mean Calculation", 
            "formula": r"\mu_{{{x},{y}}} = \frac{{1}}{{N}} \sum_{{i,j \in K_{{{x},{y}}}}} I_{{i,j}} = \frac{{1}}{{{kernel_size}^2}} \sum_{{i,j \in K_{{{x},{y}}}}} I_{{i,j}} = {mean:.3f}",
            "explanation": r"Mean ($\mu$) calculation: average intensity of all pixels in the kernel $K$ centered at $({x},{y})$. $N = {kernel_size}^2 = {total_pixels}$."
        },
        {
            "title": "Standard Deviation Calculation",
            "formula": r"\sigma_{{{x},{y}}} = \sqrt{{\frac{{1}}{{N}} \sum_{{i,j \in K_{{{x},{y}}}}} (I_{{i,j}} - \mu_{{{x},{y}}})^2}} = \sqrt{{\frac{{1}}{{{kernel_size}^2}} \sum_{{i,j \in K_{{{x},{y}}}}} (I_{{i,j}} - {mean:.3f})^2}} = {std:.3f}",
            "explanation": r"Standard deviation ($\sigma$) calculation: measure of intensity spread around the mean for all pixels in the kernel $K$ centered at $({x},{y})$."
        },
        {
            "title": "Speckle Contrast Calculation",
            "formula": r"SC_{{{x},{y}}} = \frac{{\sigma_{{{x},{y}}}}}{{\mu_{{{x},{y}}}}} = \frac{{{std:.3f}}}{{{mean:.3f}}} = {sc:.3f}",
            "explanation": r"Speckle Contrast (SC): ratio of standard deviation to mean intensity within the kernel centered at $({x},{y})$."
        }
    ]
}

@njit
def calculate_speckle_stats(local_window: np.ndarray) -> Tuple[float, float, float]:
    """Calculate mean, standard deviation, and speckle contrast for a local window."""
    local_mean = np.mean(local_window)
    local_std = np.std(local_window)
    speckle_contrast = local_std / local_mean if local_mean != 0 else 0
    return local_mean, local_std, speckle_contrast

@njit
def process_pixel(row: int, col: int, image: np.ndarray, mean_filter: np.ndarray, 
                  std_dev_filter: np.ndarray, sc_filter: np.ndarray, kernel_size: int) -> None:
    """Process a single pixel for speckle contrast calculation."""
    half_kernel = kernel_size // 2
    local_window = image[row-half_kernel:row+half_kernel+1, col-half_kernel:col+half_kernel+1]
    local_mean, local_std, speckle_contrast = calculate_speckle_stats(local_window)

    mean_filter[row, col] = local_mean
    std_dev_filter[row, col] = local_std
    sc_filter[row, col] = speckle_contrast

@njit
def apply_speckle_contrast(image: np.ndarray, kernel_size: int, pixels_to_process: int, 
                           height: int, width: int, first_x: int, first_y: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply speckle contrast calculation to the image."""
    mean_filter = np.zeros((height, width), dtype=np.float32)
    std_dev_filter = np.zeros((height, width), dtype=np.float32)
    sc_filter = np.zeros((height, width), dtype=np.float32)

    for pixel in range(pixels_to_process):
        row = first_y + pixel // (width - kernel_size + 1)
        col = first_x + pixel % (width - kernel_size + 1)
        process_pixel(row, col, image, mean_filter, std_dev_filter, sc_filter, kernel_size)

    return mean_filter, std_dev_filter, sc_filter

@cached_db
# @st.cache_data(persist=True)
def process_speckle(image: np.ndarray, kernel_size: int, max_pixels: int) -> Dict[str, Any]:
    """Process the image using Speckle Contrast calculation."""
    details = calculate_processing_details(image, kernel_size, max_pixels)
    
    mean_filter, std_dev_filter, sc_filter = apply_speckle_contrast(
        image, kernel_size, details['pixels_to_process'], details['height'], details['width'], 
        details['first_x'], details['first_y']
    )

    return {
        'mean_filter': mean_filter,
        'std_dev_filter': std_dev_filter,
        'speckle_contrast_filter': sc_filter,
        'first_pixel': (details['first_x'], details['first_y']),
        'first_pixel_stats': {
            'mean': mean_filter[details['first_y'], details['first_x']],
            'std_dev': std_dev_filter[details['first_y'], details['first_x']],
            'speckle_contrast': sc_filter[details['first_y'], details['first_x']]
        },
        'additional_info': {
            'kernel_size': kernel_size,
            'pixels_processed': details['pixels_to_process'],
            'image_dimensions': (details['height'], details['width'])
        }
    }


def prepare_speckle_variables(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    variables = kwargs.copy()
    
    if 'input_x' not in variables or 'input_y' not in variables:
        kernel_size = variables.get('kernel_size', 3)
        variables['input_x'] = variables['x'] - kernel_size // 2
        variables['input_y'] = variables['y'] - kernel_size // 2

    if 'kernel_size' in variables and 'kernel_matrix' in variables:
        variables['kernel_matrix'] = generate_kernel_matrix(variables['kernel_size'], variables['kernel_matrix'])

    return variables

def display_speckle_formula(formula_placeholder: Any, **kwargs):
    with formula_placeholder.container():
        variables = prepare_speckle_variables(kwargs)
        display_formula_section(SPECKLE_FORMULA_CONFIG, variables, 'main')
        display_additional_formulas(SPECKLE_FORMULA_CONFIG, variables)


#------ Display ------#

def prepare_speckle_filter_options_and_params(
    results: Dict[str, Any], 
    first_pixel: Tuple[int, int]
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    first_x, first_y = first_pixel
    
    return {
        "Mean Filter": results['mean_filter'],
        "Std Dev Filter": results['std_dev_filter'],
        "Speckle Contrast": results['speckle_contrast_filter']
    }, {
        "std": results['first_pixel_stats']['std_dev'],
        "mean": results['first_pixel_stats']['mean'],
        "sc": results['first_pixel_stats']['speckle_contrast'],
        "total_pixels": results['additional_info']['pixels_processed']
    }

def visualize_speckle_results(
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

    visualize_image(image_np, placeholders['original_image'], first_x, first_y, kernel_size, cmap, 
                    show_full_processed, vmin, vmax, "Original Image", "speckle")
    
    if not show_full_processed:
        visualize_image(image_np, placeholders['zoomed_original_image'], first_x, first_y, kernel_size, 
                        cmap, show_full_processed, vmin, vmax, "Zoomed-In Original Image", zoom=True)

    filter_options, specific_params = prepare_speckle_filter_options_and_params(results, (first_x, first_y))
    
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

    display_speckle_formula(placeholders['formula'], **specific_params)