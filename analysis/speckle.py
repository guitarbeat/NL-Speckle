import numpy as np
from numba import njit
from utils import calculate_processing_details
from dataclasses import dataclass
import logging

#------------------------------------------------------------------------------
# Constants
#------------------------------------------------------------------------------

SPECKLE_FORMULA_CONFIG = {
    "title": "Speckle Contrast Calculation",
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
# Core Speckle Contrast Calculation Functions

@njit
def calculate_speckle_stats(local_window):
    """Calculate mean, standard deviation, and speckle contrast for a local window."""
    local_mean = np.mean(local_window)
    local_std = np.std(local_window)
    speckle_contrast = local_std / local_mean if local_mean != 0 else 0
    return local_mean, local_std, speckle_contrast

@njit
def process_pixel(row, col, image, mean_filter, std_dev_filter, sc_filter, kernel_size, height, width):
    """Process a single pixel for speckle contrast calculation."""
    half_kernel = kernel_size // 2
    local_window = image[max(0, row-half_kernel):min(height, row+half_kernel+1),
                         max(0, col-half_kernel):min(width, col+half_kernel+1)]
    local_mean, local_std, speckle_contrast = calculate_speckle_stats(local_window)

    mean_filter[row, col] = local_mean
    std_dev_filter[row, col] = local_std
    sc_filter[row, col] = speckle_contrast

@njit
def apply_mean_filter(image, kernel_size, pixels_to_process, height, width, start_x, start_y):
    """Apply mean filter to the image."""
    mean_filter = np.zeros((height, width), dtype=np.float32)
    half_kernel = kernel_size // 2
    valid_width = width - kernel_size + 1

    for pixel in range(pixels_to_process):
        row = start_y + pixel // valid_width
        col = start_x + pixel % valid_width
        if row < height and col < width:
            local_window = image[max(0, row-half_kernel):min(height, row+half_kernel+1),
                                 max(0, col-half_kernel):min(width, col+half_kernel+1)]
            mean_filter[row, col] = np.mean(local_window)

    return mean_filter

@njit
def apply_std_dev_filter(image, mean_filter, kernel_size, pixels_to_process, height, width, start_x, start_y):
    """Apply standard deviation filter to the image."""
    std_dev_filter = np.zeros((height, width), dtype=np.float32)
    half_kernel = kernel_size // 2
    valid_width = width - kernel_size + 1

    for pixel in range(pixels_to_process):
        row = start_y + pixel // valid_width
        col = start_x + pixel % valid_width
        if row < height and col < width:
            local_window = image[max(0, row-half_kernel):min(height, row+half_kernel+1),
                                 max(0, col-half_kernel):min(width, col+half_kernel+1)]
            local_mean = mean_filter[row, col]
            std_dev_filter[row, col] = np.sqrt(np.mean((local_window - local_mean)**2))

    return std_dev_filter

@njit
def apply_speckle_contrast_filter(mean_filter, std_dev_filter):
    """Apply speckle contrast filter using mean and standard deviation filters."""
    return np.where(mean_filter != 0, std_dev_filter / mean_filter, 0)

@njit
def apply_speckle_contrast(image, kernel_size, pixels_to_process, height, width, start_x, start_y):
    """Apply speckle contrast calculation to the image."""
    mean_filter = apply_mean_filter(image, kernel_size, pixels_to_process, height, width, start_x, start_y)
    std_dev_filter = apply_std_dev_filter(image, mean_filter, kernel_size, pixels_to_process, height, width, start_x, start_y)
    sc_filter = apply_speckle_contrast_filter(mean_filter, std_dev_filter)

    return mean_filter, std_dev_filter, sc_filter

# Main Processing Function
# @st.cache_data(persist=True, show_spinner="Retrieving cached data...")
def process_speckle(image, kernel_size, max_pixels):
    """Process the image using Speckle Contrast calculation."""
    try:
        processing_info = calculate_processing_details(image, kernel_size, max_pixels)
        
        mean_filter, std_dev_filter, sc_filter = apply_speckle_contrast(
            image, kernel_size, processing_info.pixels_to_process, 
            processing_info.image_height, processing_info.image_width, 
            processing_info.start_x, processing_info.start_y
        )

        start_x, start_y = processing_info.start_x, processing_info.start_y

        return SpeckleResult(
            mean_filter=mean_filter,
            std_dev_filter=std_dev_filter,
            speckle_contrast_filter=sc_filter,
            processing_start_coord=(start_x, start_y),
            start_pixel_mean=mean_filter[start_y, start_x],
            start_pixel_std_dev=std_dev_filter[start_y, start_x],
            start_pixel_speckle_contrast=sc_filter[start_y, start_x],
            kernel_size=kernel_size,
            pixels_processed=processing_info.pixels_to_process,
            image_dimensions=(processing_info.image_height, processing_info.image_width)
        )
    except Exception as e:
        logging.error(f"Error in process_speckle: {str(e)}")
        return None

@dataclass
class SpeckleResult:
    mean_filter: np.ndarray
    std_dev_filter: np.ndarray
    speckle_contrast_filter: np.ndarray
    processing_start_coord: tuple[int, int]
    start_pixel_mean: float
    start_pixel_std_dev: float
    start_pixel_speckle_contrast: float
    kernel_size: int
    pixels_processed: int
    image_dimensions: tuple[int, int]