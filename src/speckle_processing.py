import numpy as np
from typing import Tuple, Dict, Any

def ensure_within_bounds(y: int, x: int, height: int, width: int, half_kernel: int) -> Tuple[int, int]:
    """Ensure that the given coordinates are within the image bounds."""
    return (
        max(half_kernel, min(y, height - half_kernel - 1)),
        max(half_kernel, min(x, width - half_kernel - 1))
    )

def get_window(image: np.ndarray, y: int, x: int, half_kernel: int, height: int, width: int) -> np.ndarray:
    """Get a window from the image, handling boundary conditions."""
    return image[max(0, y-half_kernel):min(height, y+half_kernel+1), 
                 max(0, x-half_kernel):min(width, x+half_kernel+1)]

def process_speckle_pixel(y: int, x: int, image: np.ndarray, kernel_size: int, 
                          processing_origin: Tuple[int, int], height: int, width: int, valid_width: int) -> Tuple[int, int, float, float, float]:
    half_kernel = kernel_size // 2
    
    y, x = ensure_within_bounds(y, x, height, width, half_kernel)
    
    local_window = get_window(image, y, x, half_kernel, height, width)
    local_mean = np.nanmean(local_window)
    local_std = np.nanstd(local_window)
    sc = local_std / local_mean if local_mean != 0 else 0
    return y, x, local_mean, local_std, sc

def format_speckle_result(result_images: Tuple[np.ndarray, ...], processing_end: Tuple[int, int], kernel_size: int, pixels_processed: int, image_dimensions: Tuple[int, int]) -> Dict[str, Any]:
    mean_filter, std_dev_filter, speckle_contrast_filter = result_images
    return {
        'mean_filter': mean_filter,
        'std_dev_filter': std_dev_filter,
        'speckle_contrast_filter': speckle_contrast_filter,
        'processing_end_coord': processing_end,
        'kernel_size': kernel_size,
        'pixels_processed': pixels_processed,
        'image_dimensions': image_dimensions,
        'filter_data': {
            'Mean Filter': mean_filter,
            'Std Dev Filter': std_dev_filter,
            'Speckle Contrast': speckle_contrast_filter
        }
    }