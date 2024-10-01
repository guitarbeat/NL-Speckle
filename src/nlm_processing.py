import numpy as np
from typing import Tuple, Dict, Any





def crop_to_min_size(P_xy: np.ndarray, P_ij: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Crop two arrays to the minimum size of both."""
    min_height = min(P_xy.shape[0], P_ij.shape[0])
    min_width = min(P_xy.shape[1], P_ij.shape[1])
    return P_xy[:min_height, :min_width], P_ij[:min_height, :min_width]



def get_search_bounds(y: int, x: int, height: int, width: int, search_radius: int) -> Tuple[int, int, int, int]:
    """Calculate the search bounds ensuring they don't go out of image bounds."""
    return (
        max(0, y - search_radius),
        min(height, y + search_radius + 1),
        max(0, x - search_radius),
        min(width, x + search_radius + 1)
    )

def get_window(image: np.ndarray, y: int, x: int, half_kernel: int, height: int, width: int) -> np.ndarray:
    """Get a window from the image, handling boundary conditions."""
    return image[max(0, y-half_kernel):min(height, y+half_kernel+1), 
                 max(0, x-half_kernel):min(width, x+half_kernel+1)]


def process_nlm_pixel(y: int, x: int, image: np.ndarray, kernel_size: int, search_window_size: int, 
                      filter_strength: float, use_full_image: bool, height: int, width: int) -> Tuple[int, int, float, float, float, float, float]:
    half_kernel = kernel_size // 2
    search_radius = search_window_size // 2 if not use_full_image else max(height, width)

    y_start, y_end, x_start, x_end = get_search_bounds(y, x, height, width, search_radius)
    P_xy = get_window(image, y, x, half_kernel, height, width)

    weights_sum = 0
    pixel_sum = 0
    max_weight = 0
    min_weight = float('inf')
    weights_squared_sum = 0

    for i in range(y_start, y_end):
        for j in range(x_start, x_end):
            if i == y and j == x:
                continue

            P_ij = get_window(image, i, j, half_kernel, height, width)
            P_xy_cropped, P_ij_cropped = crop_to_min_size(P_xy, P_ij)

            P_diff_squared_xy_ij = np.sum((P_xy_cropped - P_ij_cropped) ** 2)
            weight = np.exp(-P_diff_squared_xy_ij / (filter_strength ** 2))

            weights_sum += weight
            pixel_sum += weight * image[i, j]
            max_weight = max(max_weight, weight)
            min_weight = min(min_weight, weight)
            weights_squared_sum += weight ** 2

    if weights_sum == 0:
        return y, x, image[y, x], 0, 0, 0, 0

    nlm_value = pixel_sum / weights_sum
    weight_avg = weights_sum / ((y_end - y_start) * (x_end - x_start) - 1)
    weight_std = np.sqrt(weights_squared_sum / ((y_end - y_start) * (x_end - x_start) - 1) - weight_avg ** 2)

    return y, x, nlm_value, weight_avg, weight_std, max_weight, min_weight

def format_nlm_result(results: Tuple[np.ndarray, ...], processing_end: Tuple[int, int], kernel_size: int, pixel_count: int, image_dimensions: Tuple[int, int], search_window_size: int, filter_strength: float) -> Dict[str, Any]:
    nlm_image, normalization_factors, nl_std_image, nl_speckle_image, last_similarity_map = results
    
    return {
        'nonlocal_means': nlm_image,
        'normalization_factors': normalization_factors,
        'nonlocal_std': nl_std_image,
        'nonlocal_speckle': nl_speckle_image,
        'search_window_size': search_window_size,
        'filter_strength': filter_strength,
        'last_similarity_map': last_similarity_map,
        'processing_end_coord': processing_end,
        'kernel_size': kernel_size, 
        'pixels_processed': pixel_count,
        'image_dimensions': image_dimensions,
        'filter_data': {
            "Non-Local Means": nlm_image,
            "Normalization Factors": normalization_factors,
            "Last Similarity Map": last_similarity_map,
            "Non-Local Standard Deviation": nl_std_image,
            "Non-Local Speckle": nl_speckle_image,
        }
    }