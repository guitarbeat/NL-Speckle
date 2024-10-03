import numpy as np
from typing import Tuple, List
from numba import njit
import streamlit as st
from multiprocessing import Pool, cpu_count
## LSCI-specific Functions ##

@njit
def extract_window(y_coord: int, x_coord: int, kernel_size: int, image: np.ndarray) -> np.ndarray:
    """
    Extract a window from the image centered at (y_coord, x_coord) with the given kernel size.
    Pads with edge values if necessary.
    """
    half_kernel = kernel_size // 2
    height, width = image.shape

    top = max(0, y_coord - half_kernel)
    bottom = min(height, y_coord + half_kernel + 1)
    left = max(0, x_coord - half_kernel)
    right = min(width, x_coord + half_kernel + 1)

    window = image[top:bottom, left:right]

    return window

@njit
def compute_statistics(window: np.ndarray) -> Tuple[float, float]:
    """Compute mean and standard deviation of the given window."""
    mean = np.nanmean(window)
    std = np.nanstd(window)
    return mean, std


class LSCIProcessor:
    def __init__(self, image: np.ndarray, kernel_size: int):
        self.image = image
        self.kernel_size = kernel_size

    def process_pixel(self, y_center: int, x_center: int) -> Tuple[int, int, float, float, float]:
        patch_center = extract_window(y_center, x_center, self.kernel_size, self.image)
        mean_intensity, std_intensity = compute_statistics(patch_center)
        speckle_contrast = std_intensity / mean_intensity if mean_intensity != 0 else 0
        return y_center, x_center, float(mean_intensity), float(std_intensity), float(speckle_contrast)


## NLM-specific Functions ##
@st.cache_data
@njit
def calculate_weights(patch_xy: np.ndarray, patch_ij: np.ndarray, filter_strength: float) -> float:
    """Calculate the weight between two patches using Numba for speed."""
    squared_diff = 0.0
    for i in range(patch_xy.shape[0]):
        for j in range(patch_xy.shape[1]):
            diff = patch_xy[i, j] - patch_ij[i, j]
            squared_diff += diff * diff
    weight = np.exp(-squared_diff / (filter_strength ** 2))
    return weight

def extract_search_window(y_center: int, x_center: int, 
                          image_shape: Tuple[int, int], 
                          search_window_size: int, 
                          use_full_image: bool) -> Tuple[int, int, int, int]:
    """
    Extract the search window coordinates.
    
    Args:
        y_center, x_center: Center coordinates
        image_shape: Shape of the image (height, width)
        search_window_size: Size of the search window
        use_full_image: Whether to use the full image as search window
    
    Returns:
        Tuple of (y_start, y_end, x_start, x_end)
    """
    height, width = image_shape
    search_radius = max(height, width) if use_full_image else search_window_size // 2
    
    y_start = max(0, y_center - search_radius)
    y_end = min(height, y_center + search_radius + 1)
    x_start = max(0, x_center - search_radius)
    x_end = min(width, x_center + search_radius + 1)
    
    return y_start, y_end, x_start, x_end

@st.cache_data
def calculate_weights_for_neighbors(center_patch: np.ndarray, image: np.ndarray,
                                    search_window: Tuple[int, int, int, int],
                                    kernel_size: int, filter_strength: float) -> Tuple[List[float], List[Tuple[int, int]]]:
    """
    Calculate weights for neighboring patches within the search window.
    
    Args:
        center_patch: The patch centered at the pixel being processed
        image: The full image array
        search_window: Tuple of (y_start, y_end, x_start, x_end)
        kernel_size: Size of the kernel for patch extraction
        filter_strength: Strength of the filter (h parameter in NLM)
    
    Returns:
        Tuple of (weights, neighbor_coords)
    """
    y_start, y_end, x_start, x_end = search_window
    weights = []
    neighbor_coords = []
    
    for y_neighbor in range(y_start, y_end):
        for x_neighbor in range(x_start, x_end):
            neighbor_patch = extract_window(y_neighbor, x_neighbor, kernel_size, image)
            weight = calculate_weights(center_patch, neighbor_patch, filter_strength)
            weights.append(weight)
            neighbor_coords.append((y_neighbor, x_neighbor))
    
    return weights, neighbor_coords

class NLMProcessor:
    def __init__(self, image: np.ndarray, kernel_size: int, search_window_size: int, use_full_image: bool, filter_strength: float):
        self.image = image
        self.kernel_size = kernel_size
        self.search_window_size = search_window_size
        self.use_full_image = use_full_image
        self.filter_strength = filter_strength
        self.height, self.width = image.shape

    def process_pixel(self, coords: Tuple[int, int]) -> Tuple[int, int, float, float, np.ndarray]:
        y_center, x_center = coords
        search_window = extract_search_window(
            y_center, x_center,
            (self.height, self.width),
            self.search_window_size,
            self.use_full_image
        )
        
        patch_center = extract_window(y_center, x_center, self.kernel_size, self.image)
        
        weights, neighbor_coords = calculate_weights_for_neighbors(
            patch_center,
            self.image,
            search_window,
            self.kernel_size,
            self.filter_strength
        )
        
        similarity_map = np.zeros_like(self.image)
        for weight, (y, x) in zip(weights, neighbor_coords):
            similarity_map[y, x] = weight
        
        if not weights:
            return y_center, x_center, float(self.image[y_center, x_center]), 0.0, similarity_map

        nlm_value, average_weight = self._compute_nlm_values(y_center, x_center, weights, neighbor_coords)
        
        return y_center, x_center, float(nlm_value), float(average_weight), similarity_map

    def _compute_nlm_values(self, y_center: int, x_center: int, weights: List[float], neighbor_coords: List[Tuple[int, int]]) -> Tuple[float, float]:
        normalization_factor = sum(weights)
        weighted_sum = sum(w * self.image[y, x] for w, (y, x) in zip(weights, neighbor_coords))
        
        nlm_value = weighted_sum / normalization_factor if normalization_factor > 0 else self.image[y_center, x_center]
        average_weight = normalization_factor / len(weights) if weights else 0.0
        
        return float(nlm_value), float(average_weight)

    def process_image_parallel(self):
        coords = [(y, x) for y in range(self.height) for x in range(self.width)]
        with Pool(processes=cpu_count()) as pool:
            results = pool.map(self.process_pixel, coords)
        return results
