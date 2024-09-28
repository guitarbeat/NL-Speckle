"""
This module provides the implementation of Non-Local Means (NLM) denoising
algorithm functions.
"""

import numpy as np
from typing import Tuple
from functools import lru_cache

# --- Patch Calculation Functions ---


@lru_cache(maxsize=None)
def calculate_weight(P_diff_squared_xy_ij: float, h: float) -> float:
    return np.exp(-(P_diff_squared_xy_ij) / (h**2))


def calculate_patch_distance(P_xy: np.ndarray, P_ij: np.ndarray) -> float:
    return np.sum((P_xy - P_ij) ** 2)


def extract_patch(image: np.ndarray, x: int, y: int, patch_size: int) -> np.ndarray:
    half_patch = patch_size // 2
    return image[
        x - half_patch : x + half_patch + 1, y - half_patch : y + half_patch + 1
    ]


# --- NLM Calculation Functions ---


def calculate_nlm(
    weighted_intensity_sum: float, C: float, original_pixel_value: float
) -> float:
    """Calculate the Non-Local Means value for a pixel."""
    return weighted_intensity_sum / C if C > 0 else original_pixel_value


def calculate_nlstd(
    weighted_intensity_sum: float, weighted_intensity_squared_sum: float, C: float
) -> float:
    """Calculate the Non-Local Standard Deviation."""
    if C > 0:
        mean = weighted_intensity_sum / C
        variance = (weighted_intensity_squared_sum / C) - (mean**2)
        return np.sqrt(max(0, variance))
    return 0


def calculate_nlsc(nlstd: float, nlm: float) -> float:
    """Calculate the Non-Local Speckle Contrast."""
    return nlstd / nlm if nlm > 0 else 0


# --- NLM Core Processing Functions ---


def calculate_c_xy(
    image: np.ndarray,
    x: int,
    y: int,
    patch_size: int,
    search_window_size: int,
    h: float,
    use_full_image: bool,
) -> Tuple[float, float, float, np.ndarray]:
    height, width = image.shape
    half_patch = patch_size // 2
    half_search = search_window_size // 2
    similarity_map = np.zeros_like(image)

    P_xy = extract_patch(image, x, y, patch_size)
    weighted_intensity_sum_xy = 0.0
    weighted_intensity_squared_sum_xy = 0.0
    C_xy = 0.0

    # Determine the range of pixels to process
    if use_full_image:
        x_range = range(half_patch, height - half_patch)
        y_range = range(half_patch, width - half_patch)
    else:
        x_range = range(
            max(half_patch, x - half_search),
            min(height - half_patch, x + half_search + 1),
        )
        y_range = range(
            max(half_patch, y - half_search),
            min(width - half_patch, y + half_search + 1),
        )

    for i in x_range:
        for j in y_range:
            if i == x and j == y:
                continue

            # Ensure we can extract a valid patch for comparison
            if (
                i - half_patch < 0
                or i + half_patch >= height
                or j - half_patch < 0
                or j + half_patch >= width
            ):
                continue

            P_ij = extract_patch(image, i, j, patch_size)
            P_diff_squared_xy_ij = calculate_patch_distance(P_xy, P_ij)
            weight_xy_ij = calculate_weight(P_diff_squared_xy_ij, h)
            similarity_map[i, j] = weight_xy_ij
            neighbor_pixel = image[i, j]
            weighted_intensity_sum_xy += weight_xy_ij * neighbor_pixel
            weighted_intensity_squared_sum_xy += weight_xy_ij * (neighbor_pixel**2)
            C_xy += weight_xy_ij

    return (
        C_xy,
        weighted_intensity_sum_xy,
        weighted_intensity_squared_sum_xy,
        similarity_map,
    )


def process_nlm_pixel(args):
    (
        pixel,
        image,
        patch_size,
        search_window_size,
        h,
        use_full_image,
        processing_origin,
        height,
        width,
        valid_width,
    ) = args
    x, y = divmod(pixel, valid_width)
    x += processing_origin[1]
    y += processing_origin[0]

    half_patch = patch_size // 2

    # Ensure we can extract a valid patch
    if (
        x - half_patch < 0
        or x + half_patch >= height
        or y - half_patch < 0
        or y + half_patch >= width
    ):
        return x, y, image[x, y], 0, 0, 0, np.zeros_like(image)

    (
        C_xy,
        weighted_intensity_sum_xy,
        weighted_intensity_squared_sum_xy,
        similarity_map,
    ) = calculate_c_xy(image, x, y, patch_size, search_window_size, h, use_full_image)

    if C_xy > 0:
        NLM_xy = calculate_nlm(weighted_intensity_sum_xy, C_xy, image[x, y])
        NLstd_xy = calculate_nlstd(
            weighted_intensity_sum_xy, weighted_intensity_squared_sum_xy, C_xy
        )
        NLSC_xy = calculate_nlsc(NLstd_xy, NLM_xy)
    else:
        NLM_xy = image[x, y]
        NLstd_xy = 0
        NLSC_xy = 0

    return x, y, NLM_xy, C_xy, NLstd_xy, NLSC_xy, similarity_map



