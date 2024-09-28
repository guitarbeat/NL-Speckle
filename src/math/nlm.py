"""
This module provides the implementation of Non-Local Means (NLM) denoising
algorithm functions.
"""

from dataclasses import dataclass
import numpy as np
from typing import List, Tuple, Optional, Callable, Type, Dict
from functools import lru_cache
from src.classes import BaseResult, ResultCombinationError
import streamlit as st
from multiprocessing import Pool, cpu_count
from src.utils import validate_input, calculate_processing_end


@dataclass
class NLMResult(BaseResult):
    nonlocal_means: np.ndarray
    normalization_factors: np.ndarray
    nonlocal_std: np.ndarray
    nonlocal_speckle: np.ndarray
    search_window_size: int
    filter_strength: float
    last_similarity_map: np.ndarray

    @staticmethod
    def get_filter_options() -> List[str]:
        return [
            "Non-Local Means",
            "Normalization Factors",
            "Last Similarity Map",
            "Non-Local Standard Deviation",
            "Non-Local Speckle",
        ]

    @property
    def filter_data(self) -> Dict[str, np.ndarray]:
        return {
            "Non-Local Means": self.nonlocal_means,
            "Normalization Factors": self.normalization_factors,
            "Last Similarity Map": self.last_similarity_map,
            "Non-Local Standard Deviation": self.nonlocal_std,
            "Non-Local Speckle": self.nonlocal_speckle,
        }

    @classmethod
    def combine(class_: Type["NLMResult"], results: List["NLMResult"]) -> "NLMResult":
        if not results:
            return class_.empty_result()

        try:
            combined_arrays: Dict[str, np.ndarray] = {
                attr: np.maximum.reduce([getattr(r, attr) for r in results])
                for attr in [
                    "nonlocal_means",
                    "normalization_factors",
                    "nonlocal_std",
                    "nonlocal_speckle",
                ]
            }
        except ValueError as e:
            raise ResultCombinationError(f"Error combining NLM results: {str(e)}")

        return class_(
            **combined_arrays,
            **BaseResult.combine(results).__dict__,
            search_window_size=results[0].search_window_size,
            filter_strength=results[0].filter_strength,
            last_similarity_map=results[-1].last_similarity_map,
        )

    @classmethod
    def empty_result(class_: Type["NLMResult"]) -> "NLMResult":
        return class_(
            nonlocal_means=np.array([]),
            normalization_factors=np.array([]),
            nonlocal_std=np.array([]),
            nonlocal_speckle=np.array([]),
            processing_end_coord=(0, 0),
            kernel_size=0,
            pixels_processed=0,
            image_dimensions=(0, 0),
            search_window_size=0,
            filter_strength=0,
            last_similarity_map=np.array([]),
        )

    @classmethod
    def merge(
        class_: Type["NLMResult"], new_result: "NLMResult", existing_result: "NLMResult"
    ) -> "NLMResult":
        merged_arrays: Dict[str, np.ndarray] = {
            attr: np.maximum(getattr(new_result, attr), getattr(existing_result, attr))
            for attr in [
                "nonlocal_means",
                "normalization_factors",
                "nonlocal_std",
                "nonlocal_speckle",
            ]
        }

        return class_(
            **merged_arrays,
            processing_end_coord=max(
                new_result.processing_end_coord, existing_result.processing_end_coord
            ),
            kernel_size=new_result.kernel_size,
            pixels_processed=max(
                new_result.pixels_processed, existing_result.pixels_processed
            ),
            image_dimensions=new_result.image_dimensions,
            search_window_size=new_result.search_window_size,
            filter_strength=new_result.filter_strength,
            last_similarity_map=new_result.last_similarity_map,
        )


def process_non_local_means(status, image: np.ndarray, kernel_size: int, pixel_count: int,
                            nlm_search_window_size: int, nlm_filter_strength: float, 
                            update_progress: Callable) -> NLMResult:
    validate_input(image, kernel_size, pixel_count, nlm_search_window_size, nlm_filter_strength)
    height, width = image.shape
    half_kernel = kernel_size // 2
    _valid_height, _valid_width = height - kernel_size + 1, width - kernel_size + 1

    status.write(f"🔎 Initiating Non-Local Means denoising with {nlm_search_window_size}x{nlm_search_window_size} search window")
    status.write(f"⚖️ Applying filter strength h = {nlm_filter_strength} for weight calculations")
    try:  
        nlm_image, nl_std_image, nl_speckle_image, normalization_factors, last_similarity_map = apply_nlm_to_image(
            image, kernel_size, nlm_search_window_size, nlm_filter_strength,
            pixel_count, (half_kernel, half_kernel), update_progress)
    except Exception as e:
        raise ResultCombinationError(f"Error in apply_nlm_to_image: {str(e)}") from e

    processing_end = calculate_processing_end(width, height, kernel_size, pixel_count)
    status.write("🧮 Calculating weighted averages of similar patches for each pixel")

    return NLMResult(nonlocal_means=nlm_image, normalization_factors=normalization_factors,
                     nonlocal_std=nl_std_image, nonlocal_speckle=nl_speckle_image, 
                     processing_end_coord=processing_end, kernel_size=kernel_size,
                     pixels_processed=pixel_count, image_dimensions=(height, width),
                     search_window_size=nlm_search_window_size, filter_strength=nlm_filter_strength,
                     last_similarity_map=last_similarity_map)

def apply_nlm_to_image(
    image: np.ndarray,
    patch_size: int,
    search_window_size: int, 
    h: float,
    pixels_to_process: int,
    processing_origin: Tuple[int, int],
    progress_callback: Optional[Callable[[float], None]] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    height, width = image.shape
    valid_width = width - patch_size + 1

    NLM_image = np.zeros_like(image)
    C_xy_image = np.zeros_like(image)
    NLstd_image = np.zeros_like(image)
    NLSC_xy_image = np.zeros_like(image)

    use_full_image = st.session_state.get("use_full_image", False)
    args_list = [
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
        )
        for pixel in range(pixels_to_process)
    ]

    chunk_size = max(1, pixels_to_process // (cpu_count() * 4))
    num_processes = min(cpu_count(), pixels_to_process // chunk_size)
    update_interval = max(1, pixels_to_process // 1000)

    with Pool(processes=num_processes) as pool:
        for i, result in enumerate(pool.imap(process_nlm_pixel, args_list, chunksize=chunk_size)):
            x, y, NLM_xy, C_xy, NLstd_xy, NLSC_xy, similarity_map = result
            NLM_image[x, y] = NLM_xy
            NLstd_image[x, y] = NLstd_xy
            NLSC_xy_image[x, y] = NLSC_xy
            C_xy_image[x, y] = C_xy
            last_similarity_map = similarity_map

            if progress_callback and i % update_interval == 0:
                progress_callback((i + 1) / pixels_to_process)

    if progress_callback:
        progress_callback(1.0)

    return NLM_image, NLstd_image, NLSC_xy_image, C_xy_image, last_similarity_map


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



