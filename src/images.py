import numpy as np
import streamlit as st
from typing import Tuple, Optional, Callable
from contextlib import contextmanager
from multiprocessing import Pool, cpu_count
import logging

from src.classes import NLSpeckleResult, NLMResult, SpeckleResult, ResultCombinationError
from src.math.speckle import calculate_speckle_contrast
from src.math.nlm import process_nlm_pixel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Utility functions
def validate_input(image: np.ndarray, kernel_size: int, pixel_count: int, search_window_size: Optional[int] = None, filter_strength: Optional[float] = None):
    if not isinstance(image, np.ndarray) or image.ndim != 2:
        raise ValueError("Image must be a 2D numpy array.")
    if not isinstance(kernel_size, int) or kernel_size <= 0 or kernel_size > min(image.shape) or kernel_size % 2 == 0:
        raise ValueError("Invalid kernel_size. Must be a positive odd integer not larger than the smallest image dimension.")
    if not isinstance(pixel_count, int) or pixel_count <= 0:
        raise ValueError("pixel_count must be a positive integer.")
    if search_window_size is not None and (not isinstance(search_window_size, int) or search_window_size <= 0):
        raise ValueError("search_window_size must be a positive integer.")
    if filter_strength is not None and (not isinstance(filter_strength, (int, float)) or filter_strength <= 0):
        raise ValueError("filter_strength must be a positive number.")
    if image.dtype != np.float32:
        raise ValueError("Image must be of dtype float32.")

def calculate_processing_end(width: int, height: int, kernel_size: int, pixel_count: int) -> Tuple[int, int]:
    valid_width = width - kernel_size + 1
    if valid_width <= 0:
        raise ValueError("Invalid valid_width calculated. Check image dimensions and kernel_size.")
    end_y, end_x = divmod(pixel_count - 1, valid_width)
    half_kernel = kernel_size // 2
    return (min(end_x + half_kernel, width - 1), min(end_y + half_kernel, height - 1))

@contextmanager
def create_processing_status():
    with st.status("Processing image...", expanded=True) as status:
        progress_bar = st.progress(0)
        yield status, progress_bar

# Speckle contrast functions
def apply_speckle_contrast(
    image: np.ndarray,
    kernel_size: int,
    pixels_to_process: int,
    processing_origin: Tuple[int, int],
    progress_callback: Optional[Callable[[float], None]] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    logger.info(f"Speckle: Processing {pixels_to_process} pixels")

    height, width = image.shape
    mean_filter = np.zeros_like(image)
    std_dev_filter = np.zeros_like(image)
    sc_filter = np.zeros_like(image)
    valid_width = width - kernel_size + 1

    half_kernel = kernel_size // 2
    update_interval = max(1, pixels_to_process // 1000)

    for pixel in range(pixels_to_process):
        row = processing_origin[1] + pixel // valid_width
        col = processing_origin[0] + pixel % valid_width
        
        if row < height and col < width:
            row_start, row_end = max(0, row - half_kernel), min(height, row + half_kernel + 1)
            col_start, col_end = max(0, col - half_kernel), min(width, col + half_kernel + 1)

            local_window = image[row_start:row_end, col_start:col_end]
            local_mean = np.nanmean(local_window)
            local_std = np.nanstd(local_window)
            sc = calculate_speckle_contrast(local_std, local_mean)

            mean_filter[row, col] = local_mean
            std_dev_filter[row, col] = local_std
            sc_filter[row, col] = sc

            if progress_callback and pixel % update_interval == 0:
                progress_callback((pixel + 1) / pixels_to_process)

    if progress_callback:
        progress_callback(1.0)

    return mean_filter, std_dev_filter, sc_filter

def compute_speckle_contrast(image: np.ndarray, kernel_size: int, pixel_count: int, start_pixel: int = 0,
                             progress_callback: Optional[Callable[[float], None]] = None) -> SpeckleResult:
    validate_input(image, kernel_size, pixel_count)
    height, width = image.shape
    half_kernel = kernel_size // 2
    valid_height, valid_width = height - kernel_size + 1, width - kernel_size + 1
    pixel_count = min(pixel_count, valid_height * valid_width)
    start_y, start_x = divmod(start_pixel, valid_width)
    start_coords = (start_x + half_kernel, start_y + half_kernel)
    try:
        mean_image, std_dev_image, speckle_contrast_image = apply_speckle_contrast(
            image, kernel_size, pixel_count, start_coords,
            lambda p: progress_callback(p) if progress_callback else None)
    except Exception as e:
        raise ResultCombinationError(f"Error in apply_speckle_contrast: {str(e)}") from e
    end_pixel = start_pixel + pixel_count
    end_y, end_x = divmod(end_pixel - 1, valid_width)
    processing_end = (min(end_x + half_kernel, width - 1), min(end_y + half_kernel, height - 1))
    if progress_callback:
        progress_callback(1.0)
    return SpeckleResult(mean_filter=mean_image, std_dev_filter=std_dev_image,
                         speckle_contrast_filter=speckle_contrast_image,
                         processing_end_coord=processing_end, kernel_size=kernel_size,
                         pixels_processed=pixel_count, image_dimensions=(height, width))

# Non-local means functions
def apply_nlm_to_image(
    image: np.ndarray,
    patch_size: int,
    search_window_size: int,
    h: float,
    pixels_to_process: int,
    processing_origin: Tuple[int, int],
    progress_callback: Optional[Callable[[float], None]] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    logger.info(f"NLM: Processing {pixels_to_process} pixels")

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
        try:
            for i, result in enumerate(pool.imap(process_nlm_pixel, args_list, chunksize=chunk_size)):
                x, y, NLM_xy, C_xy, NLstd_xy, NLSC_xy, similarity_map = result
                NLM_image[x, y] = NLM_xy
                NLstd_image[x, y] = NLstd_xy
                NLSC_xy_image[x, y] = NLSC_xy
                C_xy_image[x, y] = C_xy
                last_similarity_map = similarity_map

                if progress_callback and i % update_interval == 0:
                    progress_callback((i + 1) / pixels_to_process)

        except Exception as e:
            logger.error(f"Error in apply_nlm_to_image: {str(e)}")
            raise

    if progress_callback:
        progress_callback(1.0)

    return NLM_image, NLstd_image, NLSC_xy_image, C_xy_image, last_similarity_map

def compute_non_local_means(image: np.ndarray, kernel_size: int, pixel_count: int,
                            search_window_size: int = 21, filter_strength: float = 10.0,
                            start_pixel: int = 0,
                            progress_callback: Optional[Callable[[float], None]] = None) -> NLMResult:
    validate_input(image, kernel_size, pixel_count, search_window_size, filter_strength)
    height, width = image.shape
    half_kernel = kernel_size // 2
    valid_height, valid_width = height - kernel_size + 1, width - kernel_size + 1
    total_valid_pixels = valid_height * valid_width
    end_pixel = min(start_pixel + pixel_count, total_valid_pixels)
    pixel_count = end_pixel - start_pixel
    start_y, start_x = divmod(start_pixel, valid_width)
    start_coords = (start_x + half_kernel, start_y + half_kernel)
    try:
        nlm_image, nl_std_image, nl_speckle_image, normalization_factors, last_similarity_map = apply_nlm_to_image(
            image, kernel_size, search_window_size, filter_strength,
            pixel_count, start_coords, lambda p: progress_callback(p) if progress_callback else None)
    except Exception as e:
        raise ResultCombinationError(f"Error in apply_nlm_to_image: {str(e)}") from e
    end_y, end_x = divmod(end_pixel - 1, valid_width)
    processing_end = (min(end_x + half_kernel, width - 1), min(end_y + half_kernel, height - 1))
    if progress_callback:
        progress_callback(1.0)
    return NLMResult(nonlocal_means=nlm_image, normalization_factors=normalization_factors,
                     nonlocal_std=nl_std_image, nonlocal_speckle=nl_speckle_image,
                     processing_end_coord=processing_end, kernel_size=kernel_size,
                     pixels_processed=pixel_count, image_dimensions=(height, width),
                     search_window_size=search_window_size, filter_strength=filter_strength,
                     last_similarity_map=last_similarity_map)

# Main processing function
def initialize_processing(image: np.ndarray, kernel_size: int, pixel_count: int, 
                          nlm_search_window_size: int, nlm_filter_strength: float) -> Tuple[np.ndarray, int, int, int]:
    validate_input(image, kernel_size, pixel_count, nlm_search_window_size, nlm_filter_strength)
    height, width = image.shape
    valid_pixels = (height - kernel_size + 1) * (width - kernel_size + 1)
    pixel_count = min(pixel_count, valid_pixels)
    return image, kernel_size, pixel_count, valid_pixels

def process_speckle_contrast(status, image: np.ndarray, kernel_size: int, pixel_count: int, 
                             update_progress: Callable) -> SpeckleResult:
    status.write("📊 Calculating local statistics (mean, standard deviation) for each pixel neighborhood")
    status.write(f"🔍 Using a {kernel_size}x{kernel_size} kernel to analyze {pixel_count} pixels")
    speckle_result = compute_speckle_contrast(image, kernel_size, pixel_count, 0, update_progress)
    status.write("📈 Computing speckle contrast (σ/μ) from local statistics")
    return speckle_result

def process_non_local_means(status, image: np.ndarray, kernel_size: int, pixel_count: int, 
                            nlm_search_window_size: int, nlm_filter_strength: float, 
                            update_progress: Callable) -> NLMResult:
    status.write(f"🔎 Initiating Non-Local Means denoising with {nlm_search_window_size}x{nlm_search_window_size} search window")
    status.write(f"⚖️ Applying filter strength h = {nlm_filter_strength} for weight calculations")
    nlm_result = compute_non_local_means(image, kernel_size, pixel_count, nlm_search_window_size,
                                         nlm_filter_strength, 0, update_progress)
    status.write("🧮 Calculating weighted averages of similar patches for each pixel")
    return nlm_result

def combine_results(status, speckle_result: SpeckleResult, nlm_result: NLMResult, 
                    kernel_size: int, pixel_count: int, image_dimensions: Tuple[int, int], 
                    nlm_search_window_size: int, nlm_filter_strength: float) -> NLSpeckleResult:
    status.write("🔗 Merging Speckle Contrast and Non-Local Means results")
    final_result = NLSpeckleResult(
        nlm_result=nlm_result,
        speckle_result=speckle_result,
        additional_images={},
        processing_end_coord=calculate_processing_end(image_dimensions[1], image_dimensions[0], kernel_size, pixel_count),
        kernel_size=kernel_size,
        pixels_processed=pixel_count,
        image_dimensions=image_dimensions,
        nlm_search_window_size=nlm_search_window_size,
        nlm_filter_strength=nlm_filter_strength
    )
    status.write("🏁 Analysis complete: NL-Speckle result generated")
    return final_result

def process_image() -> Optional[NLSpeckleResult]:
    if "image_array" not in st.session_state or st.session_state.image_array is None:
        st.warning("No image has been uploaded. Please upload an image before processing.")
        return None

    try:
        image, kernel_size, pixel_count, valid_pixels = initialize_processing(
            st.session_state.image_array.astype(np.float32),
            st.session_state.kernel_size,
            st.session_state.exact_pixel_count,
            st.session_state.search_window_size,
            st.session_state.filter_strength
        )

        with create_processing_status() as (status, progress_bar):
            def update_progress(task: str, progress: float):
                status.update(label=f"Processing: {task} - {progress:.1%} complete")
                progress_bar.progress(0.5 * progress if task == "Speckle" else 0.5 + 0.5 * progress)

            speckle_result = process_speckle_contrast(status, image, kernel_size, pixel_count, 
                                                      lambda p: update_progress("Speckle", p))
            
            nlm_result = process_non_local_means(status, image, kernel_size, pixel_count, 
                                                 st.session_state.search_window_size, 
                                                 st.session_state.filter_strength, 
                                                 lambda p: update_progress("NLM", p))

            if speckle_result is None or nlm_result is None:
                raise ValueError("Processing failed to produce a result")

            final_result = combine_results(status, speckle_result, nlm_result, kernel_size, pixel_count, 
                                           image.shape, st.session_state.search_window_size, 
                                           st.session_state.filter_strength)

            status.update(label="Processing complete!", state="complete")
            progress_bar.progress(1.0)

        st.session_state.nl_speckle_result = final_result
        st.session_state.speckle_result = speckle_result
        st.session_state.nlm_result = nlm_result

        return final_result

    except Exception as e:
        error_message = f"Error in process_image: {type(e).__name__}: {str(e)}"
        image_info = f"Image shape: {st.session_state.image_array.shape}, Image size: {st.session_state.image_array.size}, Image dtype: {st.session_state.image_array.dtype}"
        logger.error(f"{error_message}\n{image_info}")
        st.error(f"{error_message}\n{image_info}")
        return None