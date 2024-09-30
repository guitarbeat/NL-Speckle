"""
This module provides the implementation of the Non-Local Means (NLM) denoising algorithm 
and Speckle Contrast calculation.
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any, List, Callable, Union, get_origin, get_args
from multiprocessing import Pool, cpu_count
import time
from concurrent.futures import TimeoutError
from session_state import get_image_array, get_technique_params, handle_processing_error
from inspect import signature
import streamlit as st
import functools
import traceback
import logging

logging.basicConfig(level=logging.ERROR)

def ensure_within_bounds(y: int, x: int, height: int, width: int, half_kernel: int) -> Tuple[int, int]:
    """Ensure that the given coordinates are within the image bounds."""
    return (
        max(half_kernel, min(y, height - half_kernel - 1)),
        max(half_kernel, min(x, width - half_kernel - 1))
    )

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

def crop_to_min_size(P_xy: np.ndarray, P_ij: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Crop two arrays to the minimum size of both."""
    min_height = min(P_xy.shape[0], P_ij.shape[0])
    min_width = min(P_xy.shape[1], P_ij.shape[1])
    return P_xy[:min_height, :min_width], P_ij[:min_height, :min_width]

def enhanced_error_handling(func):
    """Decorator for enhanced error handling and reporting."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            error_msg = f"Error in {func.__name__}: {str(e)}"
            stack_trace = traceback.format_exc()
            logging.error(f"{error_msg}\n{stack_trace}")
            handle_processing_error(error_msg)
            st.error(error_msg)
            return None
    return wrapper

@enhanced_error_handling
def process_technique(technique: str) -> Optional[Dict[str, Any]]:
    try:
        logging.info(f"Starting processing for technique: {technique}")
        image = get_image_array()
        if image is None or image.size == 0:
            raise ValueError(f"No image data found in session state for {technique}")
        
        logging.info(f"Image shape for {technique}: {image.shape}")
        image = image.astype(np.float32)
        params = get_technique_params(technique)
        
        if params is None:
            raise ValueError(f"No parameters found for technique: {technique}")
        
        logging.info(f"Parameters for {technique}: {params}")
        result = apply_processing_to_image(image, technique, params)
        if result is None:
            raise ValueError(f"Processing failed for {technique}")
        
        logging.info(f"Processing completed successfully for {technique}")
        return result

    except Exception as e:
        error_msg = f"Error in process_technique for {technique}: {str(e)}"
        logging.error(error_msg, exc_info=True)
        handle_processing_error(error_msg)
        st.error(error_msg)
    
    return None

@enhanced_error_handling
def apply_processing_to_image(
    image: np.ndarray, 
    technique: str,
    params: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    try:
        logging.info(f"Starting apply_processing_to_image for {technique}")
        height, width = image.shape
        result_images = [np.zeros_like(image) for _ in range(5 if technique == 'nlm' else 3)]

        pixels = determine_pixels_to_process((height, width), params['kernel_size'], params['pixel_count'], (params['kernel_size'] // 2, params['kernel_size'] // 2))
        logging.info(f"Number of pixels to process for {technique}: {len(pixels)}")
        
        processing_function = process_nlm_pixel if technique == 'nlm' else process_speckle_pixel
        
        args_list = []
        for y, x in pixels:
            if technique == 'nlm':
                args = (processing_function, y, x, image, params['kernel_size'], params['search_window_size'], 
                        params['filter_strength'], params['use_full_image'], height, width)
            else:  # speckle
                args = (processing_function, y, x, image, params['kernel_size'], 
                        (params['kernel_size'] // 2, params['kernel_size'] // 2), 
                        height, width, width - params['kernel_size'] + 1)
            args_list.append(args)

        chunk_size = max(1, len(pixels) // (cpu_count() * 4))
        num_processes = min(cpu_count(), len(pixels) // chunk_size)

        start_time = time.time()

        total_pixels = len(pixels)
        progress_bar = st.progress(0)
        
        with Pool(processes=num_processes) as pool:
            try:
                for i, result in enumerate(pool.imap_unordered(process_pixel_wrapper, args_list, chunksize=chunk_size)):
                    if result is None:
                        raise ValueError(f"Processing failed for pixel {i}")
                    y, x, *values = result
                    for j, value in enumerate(values):
                        result_images[j][y, x] = value
                    progress = (i + 1) / total_pixels
                    progress_bar.progress(progress)
                    if time.time() - start_time > 300:  # 5 minutes timeout
                        raise TimeoutError("Processing took too long")
            except TimeoutError:
                handle_processing_error("Processing timed out. Please try with a smaller image or fewer pixels.")
                return None
            except Exception as e:
                handle_processing_error(f"An error occurred during processing: {str(e)}")
                return None
            finally:
                progress_bar.empty()
            
        processing_end = calculate_processing_end(params['pixel_count'], width - params['kernel_size'] + 1, params['kernel_size'] // 2)
        common_results = (processing_end, params['kernel_size'], params['pixel_count'], (height, width))
        
        logging.info(f"Processing completed for {technique}")
        if technique == 'nlm':
            return format_nlm_result(tuple(result_images), common_results, params['search_window_size'], params['filter_strength'])
        else:
            return format_speckle_result(tuple(result_images), common_results)

    except Exception as e:
        error_msg = f"Error in apply_processing_to_image for {technique}: {str(e)}"
        logging.error(error_msg, exc_info=True)
        handle_processing_error(error_msg)
        st.error(error_msg)
        return None

def process_pixel_wrapper(args):
    try:
        processing_function, *func_args = args
        return processing_function(*func_args)
    except Exception as e:
        logging.error(f"Error in process_pixel_wrapper: {str(e)}", exc_info=True)
        return None

def check_function_arguments(func: Callable, args: Tuple, technique: str) -> None:
    sig = signature(func)
    expected_params = list(sig.parameters.keys())
    actual_params = len(args)

    if len(expected_params) != actual_params:
        raise ValueError(
            f"Argument mismatch for {func.__name__} ({technique} technique):\n"
            f"Expected {len(expected_params)} arguments: {', '.join(expected_params)}\n"
            f"Received {actual_params} arguments: {', '.join(map(str, args))}\n"
            f"Mismatch details:\n"
            f"{'  '.join(f'{expected} (expected) vs {actual} (received)' for expected, actual in zip(expected_params, args))}"
        )

    for i, (param, arg) in enumerate(zip(expected_params, args)):
        expected_type = sig.parameters[param].annotation
        if expected_type is sig.empty:
            continue  # Skip unannotated parameters
        
        if not type_check(arg, expected_type):
            raise TypeError(
                f"Type mismatch in {func.__name__} ({technique} technique) for argument {i + 1}:\n"
                f"Parameter '{param}' expected type {expected_type}, "
                f"but received {type(arg).__name__}"
            )

def type_check(value: Any, expected_type: Any) -> bool:
    if expected_type is Any:
        return True
    if get_origin(expected_type) is Union:
        return any(type_check(value, t) for t in get_args(expected_type))
    if get_origin(expected_type) in (list, List):
        return isinstance(value, list) and all(type_check(v, get_args(expected_type)[0]) for v in value)
    if get_origin(expected_type) in (tuple, Tuple):
        return isinstance(value, tuple) and len(value) == len(get_args(expected_type)) and all(type_check(v, t) for v, t in zip(value, get_args(expected_type)))
    if expected_type is int:
        return isinstance(value, (int, np.integer))
    if expected_type is float:
        return isinstance(value, (float, np.floating))
    if expected_type is bool:
        return isinstance(value, (bool, np.bool_))
    if expected_type is np.ndarray:
        return isinstance(value, np.ndarray)
    return isinstance(value, expected_type)

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

def process_speckle_pixel(y: int, x: int, image: np.ndarray, kernel_size: int, 
                          processing_origin: Tuple[int, int], height: int, width: int, valid_width: int) -> Tuple:
    half_kernel = kernel_size // 2
    
    y, x = ensure_within_bounds(y, x, height, width, half_kernel)
    
    local_window = get_window(image, y, x, half_kernel, height, width)
    local_mean = np.nanmean(local_window)
    local_std = np.nanstd(local_window)
    sc = local_std / local_mean if local_mean != 0 else 0
    return y, x, local_mean, local_std, sc

def determine_pixels_to_process(
    image_shape: Tuple[int, int],
    kernel_size: int, 
    pixels_to_process: int,
    processing_origin: Tuple[int, int] = (0, 0)
) -> List[Tuple[int, int]]:
    height, width = image_shape
    valid_height, valid_width = height - kernel_size + 1, width - kernel_size + 1
    pixels_to_process = min(pixels_to_process, valid_height * valid_width)
    
    return [(pixel % valid_width, pixel // valid_width)
            for pixel in range(pixels_to_process)]

def calculate_processing_end(pixel_count: int, valid_width: int, half_kernel: int) -> Tuple[int, int]:
    y, x = divmod(pixel_count - 1, valid_width)
    return (x + half_kernel, y + half_kernel)

def format_nlm_result(results: Tuple, common_results: Tuple, search_window_size: int, filter_strength: float) -> Dict[str, Any]:
    nlm_image, normalization_factors, nl_std_image, nl_speckle_image, last_similarity_map = results
    processing_end, kernel_size, pixel_count, image_dimensions = common_results
    
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

def format_speckle_result(results: Tuple, common_results: Tuple) -> Dict[str, Any]:
    mean_filter, std_dev_filter, speckle_contrast_filter = results
    processing_end, kernel_size, pixel_count, image_dimensions = common_results
    
    return {
        'mean_filter': mean_filter.reshape(image_dimensions),
        'std_dev_filter': std_dev_filter.reshape(image_dimensions),
        'speckle_contrast_filter': speckle_contrast_filter.reshape(image_dimensions),
        'processing_end_coord': processing_end,
        'kernel_size': kernel_size,
        'pixels_processed': pixel_count, 
        'image_dimensions': image_dimensions,
        'filter_data': {
            "Mean Filter": mean_filter.reshape(image_dimensions),
            "Std Dev Filter": std_dev_filter.reshape(image_dimensions),
            "Speckle Contrast": speckle_contrast_filter.reshape(image_dimensions),  
        }
    }