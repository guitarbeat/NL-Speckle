import functools
import logging
import traceback
from concurrent.futures import TimeoutError
from multiprocessing import cpu_count
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar
import time
import numpy as np
import streamlit as st
import concurrent.futures
from src.nlm_processing import format_nlm_result, process_nlm_pixel
from src.session_state import (get_image_array, get_technique_params,
                               handle_processing_error, set_technique_result,
                               set_last_processed_pixel)  # Add this import
from src.speckle_processing import format_speckle_result, process_speckle_pixel

logging.basicConfig(level=logging.ERROR)

T = TypeVar('T')
ProcessingFunction = Callable[[int, int, np.ndarray, int, Tuple[int, int], float, bool, int, int], Tuple[int, int, float, float, float, float, float]]

TECHNIQUE_FUNCTIONS = {
    'nlm': process_nlm_pixel,
    'speckle': process_speckle_pixel
}

def get_processing_function(technique: str) -> ProcessingFunction:
    if technique not in TECHNIQUE_FUNCTIONS:
        raise ValueError(f"Unknown technique: {technique}")
    return TECHNIQUE_FUNCTIONS[technique]

@functools.wraps
def enhanced_error_handling(func: Callable[..., T]) -> Callable[..., Optional[T]]:
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Optional[T]:
        try:
            return func(*args, **kwargs)
        except Exception as e:
            error_msg = f"Error in {func.__name__}: {str(e)}"
            logging.error(f"{error_msg}\n{traceback.format_exc()}")
            handle_processing_error(error_msg)
            return None
    return wrapper

@enhanced_error_handling
def process_technique(technique: str) -> Optional[Dict[str, Any]]:
    image = get_image_array()
    if image is None or image.size == 0:
        logging.error(f"No image data found in session state for {technique}")
        st.error("No image data found. Please upload an image before processing.")
        return None
    
    logging.info(f"Image shape: {image.shape}, dtype: {image.dtype}")
    image = image.astype(np.float32)
    params = get_technique_params(technique)
    
    if params is None:
        logging.error(f"No parameters found for technique: {technique}")
        st.error(f"No parameters found for {technique}. Please check your configuration.")
        return None
    
    st.write(f"Processing {technique} with params: {params}")
    result = apply_processing_to_image(image, technique, params)
    if result is None:
        logging.error(f"Processing failed for {technique}")
        st.error(f"Processing failed for {technique}. Please try again or check the logs for more information.")
        return None
    
    st.write(f"Processed {technique}. Result keys: {result.keys()}")
    set_technique_result(technique, result)
    return result

@enhanced_error_handling
def apply_processing_to_image(image: np.ndarray, technique: str, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    st.write("Debug: utils.py - apply_processing_to_image")
    st.write(f"Input: technique={technique}, params={params}")
    
    height, width = image.shape
    result_images = [np.zeros_like(image) for _ in range(5 if technique == 'nlm' else 3)]
    pixels = determine_pixels_to_process((height, width), params['kernel_size'], params['pixel_count'])
    
    st.write(f"Debug: Number of pixels to process: {len(pixels)}")
    
    processing_function = get_processing_function(technique)
    args_list = create_args_list(technique, pixels, image, params, height, width, processing_function)
    
    num_processes = min(cpu_count(), 4)
    progress_bar = st.progress(0)
    status = st.empty()

    try:
        with st.spinner(f'Processing {technique.upper()}...'):
            with concurrent.futures.ProcessPoolExecutor(max_workers=num_processes) as executor:
                futures = [executor.submit(process_pixel_wrapper, args) for args in args_list]
                process_futures(futures, result_images, progress_bar, status)
    finally:
        if pixels:
            set_last_processed_pixel(pixels[-1][1], pixels[-1][0])

    processing_end = calculate_processing_end(len(pixels), width - params['kernel_size'] + 1, params['kernel_size'] // 2)
    common_results = (processing_end, params['kernel_size'], len(pixels), (height, width))
    
    st.write(f"Debug: Processing complete. processing_end={processing_end}, pixels_processed={len(pixels)}")
    
    return format_result(technique, tuple(result_images), common_results, params)

def determine_pixels_to_process(image_shape: Tuple[int, int], kernel_size: int, pixel_count: int) -> List[Tuple[int, int]]:
    st.write("Debug: utils.py - determine_pixels_to_process")
    st.write(f"Input: image_shape={image_shape}, kernel_size={kernel_size}, pixel_count={pixel_count}")
    
    height, width = image_shape
    half_kernel = kernel_size // 2
    valid_height = height - kernel_size + 1
    valid_width = width - kernel_size + 1
    total_pixels = valid_height * valid_width

    st.write(f"Debug: Valid area - height={valid_height}, width={valid_width}, total_pixels={total_pixels}")

    if pixel_count <= 0 or pixel_count >= total_pixels:
        pixels = [(y + half_kernel, x + half_kernel) for y in range(valid_height) for x in range(valid_width)]
    else:
        step = max(1, int(np.sqrt(total_pixels / pixel_count)))
        pixels = [(y + half_kernel, x + half_kernel) 
                  for y in range(0, valid_height, step) 
                  for x in range(0, valid_width, step)][:pixel_count]
    
    st.write(f"Debug: Number of pixels to process: {len(pixels)}")
    st.write(f"Debug: First few pixels: {pixels[:5]}")
    return pixels

def create_args_list(technique: str, pixels: List[Tuple[int, int]], image: np.ndarray, params: Dict[str, Any], height: int, width: int, processing_function: ProcessingFunction) -> List[Tuple[Any, ...]]:
    if technique == 'nlm':
        return [(processing_function, y, x, image, params['kernel_size'], params['search_window_size'], 
                 params['filter_strength'], params['use_full_image'], height, width) for y, x in pixels]
    else:  # speckle
        return [(processing_function, y, x, image, params['kernel_size'], 
                 (params['kernel_size'] // 2, params['kernel_size'] // 2), 
                 height, width, width - params['kernel_size'] + 1) for y, x in pixels]

def process_pixel_wrapper(args: Tuple[ProcessingFunction, ...]) -> Optional[Tuple[int, int, float, float, float, float, float]]:
    try:
        processing_function, *func_args = args
        return processing_function(*func_args)
    except Exception as e:
        logging.error(f"Error processing pixel: {str(e)}")
        return None

def process_futures(futures, result_images, progress_bar, status):
    st.write("Debug: utils.py - process_futures")
    total_pixels = len(futures)
    start_time = time.time()
    last_processed = (0, 0)
    for i, future in enumerate(concurrent.futures.as_completed(futures)):
        try:
            result = future.result(timeout=60)
            if result is None:
                raise ValueError(f"Processing failed for pixel {i}")
            y, x, *values = result
            for j, value in enumerate(values):
                result_images[j][y, x] = value
            last_processed = (y, x)
        except concurrent.futures.TimeoutError:
            logging.error(f"Timeout occurred while processing pixel {i}")
            continue

        # Update progress bar and status
        progress = (i + 1) / total_pixels
        progress_bar.progress(progress)
        elapsed_time = time.time() - start_time
        estimated_total_time = elapsed_time / progress if progress > 0 else 0
        remaining_time = estimated_total_time - elapsed_time
        status.text(f"Processed {i+1}/{total_pixels} pixels. Estimated time remaining: {remaining_time:.2f} seconds")

    set_last_processed_pixel(last_processed[1], last_processed[0])
    st.write(f"Debug: Last processed pixel: {last_processed}")
    progress_bar.progress(1.0)
    status.text("Processing complete!")

def calculate_processing_end(pixel_count: int, max_pixels: int, half_kernel: int) -> Tuple[int, int]:
    processed_pixels = min(pixel_count, max_pixels)
    return (processed_pixels // max_pixels + half_kernel, processed_pixels % max_pixels + half_kernel)

def format_result(technique: str, result_images: Tuple[np.ndarray, ...], common_results: Tuple[Any, ...], params: Dict[str, Any]) -> Dict[str, Any]:
    if technique == 'nlm':
        return format_nlm_result(result_images, *common_results, params['search_window_size'], params['filter_strength'])
    else:  # speckle
        return format_speckle_result(result_images, *common_results)