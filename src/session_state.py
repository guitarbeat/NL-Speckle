"""
This module manages all session state operations for the Streamlit application.
"""

import streamlit as st
from typing import Dict, Any, List, Union, Tuple, Optional, TypeVar
import numpy as np
import logging
import traceback

# Constants
SPECKLE_CONTRAST: str = "Speckle Contrast"
ORIGINAL_IMAGE: str = "Original Image"
NON_LOCAL_MEANS: str = "Non-Local Means"
DEFAULT_SPECKLE_VIEW: List[str] = [ORIGINAL_IMAGE, SPECKLE_CONTRAST]
DEFAULT_NLM_VIEW: List[str] = [ORIGINAL_IMAGE, NON_LOCAL_MEANS]
DEFAULT_KERNEL_SIZE: int = 3
DEFAULT_FILTER_STRENGTH: float = 10.0
DEFAULT_SEARCH_WINDOW_SIZE: int = 50

# Type variables for better type hinting
T = TypeVar('T')

# Default values
DEFAULT_VALUES: Dict[str, Any] = {
    "image": None,
    "image_array": None,
    "kernel_size": DEFAULT_KERNEL_SIZE,
    "show_per_pixel": False,
    "color_map": "gray",
    "normalization": {
        "option": "None",
        "percentile_low": 2,
        "percentile_high": 98,
    },
    "gaussian_noise": {
        "enabled": False,
        "mean": 0.1,
        "std_dev": 0.1,
    },
    "use_sat": False,
    "nlm_options": {
        "use_whole_image": True,
        "filter_strength": DEFAULT_FILTER_STRENGTH,
        "search_window_size": DEFAULT_SEARCH_WINDOW_SIZE,
    },
    "pixel_processing": {
        "percentage": 100,
        "exact_count": 0,
    },
    "filters": {
        "nlm": DEFAULT_NLM_VIEW,
        "speckle": DEFAULT_SPECKLE_VIEW,
    },
    "techniques": ["speckle", "nlm"],
    "tabs": None,
    "viz_config": None,
    "filter_selections": {
        "speckle": DEFAULT_SPECKLE_VIEW,
        "nlm": DEFAULT_NLM_VIEW,
    },
    "speckle_result": None,
    "nlm_result": None,
    "pixels_to_process": 0,
    "desired_exact_count": 0,
    "processed_image_np": None,
    "image_file": None,
    "last_processed_pixel": (0, 0),
}

# Session state management
def initialize_session_state() -> None:
    try:
        st.session_state.update({k: v for k, v in DEFAULT_VALUES.items() if k not in st.session_state})
    except Exception as e:
        handle_processing_error(f"Error initializing session state: {str(e)}")

def get_session_state(key: str, default: T = None) -> T:
    return st.session_state.get(key, default)

def set_session_state(key: str, value: Any) -> None:
    try:
        if key not in st.session_state:
            st.session_state[key] = value
        elif isinstance(value, np.ndarray):
            if not np.array_equal(st.session_state[key], value):
                st.session_state[key] = value
        else:
            if st.session_state[key] != value:
                st.session_state[key] = value
    except Exception as e:
        handle_processing_error(f"Error setting value for key '{key}': {str(e)}")

def update_nested_session_state(keys: List[str], value: Any) -> None:
    current = st.session_state
    for key in keys[:-1]:
        if key not in current:
            current[key] = {}
        current = current[key]
    current[keys[-1]] = value

# Simplified getter and setter functions
def get_value(key: str, default: T = None) -> T:
    try:
        return st.session_state.get(key, default)
    except Exception as e:
        handle_processing_error(f"Error getting value for key '{key}': {str(e)}")
        return default

def set_value(key: str, value: Any) -> None:
    try:
        if key not in st.session_state:
            st.session_state[key] = value
        elif isinstance(value, np.ndarray):
            if not np.array_equal(st.session_state[key], value):
                st.session_state[key] = value
        else:
            if st.session_state[key] != value:
                st.session_state[key] = value
    except Exception as e:
        handle_processing_error(f"Error setting value for key '{key}': {str(e)}")

# Image processing functions
def update_pixel_processing_values() -> None:
    try:
        if st.session_state.image:
            total_pixels: int = calculate_total_pixels()
            st.session_state.desired_exact_count = total_pixels
            st.session_state.pixels_to_process = total_pixels
    except Exception as e:
        handle_processing_error(f"Error updating pixel processing values: {str(e)}")

def calculate_total_pixels() -> int:
    try:
        if st.session_state.image:
            width, height = get_image_dimensions()
            kernel = kernel_size()  # Updated here
            return max((width - kernel + 1) * (height - kernel + 1), 0)
        return 0
    except Exception as e:
        handle_processing_error(f"Error calculating total pixels: {str(e)}")
        return 0

def get_image_dimensions() -> Tuple[int, int]:
    try:
        image_array: Optional[np.ndarray] = get_value('image_array')
        return image_array.shape[:2] if isinstance(image_array, np.ndarray) else (0, 0)
    except Exception as e:
        handle_processing_error(f"Error getting image dimensions: {str(e)}")
        return (0, 0)

def get_image_array() -> np.ndarray:
    try:
        return get_value('image_array', np.array([]))
    except Exception as e:
        handle_processing_error(f"Error getting image array: {str(e)}")
        return np.array([])

def set_processed_image_np(image: np.ndarray) -> None:
    set_value('processed_image_np', image)

def get_processed_image_np() -> np.ndarray:
    return get_value('processed_image_np', np.array([]))

# Filter and technique functions
def get_filter_options(technique: str) -> List[str]:
    options = {
        'speckle': ["Original Image", "Mean Filter", "Std Dev Filter", "Speckle Contrast"],
        'nlm': ["Original Image", "Non-Local Means", "Normalization Factors", "Last Similarity Map", "Non-Local Standard Deviation", "Non-Local Speckle"]
    }
    return options.get(technique, ["Original Image"])

def update_filter_selection(technique: str, selected_filters: List[str]) -> None:
    current_selections = get_value('filter_selections', {})
    current_selections[technique] = selected_filters
    set_value('filter_selections', current_selections)
    # Update the Streamlit session state as well
    st.session_state[f'{technique}_filters'] = selected_filters

def get_filter_selection(technique: str) -> List[str]:
    return get_value('filter_selections', {}).get(technique, DEFAULT_SPECKLE_VIEW if technique == 'speckle' else DEFAULT_NLM_VIEW)

def set_technique_result(technique: str, result: Any) -> None:
    set_value(f"{technique}_result", result)

def get_technique_result(technique: str) -> Any:
    try:
        return get_value(f"{technique}_result")
    except Exception as e:
        handle_processing_error(f"Error getting technique result for '{technique}': {str(e)}")
        return None

def get_technique_params(technique: str) -> Dict[str, Any]:
    params = {
        'kernel_size': kernel_size(),  # Updated here
        'pixel_count': get_value('pixels_to_process', calculate_total_pixels())
    }
    if technique == 'nlm':
        nlm_options = get_nlm_options()
        params.update({
            'search_window_size': nlm_options['search_window_size'],
            'filter_strength': nlm_options['filter_strength'],
            'use_full_image': nlm_options['use_whole_image']
        })
    return params

def update_technique_params(params: Dict[str, Any]) -> None:
    for key, value in params.items():
        set_value(key, value)

# Utility functions
def safe_get(dict_obj: Dict[str, Any], *keys: str, default: T = None) -> Union[Any, T]:
    for key in keys:
        if isinstance(dict_obj, dict) and key in dict_obj:
            dict_obj = dict_obj[key]
        else:
            return default
    return dict_obj

def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (ValueError, TypeError):
        return default

def safe_array_value(array: Optional[np.ndarray], y: int, x: int, width: int) -> float:
    if array is None or array.size == 0:
        return 0.0
    try:
        if array.ndim == 2:
            return float(array[y, x])
        elif array.ndim == 1:
            index: int = y * width + x
            return float(array[index]) if 0 <= index < array.size else 0.0
        else:
            return 0.0
    except IndexError:
        return 0.0

# Error handling
def log_error(message: str) -> None:
    logging.exception(message)


def handle_processing_error(error_message: str) -> None:
    full_error: str = f"An error occurred during processing:\n\n{error_message}\n\nFull traceback:\n{traceback.format_exc()}"
    st.error(full_error)
    logging.error(full_error)

# Getter and setter functions for specific values
def kernel_size(size: Optional[int] = None) -> int:
    try:
        if size is None:
            return get_value('kernel_size', DEFAULT_KERNEL_SIZE)
        else:
            if not isinstance(size, int) or size < 3 or size % 2 == 0:
                raise ValueError("Kernel size must be an odd integer >= 3")
            set_value('kernel_size', size)
            return size
    except Exception as e:
        handle_processing_error(f"Error accessing kernel size: {str(e)}")
        return DEFAULT_KERNEL_SIZE

def get_gaussian_noise_settings() -> Dict[str, Any]:
    return get_value('gaussian_noise', DEFAULT_VALUES['gaussian_noise'])

def set_gaussian_noise_settings(enabled: bool, mean: float, std_dev: float) -> None:
    set_value('gaussian_noise', {
        "enabled": enabled,
        "mean": mean,
        "std_dev": std_dev,
    })

def get_normalization_options() -> Dict[str, Any]:
    return get_value('normalization', DEFAULT_VALUES['normalization'])

def get_viz_config() -> Dict[str, Any]:
    try:
        return get_value('viz_config', {})
    except Exception as e:
        handle_processing_error(f"Error getting visualization config: {str(e)}")
        return {}

def set_viz_config(config: Dict[str, Any]) -> None:
    try:
        set_value('viz_config', config)
    except Exception as e:
        handle_processing_error(f"Error setting visualization config: {str(e)}")

# Additional getter functions
def get_color_map() -> str:
    return get_value('color_map', 'gray')

def get_show_per_pixel_processing() -> bool:
    return get_value('show_per_pixel', False)

def get_use_whole_image() -> bool:
    return get_value("use_whole_image", True)

def set_use_whole_image(value: bool) -> None:
    set_value('use_whole_image', value)

def get_nlm_options() -> Dict[str, Any]:
    return get_value('nlm_options', DEFAULT_VALUES['nlm_options'])

def get_image_file() -> str:
    return get_value('image_file', '')

def set_image_file(filename: str) -> None:
    set_value('image_file', filename)

def reset_session_state() -> None:
    try:
        st.session_state.clear()
        initialize_session_state()
    except Exception as e:
        handle_processing_error(f"Error resetting session state: {str(e)}")

def update_nlm_params(filter_strength: float, search_window_size: int, use_whole_image: bool) -> None:
    try:
        nlm_options = get_nlm_options()
        nlm_options.update({
            "filter_strength": filter_strength,
            "search_window_size": search_window_size,
            "use_whole_image": use_whole_image
        })
        set_value('nlm_options', nlm_options)
    except Exception as e:
        handle_processing_error(f"Error updating NLM parameters: {str(e)}")

def setup_tabs():
    if 'tabs' not in st.session_state:
        st.session_state.tabs = {
            'speckle': st.empty(),
            'nlm': st.empty(),
            'compare': st.empty()
        }

def get_tabs():
    if 'tabs' not in st.session_state:
        setup_tabs()
    return st.session_state.tabs

def get_last_processed_pixel() -> Tuple[int, int]:
    return get_value('last_processed_pixel', (0, 0))

def set_last_processed_pixel(x: int, y: int) -> None:
    set_value('last_processed_pixel', (x, y))