"""
This module manages all session state operations for the Streamlit application.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, TypeVar
from datetime import datetime

import numpy as np
import streamlit as st

# Type variables for better type hinting
T = TypeVar("T")

# Constants
ORIGINAL_IMAGE = "Original Image"
DEFAULT_SPECKLE_VIEW: List[str] = [ORIGINAL_IMAGE, "Speckle Contrast"]
DEFAULT_NLM_VIEW: List[str] = [ORIGINAL_IMAGE, "Non-Local Means"]
DEFAULT_KERNEL_SIZE: int = 3
DEFAULT_FILTER_STRENGTH: float = 10.0
DEFAULT_SEARCH_WINDOW_SIZE: int = 50

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
    "processing_config": {
        "percentage": 100,
        "exact_count": 0,
    },
    "filters": {
        "nlm": DEFAULT_NLM_VIEW,
        "speckle": DEFAULT_SPECKLE_VIEW,
    },
    "techniques": ["speckle", "nlm"],
    "speckle_result": None,
    "nlm_result": None, 
    "pixels_to_process": 0,
    "desired_exact_count": 0,
    "processed_image_np": None,
    "image_file": None,
    "last_processed_pixel": (0, 0),
}

def set_show_per_pixel_processing(show_per_pixel: bool):
    set_session_state("show_per_pixel", show_per_pixel)
    
def initialize_session_state() -> None:
    """Initialize the session state with default values."""
    for key, value in DEFAULT_VALUES.items():
        try:
            if key not in st.session_state:
                st.session_state[key] = value
        except Exception as e:
            error_message = f"Error initializing session state for key '{key}': {str(e)}"
            logging.error(error_message)
            st.error(error_message)

    # Log successful initialization
    logging.info("Session state initialized successfully")

def get_session_state(key: str, default: Optional[T] = None) -> T:
    """Get a value from the session state."""
    return st.session_state.get(key, default)

def set_session_state(key: str, value: Any) -> None:
    """Set a value in the session state."""
    try:
        st.session_state[key] = value
    except Exception as e:
        handle_error(f"Error setting session state for key '{key}': {str(e)}")
        
def update_nested_session_state(keys: List[str], value: Any) -> None:
    """Update a nested value in the session state."""
    current = st.session_state
    for key in keys[:-1]:
        if key not in current:
            current[key] = {}
        current = current[key]
    current[keys[-1]] = value

# Image Processing Functions

def update_pixel_processing_values() -> None:
    """Update pixel processing values based on current image."""
    try:
        image_array = get_session_state("image_array")
        if image_array is not None and image_array.size > 0:
            total_pixels = calculate_total_pixels()
            set_session_state("desired_exact_count", total_pixels)
            set_session_state("pixels_to_process", total_pixels)
        else:
            logging.warning("No valid image array found when updating pixel processing values.")
    except Exception as e:
        handle_error(f"Error updating pixel processing values: {str(e)}")

def calculate_total_pixels() -> int:
    """Calculate the total number of pixels to process."""
    try:
        width, height = get_image_dimensions()
        if width is None or height is None:
            raise ValueError("Width or height is None.")
        
        if not isinstance(width, int) or not isinstance(height, int):
            raise TypeError(f"Width and height must be integers. Got: width={type(width)}, height={type(height)}")
        
        kernel = get_session_state("kernel_size", DEFAULT_KERNEL_SIZE)
        return max((width - kernel + 1) * (height - kernel + 1), 0)
    except Exception as e:
        handle_error(f"Error calculating total pixels: {str(e)}")
        return 0
        
def get_image_dimensions() -> Tuple[int, int]:
    """Get the dimensions of the current image."""
    try:
        image_array = get_session_state("image_array")
        return (
            image_array.shape[:2] if isinstance(image_array, np.ndarray) else (0, 0)  
        )
    except Exception as e:
        handle_error(f"Error getting image dimensions: {str(e)}")
        return (0, 0)

def get_image_array() -> np.ndarray:
    """Get the current image array."""
    return get_session_state("image_array", np.array([]))

# Filters and Techniques 

def get_filter_options(technique: str) -> List[str]:
    """Get filter options for a specific technique."""
    options = {
        "speckle": [
            ORIGINAL_IMAGE,
            "Mean Filter", 
            "Std Dev Filter",
            "Speckle Contrast",
        ],
        "nlm": [
            ORIGINAL_IMAGE,
            "Non-Local Means",
            "Normalization Factors",
            "Last Similarity Map", 
            "Non-Local Standard Deviation",
            "Non-Local Speckle",
        ],
    }
    return options.get(technique, [ORIGINAL_IMAGE])

def update_filter_selection(technique: str, selected_filters: List[str]) -> None:
    """Update the filter selection for a specific technique."""
    set_session_state(f"filter_selections.{technique}", selected_filters)
    
def get_filter_selection(technique: str) -> List[str]:
    """Get the current filter selection for a technique."""
    default = DEFAULT_SPECKLE_VIEW if technique == "speckle" else DEFAULT_NLM_VIEW
    return get_session_state(f"filter_selections.{technique}", default)

def set_technique_result(technique: str, result: Any) -> None:
    """Set the result for a specific technique."""
    set_session_state(f"{technique}_result", result)

def get_technique_result(technique: str) -> Any:
    """Get the result for a specific technique."""
    try:
        return get_session_state(f"{technique}_result")
    except Exception as e:
        handle_error(f"Error getting {technique} result: {str(e)}")
        return None
        
def get_technique_params(technique: str) -> Dict[str, Any]:
    params = {
        "kernel_size": kernel_size(),  # This should get the most recent kernel size
    }
    if technique == "nlm":
        nlm_options = get_nlm_options()
        params.update(
            {
                "search_window_size": nlm_options["search_window_size"],
                "filter_strength": nlm_options["filter_strength"],
                "use_full_image": nlm_options["use_whole_image"],
            }
        )
    return params

# Utility Functions

def safe_get(dict_obj: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    """Safely get a nested value from a dictionary."""
    for key in keys:
        if isinstance(dict_obj, dict) and key in dict_obj:
            dict_obj = dict_obj[key]
        else:
            return default
    return dict_obj




# Error Handling 

def handle_error(error_message: str) -> None:
    """Handle an error by logging it and displaying to user."""
    st.error(f"An error occurred: {error_message}")
    st.exception(error_message)

# Getter and Setter Functions

def kernel_size(size: Optional[int] = None) -> int:
    """Get or set the kernel size."""
    if size is None:
        return get_session_state("kernel_size", DEFAULT_KERNEL_SIZE)
    else:
        if size < 3 or size % 2 == 0:
            raise ValueError("Kernel size must be an odd integer >= 3")
        set_session_state("kernel_size", size)
        return size
        
def get_gaussian_noise_settings() -> Dict[str, Any]:
    """Get the current Gaussian noise settings."""
    return safe_get(st.session_state, "gaussian_noise", default=DEFAULT_VALUES["gaussian_noise"])

def set_gaussian_noise_settings(enabled: bool, mean: float, std_dev: float) -> None:
    """Set the Gaussian noise settings."""
    set_session_state(
        "gaussian_noise",
        {"enabled": enabled, "mean": mean, "std_dev": std_dev},
    )
    
def get_normalization_options() -> Dict[str, Any]:
    """Get the current normalization options."""
    return safe_get(st.session_state, "normalization", default=DEFAULT_VALUES["normalization"])

    
def get_color_map() -> str:
    """Get the current color map."""
    return get_session_state("color_map", "gray")

def get_show_per_pixel_processing() -> bool:
    """Get the current setting for showing per-pixel processing."""
    return get_session_state("show_per_pixel", False)

def get_use_whole_image() -> bool:
    """Get the current setting for using the whole image."""
    return get_session_state("use_whole_image", True)  

def set_use_whole_image(value: bool) -> None:
    """Set the setting for using the whole image."""
    set_session_state("use_whole_image", value)

def get_nlm_options() -> Dict[str, Any]:
    """Get the current NLM options."""
    return get_session_state("nlm_options", DEFAULT_VALUES["nlm_options"]) 

def get_image_file() -> str:
    """Get the current image file name."""
    return get_session_state("image_file", "")

def set_image_file(filename: str) -> None:
    """Set the current image file name.""" 
    set_session_state("image_file", filename)

def reset_session_state() -> None:
    """Reset the session state to default values."""
    try:
        st.session_state.clear()
        initialize_session_state()
    except Exception as e:
        handle_error(f"Error resetting session state: {str(e)}")

def update_nlm_params(
    filter_strength: float, search_window_size: int, use_whole_image: bool
) -> None:
    """Update the NLM parameters."""
    nlm_options = safe_get(st.session_state, "nlm_options", default={})
    nlm_options.update(
        {
            "filter_strength": filter_strength,
            "search_window_size": search_window_size,
            "use_whole_image": use_whole_image,
        }
    )
    set_session_state("nlm_options", nlm_options)
    



def get_last_processed_pixel() -> Tuple[int, int]:
    """Get the last processed pixel coordinates."""
    return get_session_state("last_processed_pixel", (0, 0))

def set_last_processed_pixel(x: int, y: int) -> None:
    """Set the last processed pixel coordinates."""
    set_session_state("last_processed_pixel", (x, y))

def clear_technique_result(technique: str) -> None:
    """Clear the result for a specific technique."""
    set_session_state(f"{technique}_result", None)
    logging.info(f"Cleared result for technique: {technique}")

def handle_processing_error(error_msg: str):
    """Handle processing errors by logging them and updating the session state."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    full_error_msg = f"[{timestamp}] {error_msg}"
    
    logging.error(full_error_msg)
    
    # Add error to error history
    error_history = get_session_state("error_history", [])
    error_history.append(full_error_msg)
    set_session_state("error_history", error_history[-10:])  # Keep last 10 errors
    
    set_session_state("last_error", full_error_msg)
    st.error(f"Processing error: {error_msg}")

