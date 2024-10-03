"""
This module manages all session state operations for the Streamlit application.
"""

from typing import Any, Dict, List, Optional, TypeVar

import numpy as np
import streamlit as st

# Type variables for better type hinting
T = TypeVar("T")
# Color Maps
AVAILABLE_COLOR_MAPS = [
    "viridis_r",
    "viridis",
    "gray",
    "plasma",
    "inferno",
    "magma",
    "pink",
    "hot",
    "cool",
    "YlOrRd",
]

# Preloaded Images
PRELOADED_IMAGE_PATHS = {
    "image50.png": "media/image50.png",
    "spatial.tif": "media/spatial.tif",
    "logo.jpg": "media/logo.jpg",
}

# Constants
ORIGINAL_IMAGE = "Original Image"
DEFAULT_SPECKLE_VIEW: List[str] = [ORIGINAL_IMAGE, "LSCI"]
DEFAULT_NLM_VIEW: List[str] = [ORIGINAL_IMAGE, "NL Means"]
DEFAULT_KERNEL_SIZE: int = 3
DEFAULT_FILTER_STRENGTH: float = 10.0
DEFAULT_SEARCH_WINDOW_SIZE: int = 50

# Constants to define style and other repeated values
DEFAULT_KERNEL_OUTLINE_COLOR = "red"
DEFAULT_KERNEL_CENTER_PIXEL_COLOR = "green"
DEFAULT_SEARCH_WINDOW_COLOR = "blue"
DEFAULT_PIXEL_TEXT_COLOR = "white"
DEFAULT_PIXEL_FONT_SIZE = 8


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
        "lsci": DEFAULT_SPECKLE_VIEW,
    },
    "techniques": ["lsci", "nlm"],
    "pixels_to_process": 1,
    "desired_exact_count": 0,
    "processed_image_np": None,
    "image_file": None,
    "last_processed_pixel": (0, 0),
}

#used in sidebar
def set_show_per_pixel_processing(value: bool):
    st.session_state["show_per_pixel"] = value

#used in main
def initialize_session_state() -> None:
    """Initialize the session state with default values."""
    for key, value in DEFAULT_VALUES.items():
        try:
            if key not in st.session_state:
                st.session_state[key] = value
        except Exception as e:
            error_message = (
                f"Error initializing session state for key '{key}': {str(e)}"
            )
            st.error(error_message)

# used in main, images, sidebar, and formula
def get_session_state(key: str, default: Optional[T] = None) -> T:
    """Get a value from the session state."""
    return st.session_state.get(key, default)

# used in images, main, and sidebar
def set_session_state(key: str, value: Any) -> None:
    """Set a value in the session state."""
    try:
        st.session_state[key] = value
    except Exception as e:
        st.error(f"Error setting session state for key '{key}': {str(e)}")

# used in main, images, sidebar, and formula
def get_image_array() -> np.ndarray:
    """Get the current image array."""
    return get_session_state("image_array", np.array([]))

#used in main
def get_filter_options(technique: str) -> List[str]:
    """Get filter options for a specific technique."""
    options = {
        "lsci": [
            ORIGINAL_IMAGE,
            "Mean Filter",
            "Std Dev Filter",
            "LSCI",
        ],
        "nlm": [
            ORIGINAL_IMAGE,
            "NL Means",
            "Normalization Factors",
            "Last Similarity Map",
        ],
    }
    return options.get(technique, [ORIGINAL_IMAGE])

# used in images and main
def get_filter_selection(technique: str) -> List[str]:
    """Get the current filter selection for a technique."""
    default = DEFAULT_SPECKLE_VIEW if technique == "lsci" else DEFAULT_NLM_VIEW
    return get_session_state(f"filter_selections.{technique}", default)

# used in main and images
def get_technique_result(technique: str) -> Optional[Dict[str, Any]]:
    result = get_session_state(f"{technique}_result")
    return result

# used in main and images
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

#used in sidebar, images, formula, overlay, and main
def kernel_size(size: Optional[int] = None) -> int:
    """Get or set the kernel size."""
    if size is None:
        return get_session_state("kernel_size", DEFAULT_KERNEL_SIZE)
    else:
        if size < 3 or size % 2 == 0:
            raise ValueError("Kernel size must be an odd integer >= 3")
        set_session_state("kernel_size", size)
        return size

#used in sidebar
def get_gaussian_noise_settings() -> Dict[str, Any]:
    """Get the current Gaussian noise settings."""
    return st.session_state.get("gaussian_noise", DEFAULT_VALUES["gaussian_noise"])

# used in sidebar
def get_normalization_options() -> Dict[str, Any]:
    """Get the current normalization options."""
    return st.session_state.get("normalization", DEFAULT_VALUES["normalization"])

# used in main and si
def get_color_map() -> str:
    """Get the current color map."""
    return get_session_state("color_map", "gray")

# used in images, sidebar, and formula
def get_nlm_options() -> Dict[str, Any]:
    """Get the current NLM options."""
    return get_session_state("nlm_options", DEFAULT_VALUES["nlm_options"])

def update_nlm_params(filter_strength=None, search_window_size=None, use_whole_image=None):
    """
    Update NLM parameters. If called without arguments, it uses values from session state.
    Otherwise, it updates with the provided values.
    """
    if all(param is None for param in [filter_strength, search_window_size, use_whole_image]):
        # Use values from session state
        st.session_state['nlm_options'] = {
            'use_whole_image': st.session_state.get('use_whole_image_checkbox', False),
            'search_window_size': st.session_state.get('search_window_size_slider', DEFAULT_SEARCH_WINDOW_SIZE),
            'filter_strength': st.session_state.get('filter_strength_slider', DEFAULT_FILTER_STRENGTH)
        }
    else:
        # Update with provided values
        st.session_state['nlm_options'] = {
            'filter_strength': filter_strength if filter_strength is not None else st.session_state['nlm_options'].get('filter_strength', DEFAULT_FILTER_STRENGTH),
            'search_window_size': search_window_size if search_window_size is not None else st.session_state['nlm_options'].get('search_window_size', DEFAULT_SEARCH_WINDOW_SIZE),
            'use_whole_image': use_whole_image if use_whole_image is not None else st.session_state['nlm_options'].get('use_whole_image', False)
        }

    # Ensure search window size is odd
    if st.session_state['nlm_options']['search_window_size'] % 2 == 0:
        st.session_state['nlm_options']['search_window_size'] += 1


# used in main
def needs_processing(technique: str) -> bool:
    """Check if the technique needs processing based on current state."""
    result = get_session_state(f"{technique}_result")
    pixels_to_process = get_session_state("pixels_to_process")
    last_processed = get_session_state(f"{technique}_last_processed", 0)
    return result is None or last_processed != pixels_to_process

# used in main
def set_last_processed(technique: str, value: int) -> None:
    """Set the last processed pixel count for a technique."""
    set_session_state(f"{technique}_last_processed", value)

def update_show_per_pixel():
    st.session_state['show_per_pixel'] = st.session_state['show_per_pixel_toggle']

def update_pixels_to_process():
    st.session_state['pixels_to_process'] = st.session_state['max_pixels_slider']

def update_kernel_size():
    new_kernel_size = st.session_state['kernel_size_input']
    if new_kernel_size < 1 or new_kernel_size % 2 == 0:
        st.warning("Kernel size must be an odd integer >= 1. Adjusting to nearest valid value.")
        new_kernel_size = max(3, new_kernel_size + (new_kernel_size % 2 == 0))
    st.session_state['kernel_size'] = new_kernel_size
    # Reprocess the number of pixels
    st.session_state['pixels_to_process'] = new_kernel_size ** 2
    

def update_nlm_use_whole_image():
    st.session_state['nlm_options']['use_whole_image'] = st.session_state['use_whole_image_checkbox']
    # If using whole image, update search window size to None or max value
    if st.session_state['nlm_options']['use_whole_image']:
        st.session_state['nlm_options']['search_window_size'] = None

def update_nlm_search_window_size():
    new_size = st.session_state['search_window_size_slider']
    # Ensure the search window size is odd
    if new_size % 2 == 0:
        new_size += 1
    st.session_state['nlm_options']['search_window_size'] = new_size

def update_nlm_filter_strength():
    st.session_state['nlm_options']['filter_strength'] = st.session_state['filter_strength_slider']



