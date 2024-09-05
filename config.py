import streamlit as st
from typing import Dict, List, Tuple

# Page Configuration
PAGE_CONFIG = {
    "page_title": "Speckle Contrast Visualization",
    "layout": "wide",
    "page_icon": "favicon.png",
    "initial_sidebar_state": "expanded",
}

# Image Configuration
LOGO_PATH = "media/logo.png"
PRELOADED_IMAGES: Dict[str, str] = {
    "image50.png": "media/image50.png",
    "spatial.tif": "media/spatial.tif",
    "logo.jpg": "media/logo.jpg",
}

# Color Maps
COLOR_MAPS: List[str] = ["viridis", "plasma", "inferno", "magma", "cividis", "gray","pink"]

# Default Values
DEFAULT_KERNEL_SIZE = 5
DEFAULT_STRIDE = 1
DEFAULT_SEARCH_WINDOW_STEP = 1
DEFAULT_FILTER_STRENGTH = 10.0
DEFAULT_ANIMATION_SPEED = 0.001

# Slider Ranges
KERNEL_SIZE_RANGE = (1, 11)
STRIDE_RANGE = (1, 5)
FILTER_STRENGTH_RANGE = (0.1, 50.0, 0.1)  # min, max, step
ANIMATION_SPEED_RANGE = (0.0001, 0.005, 0.0001) # min, max, step

# Function to get dynamic search window size range
def get_search_window_size_range(kernel_size: int, image_size: Tuple[int, int]) -> Tuple[int, int, int]:
    """
    Calculate the range for the search window size slider based on kernel size and image dimensions.
    
    :param kernel_size: Size of the kernel
    :param image_size: Tuple containing (width, height) of the image
    :return: Tuple of (min, max, default) for the search window size slider
    """
    min_size = kernel_size + 2
    max_size = min(max(image_size) // 2, 35)  
    return min_size, max_size, DEFAULT_SEARCH_WINDOW_STEP

# Session State Keys
SESSION_STATE_KEYS = [
    "is_animating",
    "cache",
    "animation_mode"
]



# Function to set page config
def set_page_config():
    st.set_page_config(**PAGE_CONFIG)
    st.logo(LOGO_PATH)
