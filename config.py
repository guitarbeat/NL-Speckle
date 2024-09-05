import streamlit as st
from typing import Dict, List

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
DEFAULT_KERNEL_SIZE = 3
DEFAULT_STRIDE = 1
DEFAULT_SEARCH_WINDOW_SIZE = 5
DEFAULT_FILTER_STRENGTH = 10.0
DEFAULT_ANIMATION_SPEED = 0.01

# Slider Ranges
KERNEL_SIZE_RANGE = (1, 10)
STRIDE_RANGE = (1, 5)
SEARCH_WINDOW_SIZE_RANGE = (1, 21, 2)  # min, max, step
FILTER_STRENGTH_RANGE = (0.1, 50.0, 0.1)  # min, max, step
ANIMATION_SPEED_RANGE = (0.0001, 0.005, 0.0001) # min, max, step

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
