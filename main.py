"""
This module serves as the main entry point for the Streamlit application.
It imports necessary utilities and plotting functions for image comparison.
"""

from typing import List
import streamlit as st
from src.sidebar import setup_ui
from src.utils import handle_image_comparison
from src.images import process_and_visualize_image

# App Configuration
APP_CONFIG = {
    "page_title": "Speckle Contrast Visualization",
    "layout": "wide",
    "page_icon": "favicon.png",
    "initial_sidebar_state": "expanded",
}

# Color Maps
AVAILABLE_COLOR_MAPS = [
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

SPECKLE_CONTRAST, ORIGINAL_IMAGE, NON_LOCAL_MEANS = "Speckle Contrast", "Original Image", "Non-Local Means"
DEFAULT_SPECKLE_VIEW, DEFAULT_NLM_VIEW = [ORIGINAL_IMAGE, SPECKLE_CONTRAST], [ORIGINAL_IMAGE, NON_LOCAL_MEANS]

def main():
    """Main function to set up the Streamlit app configuration, logo, and run the application."""
    setup_streamlit_config()
    
    try:
        setup_app()
    except (ValueError, TypeError) as e:
        st.error(f"An error occurred: {e}. Please check your input and try again.")

def setup_streamlit_config():
    """Set up Streamlit configuration and logo."""
    st.set_page_config(**APP_CONFIG)
    st.logo("media/logo.png")
    
    if "techniques" not in st.session_state:
        st.session_state.techniques = ["speckle", "nlm"]

def setup_app():
    """Set up the main application components."""
    sidebar_params = setup_ui()
    if sidebar_params is None:
        st.warning("Please upload an image in the sidebar to begin.")
        return
    if "image_array" not in st.session_state:
        st.warning("No image data found. Please try uploading the image again.")
        return
    
    tabs = setup_tabs()
    
    # Check if all required parameters are present
    required_keys = ["image_array", "kernel_size", "exact_pixel_count", "search_window_size", "filter_strength"]
    if all(key in st.session_state for key in required_keys):
        process_and_display_results(tabs)
    else:
        st.warning("Some required parameters are missing. Please ensure all settings are properly configured.")

def process_and_display_results(tabs):
    """Process the image and display results in the appropriate tabs."""
    techniques: List[str] = st.session_state.get("techniques", [])
    
    for technique, tab in zip(techniques, tabs):
        if tab is not None:
            with tab:
                process_and_visualize_image(technique, tab)

    with tabs[2]:
        handle_image_comparison(tabs[2])

def setup_tabs() -> List:
    """Set up the tabs for the application."""
    tabs = st.tabs(["Speckle", "NL-Means", "Image Comparison"])
    st.session_state.tabs = tabs
    return tabs

if __name__ == "__main__":
    main()
