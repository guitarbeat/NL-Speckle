"""
This module serves as the main entry point for the Streamlit application.
It imports necessary utilities and plotting functions for image comparison.
"""

import hashlib
import time
import streamlit as st  # Moved this import below hashlib
from src.utils import ImageComparison, calculate_processing_details
from src.plotting import (
    prepare_comparison_images,
    setup_and_run_analysis_techniques,
    DEFAULT_KERNEL_SIZE,
    VisualizationConfig,
)
from src.sidebar import SidebarUI

APP_CONFIG = {
    "page_title": "Speckle Contrast Visualization",
    "layout": "wide",
    "page_icon": "favicon.png",
    "initial_sidebar_state": "expanded",
}


def main():
    """Main function to set up the Streamlit app configuration and logo."""
    st.set_page_config(**APP_CONFIG)
    st.logo("media/logo.png")
    initialize_session_state()

    try:
        run_application()
    except (ValueError, TypeError) as e:  # Specify the exceptions you expect
        st.error(f"An error occurred: {e}. Please check your input and try again.")


def initialize_session_state():
    """Initialize the session state with a unique session ID."""
    if "session_id" not in st.session_state:
        st.session_state.session_id = hashlib.md5(str(time.time()).encode()).hexdigest()
    if "color_map" not in st.session_state:
        config = VisualizationConfig()
        st.session_state.color_map = config.color_map
    if "techniques" not in st.session_state:
        st.session_state.techniques = ["speckle", "nlm"]


def run_application():
    """Run the main application, setting up the sidebar and tabs."""
    sidebar_params = setup_sidebar()
    tabs = setup_tabs()
    params = get_analysis_params(sidebar_params)
    run_analysis(params)
    handle_image_comparison(tabs)


def setup_sidebar():
    """Sets up the sidebar parameters for the application."""
    sidebar_params = SidebarUI.setup()
    st.session_state.sidebar_params = sidebar_params
    return sidebar_params


def setup_tabs():
    """Sets up the tabs for the application."""
    tabs = st.tabs(["Speckle", "NL-Means", "Image Comparison"])
    st.session_state.tabs = tabs
    return tabs


def get_analysis_params(sidebar_params):
    """Retrieve analysis parameters from the sidebar."""
    kernel_size = sidebar_params.get("kernel_size", DEFAULT_KERNEL_SIZE)
    pixels_to_process = (
        None
        if sidebar_params["show_per_pixel_processing"]
        else sidebar_params["pixels_to_process"]
    )

    details = calculate_processing_details(
        sidebar_params["image_array"], kernel_size, pixels_to_process
    )

    return {
        "image_array": sidebar_params["image_array"],
        "show_per_pixel_processing": sidebar_params["show_per_pixel_processing"],
        "total_pixels": details.valid_dimensions.width
        * details.valid_dimensions.height,
        "pixels_to_process": details.pixels_to_process,
        "image_dimensions": details.image_dimensions,
        "kernel_size": kernel_size,
        "search_window_size": sidebar_params.get("search_window_size"),
        "filter_strength": sidebar_params.get("filter_strength"),
        "processing_details": details,
        "use_full_image": sidebar_params.get("use_full_image", False),
    }


def run_analysis(params):
    """Run analysis with the given parameters."""
    st.session_state.analysis_params = params
    setup_and_run_analysis_techniques(params)


def handle_image_comparison(tabs):
    """Handles the comparison of images in the specified tab."""
    with tabs[2]:
        comparison_images = prepare_comparison_images()
        ImageComparison.handle(tabs[2], st.session_state.color_map, comparison_images)


if __name__ == "__main__":
    main()
