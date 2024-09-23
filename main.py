"""
This module serves as the main entry point for the Streamlit application.
It imports necessary utilities and plotting functions for image comparison.
"""

import hashlib
import time
import streamlit as st
from src.utils import ImageComparison
from src.plotting import (
    prepare_comparison_images,
    VisualizationConfig,
    run_technique
)
from typing import List
from src.sidebar import SidebarUI
from src.processing import calculate_processing_details
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


APP_CONFIG = {
    "page_title": "Speckle Contrast Visualization",
    "layout": "wide",
    "page_icon": "favicon.png",
    "initial_sidebar_state": "expanded",
}

def main():
    """Main function to set up the Streamlit app configuration, logo, and run the application."""
    st.set_page_config(**APP_CONFIG)
    st.logo("media/logo.png")

    if "session_id" not in st.session_state:
        st.session_state.session_id = hashlib.md5(str(time.time()).encode()).hexdigest()
    if "color_map" not in st.session_state:
        config = VisualizationConfig()
        st.session_state.color_map = config.color_map
    if "techniques" not in st.session_state:
        st.session_state.techniques = ["speckle", "nlm"]

    try:
        setup_app(st)
    except (ValueError, TypeError) as e:  # Specify the exceptions you expect
        st.error(f"An error occurred: {e}. Please check your input and try again.")


def setup_app(st):
    sidebar_params = SidebarUI.setup()
    st.session_state.sidebar_params = sidebar_params

    tabs = st.tabs(["Speckle", "NL-Means", "Image Comparison"])
    st.session_state.tabs = tabs

    kernel_size = sidebar_params.get("kernel_size", 5)
    pixels_to_process = (
        None
        if sidebar_params["show_per_pixel_processing"]
        else sidebar_params["pixels_to_process"]
    )

    details = calculate_processing_details(
        sidebar_params["image_array"], kernel_size, pixels_to_process
    )

    params = {
        "image_array": sidebar_params["image_array"],
        "show_per_pixel_processing": sidebar_params["show_per_pixel_processing"],
        "total_pixels": details.valid_dimensions[0] * details.valid_dimensions[1],  # Change here
        "pixels_to_process": details.pixels_to_process,
        "image_dimensions": details.image_dimensions,
        "kernel_size": kernel_size,
        "search_window_size": sidebar_params.get("search_window_size"),
        "filter_strength": sidebar_params.get("filter_strength"),
        "processing_details": details,
        "use_full_image": sidebar_params.get("use_full_image", False),
    }

    st.session_state.analysis_params = params

    techniques: List[str] = st.session_state.get("techniques", [])

    for technique, tab in zip(techniques, tabs):
        if tab is not None:
            with tab:
                run_technique(
                    technique,
                    tab,
                    params,
                )

    with tabs[2]:
        comparison_images = prepare_comparison_images()
        ImageComparison.handle(tabs[2], st.session_state.color_map, comparison_images)
        
if __name__ == "__main__":
    main()