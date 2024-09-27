"""
This module serves as the main entry point for the Streamlit application.
It imports necessary utilities and plotting functions for image comparison.
"""

from typing import List, Dict, Any

import streamlit as st
from src.plotting import prepare_comparison_images, run_technique
from src.sidebar import SidebarUI
from src.utils import ImageComparison
from src.config import APP_CONFIG
from src.images import process_nl_speckle


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
    sidebar_params = setup_sidebar()
    if sidebar_params is None:
        st.warning("Please upload an image in the sidebar to begin.")
        return
    if sidebar_params.get("image_array") is None:
        st.warning("No image data found. Please try uploading the image again.")
        return
    tabs = setup_tabs()
    params = prepare_analysis_params(sidebar_params)
    
    process_image(params)
    display_results(tabs, params)

def setup_sidebar():
    """Set up the sidebar and return its parameters."""
    sidebar_params = SidebarUI.setup()
    st.session_state.sidebar_params = sidebar_params
    return sidebar_params

def setup_tabs():
    """Set up the tabs for the application."""
    tabs = st.tabs(["Speckle", "NL-Means", "Image Comparison"])
    st.session_state.tabs = tabs
    return tabs

def prepare_analysis_params(sidebar_params):
    """Prepare the analysis parameters based on sidebar inputs."""
    return {
        "image_array": sidebar_params["image_array"],
        "show_per_pixel_processing": sidebar_params["show_per_pixel_processing"],
        "total_pixels": sidebar_params["image_array"].size,
        "pixels_to_process": st.session_state.pixels_to_process,
        "image_dimensions": sidebar_params["image_array"].shape,
        "kernel_size": sidebar_params.get("kernel_size", 5),
        "search_window_size": sidebar_params.get("search_window_size"),
        "filter_strength": sidebar_params.get("filter_strength"),
        "use_full_image": sidebar_params.get("use_full_image"),
    }

def process_image(params: Dict[str, Any]) -> None:
    """Process the image using NL-Speckle."""
    nl_speckle_result = process_nl_speckle(
        image=params["image_array"],
        kernel_size=params["kernel_size"],
        pixels_to_process=params["pixels_to_process"],
        nlm_search_window_size=params["search_window_size"],
        nlm_h=params["filter_strength"]
    )
    st.session_state.nl_speckle_result = nl_speckle_result

def display_results(tabs, params):
    """Display the results in the appropriate tabs."""
    techniques: List[str] = st.session_state.get("techniques", [])
    
    for technique, tab in zip(techniques, tabs):
        if tab is not None:
            with tab:
                run_technique(
                    technique,
                    tab,
                    params,
                    st.session_state.nl_speckle_result
                )

    with tabs[2]:
        comparison_images = prepare_comparison_images()
        ImageComparison.handle(tabs[2], st.session_state.color_map, comparison_images)


if __name__ == "__main__":
    main()
