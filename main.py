"""
This module serves as the main entry point for the Streamlit application.
It imports necessary utilities and plotting functions for image comparison.
"""

from typing import List
import streamlit as st
from src.plotting import prepare_comparison_images, run_technique
from src.sidebar import setup_sidebar
from src.utils import handle_image_comparison
from src.config import APP_CONFIG
from src.images import process_image

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
    st.logo("media/logo.png")  # Changed from st.logo to st.image
    
    if "techniques" not in st.session_state:
        st.session_state.techniques = ["speckle", "nlm"]

def setup_app():
    """Set up the main application components."""
    sidebar_params = setup_sidebar()
    if sidebar_params is None:
        st.warning("Please upload an image in the sidebar to begin.")
        return
    if "image_array" not in st.session_state:
        st.warning("No image data found. Please try uploading the image again.")
        return
    
    tabs = setup_tabs()
    
    # Move this block inside the function to ensure it runs on every rerun
    required_keys = ["image_array", "kernel_size", "exact_pixel_count", "search_window_size", "filter_strength"]
    if all(key in st.session_state for key in required_keys):
        try:
            nl_speckle_result = process_image()
            if nl_speckle_result is not None:
                st.session_state.nl_speckle_result = nl_speckle_result
        except Exception as e:
            st.error(f"Failed to process image: {e}")
    
    display_results(tabs)

def display_results(tabs):
    """Display the results in the appropriate tabs."""
    techniques: List[str] = st.session_state.get("techniques", [])
    
    for technique, tab in zip(techniques, tabs):
        if tab is not None:
            with tab:
                run_technique(technique, tab)

    with tabs[2]:
        comparison_images = prepare_comparison_images()
        handle_image_comparison(tabs[2], comparison_images)

def setup_tabs() -> List:
    """Set up the tabs for the application."""
    tabs = st.tabs(["Speckle", "NL-Means", "Image Comparison"])
    st.session_state.tabs = tabs
    return tabs

if __name__ == "__main__":
    main()
