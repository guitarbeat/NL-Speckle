import streamlit as st
import streamlit_nested_layout  # noqa: F401
from shared_types import (
    ImageComparison,
    SidebarUI,
    calculate_processing_details,
    DEFAULT_KERNEL_SIZE,
    DEFAULT_COLOR_MAP,
    ProcessingDetails
)
from frontend.plotting import (
    prepare_comparison_images,
    setup_and_run_analysis_techniques
)
import hashlib
import time

# Application configuration
APP_CONFIG = {
    "page_title": "Speckle Contrast Visualization",
    "layout": "wide",
    "page_icon": "favicon.png",
    "initial_sidebar_state": "expanded"
}

def main():
    st.set_page_config(**APP_CONFIG)
    st.logo("media/logo.png")

    # Generate a unique session ID if not already present
    if 'session_id' not in st.session_state:
        st.session_state.session_id = hashlib.md5(str(time.time()).encode()).hexdigest()

    try:
        run_application()
    except Exception as e:
        st.error("An unexpected error occurred. Please try reloading the application.")
        st.exception(e)

def run_application():
    # Set up the sidebar and create tabs
    sidebar_params = SidebarUI.setup()
    tabs = create_tabs()

    # Initialize color_map in session state if it doesn't exist
    if 'color_map' not in st.session_state:
        st.session_state.color_map = DEFAULT_COLOR_MAP
    if 'techniques' not in st.session_state:
        st.session_state.techniques = ['speckle', 'nlm']  # Add or remove techniques as needed

    # Use the user-selected kernel size from sidebar_params, defaulting to DEFAULT_KERNEL_SIZE
    kernel_size = sidebar_params.get('kernel_size', DEFAULT_KERNEL_SIZE)

    # Calculate processing details based on user input
    details: ProcessingDetails = calculate_processing_details(
        sidebar_params['image_array'],
        kernel_size,
        None if sidebar_params['show_per_pixel_processing'] else sidebar_params['pixels_to_process']
    )

    # Store important data in session state for persistence
    st.session_state.tabs = tabs
    st.session_state.sidebar_params = sidebar_params
    st.session_state.analysis_params = {
        "image_array": sidebar_params['image_array'],
        "show_per_pixel_processing": sidebar_params['show_per_pixel_processing'],
        "total_pixels": sidebar_params['total_pixels'],
        "pixels_to_process": sidebar_params['pixels_to_process'],
        "image_dimensions": details.image_dimensions,
        "kernel_size": kernel_size,
    }

    # Perform analysis and handle image comparison
    setup_and_run_analysis_techniques(st.session_state.analysis_params)

    # Only show comparison UI in the third tab
    with tabs[2]:
        comparison_images = prepare_comparison_images()
        ImageComparison.handle(tabs[2], st.session_state.color_map, comparison_images)

def create_tabs():
    """Create and return the main application tabs"""
    return st.tabs(["Speckle", "NL-Means", "Image Comparison"])

if __name__ == "__main__":
    main()