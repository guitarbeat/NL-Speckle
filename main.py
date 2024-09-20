import streamlit as st
import streamlit_nested_layout  # noqa: F401
from src.utils import (
    ImageComparison,
    calculate_processing_details,
    # ProcessingDetails
)
from src.plotting import (
    SidebarUI,
    prepare_comparison_images,
    setup_and_run_analysis_techniques,
    DEFAULT_KERNEL_SIZE,
    DEFAULT_COLOR_MAP,
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
    """Main function to run the Streamlit application."""
    st.set_page_config(**APP_CONFIG)
    st.logo("media/logo.png")

    # Generate a unique session ID if not already present
    if 'session_id' not in st.session_state:
        st.session_state.session_id = hashlib.md5(str(time.time()).encode()).hexdigest()

    try:
        # Run the main application logic
        run_application()
    except Exception as e:
        st.error("An unexpected error occurred. Please try reloading the application.")
        st.exception(e)

def run_application():
    """Run the main application logic."""
    setup_sidebar_and_tabs()
    initialize_session_state()
    params = get_analysis_params()
    run_analysis(params)
    handle_image_comparison()

def setup_sidebar_and_tabs():
    sidebar_params = SidebarUI.setup()
    st.session_state.tabs = st.tabs(["Speckle", "NL-Means", "Image Comparison"])
    st.session_state.sidebar_params = sidebar_params

def initialize_session_state():
    if 'color_map' not in st.session_state:
        st.session_state.color_map = DEFAULT_COLOR_MAP
    if 'techniques' not in st.session_state:
        st.session_state.techniques = ['speckle', 'nlm']

# Updated get_analysis_params function
def get_analysis_params():
    sidebar_params = st.session_state.sidebar_params
    kernel_size = sidebar_params.get('kernel_size', DEFAULT_KERNEL_SIZE)
    pixels_to_process = None if sidebar_params['show_per_pixel_processing'] else sidebar_params['pixels_to_process']
    
    details = calculate_processing_details(
        sidebar_params['image_array'],
        kernel_size,
        pixels_to_process
    )
    
    return {
        "image_array": sidebar_params['image_array'],
        "show_per_pixel_processing": sidebar_params['show_per_pixel_processing'],
        "total_pixels": details.valid_dimensions.width * details.valid_dimensions.height,
        "pixels_to_process": details.pixels_to_process,
        "image_dimensions": details.image_dimensions,
        "kernel_size": kernel_size,
        "search_window_size": sidebar_params.get('search_window_size'),
        "filter_strength": sidebar_params.get('filter_strength'),
        "processing_details": details
    }

def run_analysis(params):
    st.session_state.analysis_params = params
    setup_and_run_analysis_techniques(params)

def handle_image_comparison():
    with st.session_state.tabs[2]:
        comparison_images = prepare_comparison_images()
        ImageComparison.handle(st.session_state.tabs[2], st.session_state.color_map, comparison_images)

if __name__ == "__main__":
    main()