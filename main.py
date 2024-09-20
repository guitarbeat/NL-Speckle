import streamlit as st
from src.utils import ImageComparison, calculate_processing_details
from src.plotting import prepare_comparison_images, setup_and_run_analysis_techniques, DEFAULT_KERNEL_SIZE, VisualizationConfig
from src.sidebar import SidebarUI
import hashlib
import time

APP_CONFIG = {
    "page_title": "Speckle Contrast Visualization",
    "layout": "wide",
    "page_icon": "favicon.png",
    "initial_sidebar_state": "expanded"
}

def main():
    st.set_page_config(**APP_CONFIG)
    st.logo("media/logo.png")
    initialize_session_state()
    
    try:
        run_application()
    except Exception as e:
        st.error("An unexpected error occurred. Please try reloading the application.")
        st.exception(e)

def initialize_session_state():
    if 'session_id' not in st.session_state:
        st.session_state.session_id = hashlib.md5(str(time.time()).encode()).hexdigest()
    if 'color_map' not in st.session_state:
        config = VisualizationConfig()
        st.session_state.color_map = config.color_map
    if 'techniques' not in st.session_state:
        st.session_state.techniques = ['speckle', 'nlm']

def run_application():
    sidebar_params = setup_sidebar()
    tabs = setup_tabs()
    params = get_analysis_params(sidebar_params)
    run_analysis(params)
    handle_image_comparison(tabs)

def setup_sidebar():
    sidebar_params = SidebarUI.setup()
    st.session_state.sidebar_params = sidebar_params
    return sidebar_params

def setup_tabs():
    tabs = st.tabs(["Speckle", "NL-Means", "Image Comparison"])
    st.session_state.tabs = tabs
    return tabs

def get_analysis_params(sidebar_params):
    kernel_size = sidebar_params.get('kernel_size', DEFAULT_KERNEL_SIZE)
    pixels_to_process = None if sidebar_params['show_per_pixel_processing'] else sidebar_params['pixels_to_process']
    
    details = calculate_processing_details(sidebar_params['image_array'], kernel_size, pixels_to_process)
    
    return {
        "image_array": sidebar_params['image_array'],
        "show_per_pixel_processing": sidebar_params['show_per_pixel_processing'],
        "total_pixels": details.valid_dimensions.width * details.valid_dimensions.height,
        "pixels_to_process": details.pixels_to_process,
        "image_dimensions": details.image_dimensions,
        "kernel_size": kernel_size,
        "search_window_size": sidebar_params.get('search_window_size'),
        "filter_strength": sidebar_params.get('filter_strength'),
        "processing_details": details,
        "use_full_image": sidebar_params.get('use_full_image', False)
    }

def run_analysis(params):
    st.session_state.analysis_params = params
    setup_and_run_analysis_techniques(params)

def handle_image_comparison(tabs):
    with tabs[2]:
        comparison_images = prepare_comparison_images()
        ImageComparison.handle(tabs[2], st.session_state.color_map, comparison_images)

if __name__ == "__main__":
    main()