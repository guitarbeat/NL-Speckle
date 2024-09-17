import streamlit as st
import streamlit_nested_layout  # noqa: F401
from shared_types import ( ImageComparison,
    SidebarUI,calculate_processing_details)
from frontend.plotting import (
    prepare_comparison_images,
    get_technique_params,
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
        st.session_state.color_map = 'gray'  # Set a default value

    # Use the user-selected kernel size from sidebar_params
    kernel_size = sidebar_params.get('kernel_size', 3)  # Default to 3 if not set


    # Calculate processing details based on user input
    details = calculate_processing_details(
        sidebar_params['image_array'], 
        kernel_size,  # Use the dynamic kernel_size here
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
        "image_height": details.image_height,
        "image_width": details.image_width,
        "kernel_size": kernel_size,  # Add this line to ensure kernel_size is in analysis_params
    }
    
    # Set up and run analysis for each technique (Speckle and NL-Means)
    for technique in ["speckle", "nlm"]:
        tab_index = 0 if technique == "speckle" else 1
        with st.session_state.tabs[tab_index]:
            technique_params = get_technique_params(technique, st.session_state.analysis_params)
            st.session_state[f"{technique}_params"] = technique_params

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