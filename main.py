import streamlit as st
import streamlit_nested_layout  # noqa: F401
from frontend.plotting import setup_and_run_analysis_techniques
from utils import calculate_processing_details
from frontend.ui_elements import (handle_image_comparison, prepare_comparison_images,
                                  setup_sidebar, get_technique_params)
import hashlib
import time
# Add this import
# from memory_profiler import profile

PAGE_CONFIG = {
    "page_title": "Speckle Contrast Visualization",
    "layout": "wide",
    "page_icon": "favicon.png",
    "initial_sidebar_state": "expanded"
}

# Modify the main function to include memory profiling
# @profile
def main():
    st.set_page_config(**PAGE_CONFIG)
    st.logo("media/logo.png")  # Changed from st.logo to st.image
    
    if 'session_id' not in st.session_state:
        st.session_state.session_id = hashlib.md5(str(time.time()).encode()).hexdigest()

    try:
        sidebar_params = setup_sidebar()
        if sidebar_params is None:
            st.stop()

   

        tabs = st.tabs(["Speckle", "NL-Means", "Image Comparison"])
        
        processing_params = {
            'kernel_size': 7,  # Default value, adjust as needed
        }

        details = calculate_processing_details(
            sidebar_params['image_np'], 
            processing_params['kernel_size'], 
            None if sidebar_params['show_per_pixel'] else sidebar_params['pixels_to_process']
        )
        
        st.session_state.tabs = tabs 
        st.session_state.sidebar_params = sidebar_params
        
        st.session_state.analysis_params = {
            "image_np": sidebar_params['image_np'],
            "show_per_pixel": sidebar_params['show_per_pixel'],
            "max_pixels": sidebar_params['pixels_to_process'],
            "image_height": details.image_height,
            "image_width": details.image_width,
            **processing_params
        }
        
        for technique in ["speckle", "nlm"]:
            with st.session_state.tabs[0 if technique == "speckle" else 1]:
                technique_params = get_technique_params(technique, st.session_state.analysis_params)
                st.session_state[f"{technique}_params"] = technique_params
                
        setup_and_run_analysis_techniques(st.session_state.analysis_params)
        comparison_images = prepare_comparison_images()
        handle_image_comparison(tabs[2], st.session_state.cmap, comparison_images)


    except Exception as e:
        st.error("An unexpected error occurred. Please try reloading the application.")
        st.exception(e)

if __name__ == "__main__":
    main()