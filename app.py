
# import streamlit_nested_layout  # type: ignore  # noqa: F401
import streamlit as st

from ui_components import (setup_sidebar)
from visualization import prepare_comparison_images
from image_processing import calculate_processing_details, handle_image_comparison
from viz import process_and_display_images, handle_animation

import logging

PAGE_CONFIG = {
    "page_title": "Speckle Contrast Visualization",
    "layout": "wide",
    "page_icon": "favicon.png",
    "initial_sidebar_state": "expanded"
}

def setup_page_config() -> None:
    """Set up the page configuration."""
    st.set_page_config(**PAGE_CONFIG)
    st.logo("media/logo.png")

def main() -> None:
    """Main application flow."""
    setup_page_config()
    
    try:
        sidebar_params = setup_sidebar()
        tabs = st.tabs(["Speckle", "NL-Means", "Image Comparison"])
        
        # Incorporate create_analysis_params logic here
        details = calculate_processing_details(
            sidebar_params['image_np'], 
            sidebar_params['kernel_size'], 
            None if sidebar_params['show_full_processed'] else sidebar_params['max_pixels']
        )
        
        analysis_params = {
            "image_np": sidebar_params['image_np'],
            "kernel_size": sidebar_params['kernel_size'],
            "search_window_size": sidebar_params['search_window_size'],
            "filter_strength": sidebar_params['filter_strength'],
            "cmap": sidebar_params['cmap'],
            "max_pixels": details['pixels_to_process'],
            "height": details['height'],
            "width": details['width'],
            "show_full_processed": sidebar_params['show_full_processed'],
            "technique": sidebar_params['technique']
        }
        
        st.session_state.tabs = tabs 
        st.session_state.sidebar_params = sidebar_params
        st.session_state.analysis_params = analysis_params
        
        handle_animation(sidebar_params, analysis_params)
        
        process_and_display_images(analysis_params)
        
        comparison_images = prepare_comparison_images()
        handle_image_comparison(tabs[2], analysis_params['cmap'], comparison_images)
        
    
    except Exception as e:
        st.error("An error occurred during processing.")
        st.error(f"Error details: {str(e)}")
        st.exception(e)  # This will display the full traceback
        st.info("Please check your inputs and try again. If the problem persists, contact support.")
        
        # Improved error logging
        logging.exception("An error occurred in main()")
        
        # You can also add more context to the log if needed
        logging.error(f"Error occurred with sidebar_params: {sidebar_params}")
        logging.error(f"Error occurred with analysis_params: {analysis_params}")

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.ERROR,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        filename='app_error.log'
    )
    main()