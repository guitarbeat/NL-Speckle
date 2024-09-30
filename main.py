"""
This module serves as the main entry point for the Streamlit application.
It imports necessary utilities and plotting functions for image comparison.
"""

import streamlit as st
from src.sidebar import setup_ui
from src.images import create_technique_ui_and_config, visualize_image_and_results
from src.draw.compare import handle_image_comparison
from session_state import (
    initialize_session_state, get_filter_options,
    update_filter_selection, get_filter_selection, set_technique_result,
    get_technique_result
)
from src.math.nlm import process_technique
import json
import logging
import sys

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
file_handler = logging.FileHandler('app.log')
file_handler.setLevel(logging.INFO)
logging.getLogger().addHandler(file_handler)

# Capture stdout and stderr
class StreamToLogger:
    def __init__(self, logger, log_level=logging.INFO):
        self.logger = logger
        self.log_level = log_level
        self.linebuf = ''

    def write(self, buf):
        for line in buf.rstrip().splitlines():
            self.logger.log(self.log_level, line.rstrip())

    def flush(self):
        pass

sys.stdout = StreamToLogger(logging.getLogger('STDOUT'), logging.INFO)
sys.stderr = StreamToLogger(logging.getLogger('STDERR'), logging.ERROR)

# App Configuration
APP_CONFIG = {
    "page_title": "Speckle Contrast Visualization",
    "layout": "wide",
    "page_icon": "favicon.png",
    "initial_sidebar_state": "expanded",
}

# Color Maps 
AVAILABLE_COLOR_MAPS = [
    "viridis_r",
    "viridis",
    "gray",
    "plasma",
    "inferno", 
    "magma",
    "pink",
    "hot",
    "cool",
    "YlOrRd",
]

# Preloaded Images
PRELOADED_IMAGE_PATHS = {
    "image50.png": "media/image50.png",
    "spatial.tif": "media/spatial.tif",
    "logo.jpg": "media/logo.jpg",
}

def main():
    """Main function to set up and run the Streamlit application."""
    try:

        # Initialize session state
        initialize_session_state()

        # Set up Streamlit configuration
        st.set_page_config(**APP_CONFIG)
        st.logo("media/logo.png")
        
   
        
        # Set up UI components
        setup_ui()
        
        # Add debug checkbox
        debug_mode = st.sidebar.checkbox("Debug Mode")
        
        if debug_mode:
            st.sidebar.subheader("Session State")
            # Convert session state to a pretty-printed JSON string
            session_state_str = json.dumps(
                {k: str(v) for k, v in st.session_state.items()},
                indent=2
            )
            st.sidebar.code(session_state_str, language="json")
        
        # Set up tabs
        tabs = st.tabs(["Speckle", "NL-Means", "Image Comparison"])
        st.session_state.tabs = tabs
    except Exception as e:
        st.error(f"An error occurred during initial setup: {e}")
        st.exception(e)
        return

    for technique, tab in zip(st.session_state.techniques, tabs[:2]):
        with tab:
            try:
                # Move the filter selection code inside the tab context
                filter_options = get_filter_options(technique)
                current_selection = get_filter_selection(technique)

                new_selection = st.multiselect(
                    f"Select views to display for {technique.upper()}",
                    filter_options,
                    default=current_selection,
                    key=f"{technique}_filter_selection_key"
                )

                if new_selection != current_selection:
                    update_filter_selection(technique, new_selection)

                show_per_pixel_processing = st.session_state.show_per_pixel
                
                if get_technique_result(technique) is None:
                    with st.spinner(f"Processing {technique}..."):
                        result = process_technique(technique)
                    set_technique_result(technique, result)
                
                viz_config = create_technique_ui_and_config(
                    technique, 
                    tab, 
                    show_per_pixel_processing, 
                    get_technique_result(technique),
                    st.session_state.image_array
                )
                
                if viz_config:
                    st.session_state.viz_config = viz_config
                    visualize_image_and_results(st.session_state.viz_config)
                
            except Exception as e:
                st.error(f"Error processing and visualizing {technique}: {e}")
                st.exception(e)

    # Handle image comparison in the third tab
    with tabs[2]:
        try:
            handle_image_comparison(tabs[2])  
        except Exception as e:
            st.error(f"Error in image comparison: {e}")
            st.exception(e)

if __name__ == "__main__":
    main()