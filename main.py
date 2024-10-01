"""
This module serves as the main entry point for the Streamlit application.
It imports necessary utilities and plotting functions for image comparison.
"""

import streamlit as st
import json
from src.sidebar import setup_ui

from src.images import prepare_visualization, create_technique_config, display_filters

from src.draw.formula import display_formula_details

from src.draw.compare import handle_image_comparison

from src.session_state import (
    initialize_session_state,
    set_technique_result,
    get_technique_result,
    get_session_state,
    setup_tabs,
        get_filter_options,
    update_filter_selection,
    get_filter_selection
    )

from src.utils import process_technique

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
        # Set up Streamlit configuration (this should be the first Streamlit command)
        st.set_page_config(**APP_CONFIG)
        
        # Initialize session state
        initialize_session_state()
        
        setup_ui()
        
        # Debug mode setup
        debug_mode = st.sidebar.checkbox("Debug Mode")
        if debug_mode:
            st.sidebar.subheader("Session State")
            session_state_str = json.dumps({k: str(v) for k, v in st.session_state.items()}, indent=2)
            st.sidebar.code(session_state_str, language="json")
        
        setup_tabs()

        tab_speckle, tab_nlm, tab_compare = st.tabs(["Speckle", "NL-Means", "Compare"])

        for technique, tab in zip(['speckle', 'nlm'], [tab_speckle, tab_nlm]):
            with tab:
                if get_session_state('image') is None:
                    st.warning("Please load an image before processing.")
                    continue
                
                # Filter selection
                filter_options = get_filter_options(technique)
                if f'{technique}_filters' not in st.session_state:
                    st.session_state[f'{technique}_filters'] = get_filter_selection(technique)
                
                st.write(f"{technique.upper()} Filters:")
                selected_filters = st.multiselect(
                    f"Select {technique.upper()} filters to display",
                    options=filter_options,
                    default=st.session_state[f'{technique}_filters'],
                    key=f"{technique}_filter_selection"
                )
                
                if selected_filters != st.session_state[f'{technique}_filters']:
                    st.session_state[f'{technique}_filters'] = selected_filters
                    update_filter_selection(technique, selected_filters)
                
                # Processing
                if get_technique_result(technique) is None:
                    with st.spinner(f"Processing {technique}..."):
                        result = process_technique(technique)
                    set_technique_result(technique, result)
                
                # Display results
                config = create_technique_config(technique, tab)
                if config is not None:
                    display_filters(config)
                    display_formula_details(config)

        # Handle image comparison in the third tab
        with tab_compare:
            speckle_result = get_technique_result('speckle')
            nlm_result = get_technique_result('nlm')
            
            if speckle_result is None and nlm_result is None:
                st.warning("No processed images available for comparison. Please process at least one image using Speckle or NL-Means techniques.")
            else:
                handle_image_comparison(tab_compare)

        # Prepare visualization settings
        prepare_visualization()

    except Exception as e:
        st.error(f"An error occurred: {e}")
        st.exception(e)

if __name__ == "__main__":
    main()