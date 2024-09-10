import streamlit as st
import streamlit_nested_layout # type: ignore  # noqa: F401


from image_processing import (
    prepare_comparison_images, display_processing_statistics, 
    create_analysis_params, handle_animation_if_needed,)

from ui_components import create_tabs, setup_sidebar, process_and_display_images, handle_image_comparison

from constants import PAGE_CONFIG

# Main Application Flow
def main():
    setup_page_config()
    sidebar_params = setup_sidebar()
    tabs = create_tabs()
    
    analysis_params = create_analysis_params(sidebar_params)
    store_session_state(tabs, sidebar_params, analysis_params)
    
    handle_animation_if_needed(sidebar_params, analysis_params)
    
    process_and_display_images(analysis_params)
    
    comparison_images = prepare_comparison_images()
    handle_image_comparison(tabs[2], analysis_params['cmap'], comparison_images)
    
    display_processing_statistics(analysis_params)

# Page Setup and Configuration
def setup_page_config():
    st.set_page_config(**PAGE_CONFIG)
    st.logo("media/logo.png")




def store_session_state(tabs, sidebar_params, analysis_params):
    st.session_state.tabs = tabs 
    st.session_state.sidebar_params = sidebar_params
    st.session_state.analysis_params = analysis_params


if __name__ == "__main__":
    main()
