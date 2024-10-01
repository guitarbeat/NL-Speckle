"""
Main entry point for the Speckle Contrast Visualization Streamlit application.
"""

import streamlit as st
import json
from src.sidebar import setup_ui
import numpy as np
from src.images import apply_processing, create_technique_config, display_filters, get_zoomed_image_section
from src.draw.formula import display_formula_details
import src.session_state as session_state
import matplotlib.pyplot as plt
from src.draw.overlay import add_overlays

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

def process_technique(technique):
    """
    Apply the specified image processing technique and store the result in session state.
    
    Args:
        technique (str): The name of the technique to apply (e.g., 'nlm', 'speckle').
    
    Returns:
        dict or None: The result of the processing if successful, else None.
    """
    image = session_state.get_image_array()
    params = session_state.get_technique_params(technique)
    pixel_percentage = session_state.get_session_state("desired_percentage", 100)

    if image is None or image.size == 0 or params is None:
        st.error(
            f"{'No image data found' if image is None or image.size == 0 else 'No parameters found for ' + technique}. Please check your input."
        )
        return None

    result = apply_processing(image.astype(np.float32), technique, params, pixel_percentage)

    if result is not None:
        session_state.set_technique_result(technique, result)

    return result

def setup_debug_mode():
    """
    Set up debug mode in the sidebar to display session state information.
    """
    debug_mode = st.sidebar.checkbox("Debug Mode")
    if debug_mode:
        st.sidebar.subheader("Session State")
        session_state_str = json.dumps({k: str(v) for k, v in st.session_state.items()}, indent=2)
        st.sidebar.code(session_state_str, language="json")

def process_technique_tab(technique, tab):
    """
    Process and display results for a given technique within its designated tab.
    
    Args:
        technique (str): The name of the technique (e.g., 'nlm', 'speckle').
        tab: The Streamlit tab object where the output will be displayed.
    """
    with tab:
        if session_state.get_session_state('image') is None:
            st.warning("Please load an image before processing.")
            return

        filter_options = session_state.get_filter_options(technique)
        if f'{technique}_filters' not in st.session_state:
            st.session_state[f'{technique}_filters'] = session_state.get_filter_selection(technique)

        st.write(f"{technique.upper()} Filters:")
        selected_filters = st.multiselect(
            f"Select {technique.upper()} filters to display",
            options=filter_options,
            default=st.session_state[f'{technique}_filters'],
            key=f"{technique}_filter_selection"
        )

        if selected_filters != st.session_state[f'{technique}_filters']:
            st.session_state[f'{technique}_filters'] = selected_filters
            session_state.update_filter_selection(technique, selected_filters)

        result = session_state.get_technique_result(technique)
        if result is None:
            result = process_technique(technique)
            session_state.set_technique_result(technique, result)

        config = create_technique_config(technique, tab)
        if config is not None:
            display_data = display_filters(config)
            for plot_config, placeholder, zoomed in display_data:
                display_image(plot_config, placeholder, zoomed)
            display_formula_details(config)

def display_image(plot_config, placeholder, zoomed=False):
    """
    Display either the main image or a zoomed section based on the configuration.
    
    Args:
        plot_config (dict): Configuration for plotting the image.
        placeholder: Streamlit placeholder to render the image.
        zoomed (bool): Whether to display a zoomed-in section of the image.
    """
    zoom_data, _ = (
        (plot_config["filter_data"], "")
        if not zoomed
        else get_zoomed_image_section(
            plot_config["filter_data"],
            *session_state.get_last_processed_pixel(),
            session_state.kernel_size(),
        )
    )

    fig, ax = plt.subplots(1, 1, figsize=(4 if zoomed else 8, 4 if zoomed else 8))
    ax.set_title(f"{'Zoomed ' if zoomed else ''}{plot_config['title']}")
    ax.imshow(
        zoom_data,
        vmin=plot_config["vmin"],
        vmax=plot_config["vmax"],
        cmap=session_state.get_color_map(),
    )
    ax.axis("off")
    add_overlays(ax, zoom_data, plot_config)
    fig.tight_layout(pad=2)

    (
        placeholder
        if not zoomed
        else placeholder.get(f"zoomed_{plot_config['title'].lower().replace(' ', '_')}")
    ).pyplot(fig)
    plt.close(fig)

def main():
    """
    Main function to set up and run the Streamlit application.
    """
    try:
        st.set_page_config(**APP_CONFIG)
        session_state.initialize_session_state()
        setup_ui()
        setup_debug_mode()

        tab_speckle, tab_nlm = st.tabs(["Speckle", "NL-Means"])
        
        for technique, tab in zip(['speckle', 'nlm'], [tab_speckle, tab_nlm]):
            process_technique_tab(technique, tab)

    except Exception as e:
        st.error(f"An error occurred: {e}")
        st.exception(e)

if __name__ == "__main__":
    main()