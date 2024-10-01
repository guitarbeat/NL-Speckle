"""
This module handles image visualization for the Streamlit application.
"""

from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from src.draw.overlay import add_overlays
from src.session_state import (
    kernel_size,
    get_color_map, get_image_array, get_show_per_pixel_processing, get_technique_result, get_value,
    set_value,
    get_filter_selection, DEFAULT_SPECKLE_VIEW, DEFAULT_NLM_VIEW, update_filter_selection,
    get_last_processed_pixel,
)



def create_technique_config(technique: str, tab: st.delta_generator.DeltaGenerator) -> Optional[Dict[str, Any]]:
    st.write("Debug: images.py - create_technique_config")
    st.write(f"Input: technique={technique}")
    
    result_image = get_technique_result(technique)
    if result_image is None:
        st.error(f"No results available for {technique}.")
        return None

    selected_filters = st.session_state.get(f'{technique}_filters', get_filter_selection(technique))
    if not selected_filters:
        selected_filters = DEFAULT_SPECKLE_VIEW if technique == 'speckle' else DEFAULT_NLM_VIEW
        st.session_state[f'{technique}_filters'] = selected_filters
        update_filter_selection(technique, selected_filters)

    ui_placeholders = create_ui_placeholders(tab, technique, selected_filters)

    config = {
        'technique': technique,
        'results': result_image,
        'ui_placeholders': ui_placeholders,
        'last_processed_pixel': get_last_processed_pixel(),
        'selected_filters': selected_filters,
        'kernel': {'size': kernel_size()},
        'show_kernel': get_show_per_pixel_processing(),
        'zoom': False,
        'show_per_pixel_processing': get_show_per_pixel_processing(),
    }
    
    st.write(f"Debug: Config created: {config}")
    return config

def display_filters(config: Dict[str, Any]) -> None:
    st.write("Debug: images.py - display_filters")
    technique = config['technique']
    filter_options = config['results'].get('filter_data', {})
    filter_options['Original Image'] = get_image_array()

    selected_filters = config['selected_filters']
    
    st.write(f"Debug: Selected filters for {technique}: {selected_filters}")
    st.write(f"Debug: Available filter options: {list(filter_options.keys())}")

    for filter_name in selected_filters:
        if filter_name in filter_options:
            st.write(f"Debug: Processing filter: {filter_name}")
            filter_data = filter_options[filter_name]
            # Ensure filter_data is 2D
            filter_data = filter_data.reshape(get_image_array().shape) if filter_data.ndim == 1 else filter_data
            
            plot_key = filter_name.lower().replace(' ', '_')
            if plot_key in config['ui_placeholders']:
                st.write(f"Plotting {filter_name}")
                plot_config = {
                    **config,
                    "vmin": None if filter_name == "Original Image" else np.min(filter_data),
                    "vmax": None if filter_name == "Original Image" else np.max(filter_data),
                    "zoom": False,
                    "show_kernel": get_show_per_pixel_processing(),
                    "show_per_pixel_processing": get_show_per_pixel_processing(),
                    "title": filter_name
                }

                # Main image
                fig, ax = plt.subplots(1, 1, figsize=(8, 8))
                ax.imshow(filter_data, vmin=plot_config['vmin'], vmax=plot_config['vmax'], cmap=get_color_map())
                ax.set_title(plot_config['title'])
                ax.axis("off")
                add_overlays(ax, filter_data, plot_config)
                fig.tight_layout(pad=2)
                config['ui_placeholders'][plot_key].pyplot(fig)
                plt.close(fig)

                # Zoomed-in image
                if get_show_per_pixel_processing():
                    center_y, center_x = get_last_processed_pixel()
                    zoomed_data, new_center_x, new_center_y = get_zoomed_image_section(filter_data, center_x, center_y, kernel_size())
                    zoomed_fig, zoomed_ax = plt.subplots(1, 1, figsize=(4, 4))
                    zoomed_ax.imshow(zoomed_data, vmin=plot_config['vmin'], vmax=plot_config['vmax'], cmap=get_color_map())
                    zoomed_ax.set_title(f"Zoomed {filter_name}")
                    zoomed_ax.axis("off")
                    
                    # Update this block to draw overlays on zoomed image
                    zoomed_config = {
                        **plot_config,
                        "zoom": True,
                        "last_processed_pixel": (new_center_y, new_center_x),
                        "kernel": {"size": kernel_size()},
                    }
                    add_overlays(zoomed_ax, zoomed_data, zoomed_config)
                    
                    zoom_placeholder = config['ui_placeholders'].get(f"zoomed_{plot_key}")
                    if zoom_placeholder:
                        zoom_placeholder.pyplot(zoomed_fig)
                    plt.close(zoomed_fig)

        else:
            st.warning(f"Data for {filter_name} is not available.")

    st.write("Debug: Filter display complete")

def create_ui_placeholders(tab: st.delta_generator.DeltaGenerator, technique: str, selected_filters: List[str]) -> Dict[str, st.delta_generator.DeltaGenerator]:
    """Create UI placeholders for filters and formula."""
    with tab:
        ui_placeholders = {"formula": st.empty()}
        
        num_columns = max(1, len(selected_filters))
        columns = st.columns(num_columns)
        
        for i, filter_name in enumerate(selected_filters):
            filter_key = filter_name.lower().replace(" ", "_")
            ui_placeholders[filter_key] = columns[i].empty()
            if get_show_per_pixel_processing():
                ui_placeholders[f"zoomed_{filter_key}"] = columns[i].expander(f"Zoomed-in {filter_name}", expanded=False).empty()
        if get_show_per_pixel_processing():
            ui_placeholders["zoomed_kernel"] = st.empty()

    return ui_placeholders

def get_zoomed_image_section(image: np.ndarray, center_x: int, center_y: int, kernel_size: int):
    st.write("Debug: images.py - get_zoomed_image_section")
    st.write(f"Input: center_x={center_x}, center_y={center_y}, kernel_size={kernel_size}")
    
    half_zoom = kernel_size // 2
    top, bottom = max(0, center_y - half_zoom), min(image.shape[0], center_y + half_zoom + 1)
    left, right = max(0, center_x - half_zoom), min(image.shape[1], center_x + half_zoom + 1)
    
    st.write(f"Debug: Zoomed section - top={top}, bottom={bottom}, left={left}, right={right}")
    return image[top:bottom, left:right], center_x - left, center_y - top

def prepare_visualization():
    """Prepare for visualization by updating session state."""
    set_value('kernel_size', get_value('kernel_size_slider'))
    set_value('show_per_pixel', get_value('show_per_pixel')) 
    set_value('color_map', get_value('color_map_select'))
    # Add more updates for other sidebar inputs as needed