from typing import Dict, Any, List, Tuple
import src.session_state as session_state
import streamlit as st
import numpy as np
from src.images import create_shared_config
from src.session_state import (
    DEFAULT_KERNEL_OUTLINE_COLOR,
    DEFAULT_KERNEL_CENTER_PIXEL_COLOR,
    DEFAULT_SEARCH_WINDOW_COLOR,
    DEFAULT_PIXEL_TEXT_COLOR,
    DEFAULT_PIXEL_FONT_SIZE,
)

def create_technique_config(
    technique: str, tab: Any
) -> Dict[str, Any]:
    """Create a configuration dictionary for the specified denoising technique."""
    result_image = session_state.get_technique_result(technique)
    if result_image is None:
        st.error(f"No results available for {technique}.")
        return None

    params = session_state.get_technique_params(technique)
    shared_config = create_shared_config(technique, params)

    selected_filters = session_state.get_session_state(
        f"{technique}_filters", session_state.get_filter_selection(technique)
    )

    # Get the last processed pixel, or use a default value if it's not available
    last_processed_pixel = result_image.get("last_processed_pixel")
    if last_processed_pixel is None:
        # Use the center of the image as a default
        image_array = session_state.get_image_array()
        if image_array is not None:
            height, width = image_array.shape[:2]
            last_processed_pixel = (height // 2, width // 2)
        else:
            last_processed_pixel = (0, 0)  # Fallback default

    return {
        **shared_config,
        "results": result_image,
        "ui_placeholders": create_ui_placeholders(tab, selected_filters),
        "selected_filters": selected_filters,
        "kernel": create_kernel_config(shared_config),
        "search_window": create_search_window_config(shared_config),
        "pixel_value": create_pixel_value_config(),
        "zoom": False,
        "processable_area": shared_config.get("processable_area"),
        "last_processed_pixel": last_processed_pixel,
    }


def create_kernel_config(shared_config: Dict[str, Any]) -> Dict[str, Any]:
    """Create the kernel configuration."""
    return {
        "size": shared_config.get("kernel_size"),
        "outline_color": DEFAULT_KERNEL_OUTLINE_COLOR,
        "outline_width": 1,
        "grid_line_color": DEFAULT_KERNEL_OUTLINE_COLOR,
        "grid_line_style": ":",
        "grid_line_width": 1,
        "center_pixel_color": DEFAULT_KERNEL_CENTER_PIXEL_COLOR,
        "center_pixel_opacity": 0.5,
    }


def create_search_window_config(shared_config: Dict[str, Any]) -> Dict[str, Any]:
    """Create the search window configuration."""
    return {
        "outline_color": DEFAULT_SEARCH_WINDOW_COLOR,
        "outline_width": 2,
        "size": shared_config.get("search_window_size"),
    }


def create_pixel_value_config() -> Dict[str, Any]:
    """Create the pixel value configuration."""
    return {
        "text_color": DEFAULT_PIXEL_TEXT_COLOR,
        "font_size": DEFAULT_PIXEL_FONT_SIZE,
    }


def display_filters(config: Dict[str, Any]) -> List[Tuple[Dict[str, Any], Any, bool]]:
    """Prepare filters data based on the provided configuration."""
    filter_options = {
        **config["results"].get("filter_data", {}),
        "Original Image": session_state.get_image_array(),
    }

    display_data = []
    for filter_name in config["selected_filters"]:
        if filter_name in filter_options:
            filter_data = prepare_filter_data(filter_options[filter_name])
            plot_config = create_plot_config(config, filter_name, filter_data)
            display_data.extend(create_display_data(config, plot_config, filter_name))
        else:
            st.warning(f"Data for {filter_name} is not available.")

    return display_data


def create_display_data(
    config: Dict[str, Any], plot_config: Dict[str, Any], filter_name: str
) -> List[Tuple[Dict[str, Any], Any, bool]]:
    """Create display data for a single filter."""
    filter_key = filter_name.lower().replace(" ", "_")
    placeholders = config["ui_placeholders"]

    data = [(plot_config, placeholders[filter_key], False)]

    if config.get("show_per_pixel_processing", False):
        data.append((plot_config, placeholders, True))

    return data


def prepare_filter_data(filter_data: np.ndarray) -> np.ndarray:
    """Ensure filter data is 2D."""
    image_shape = session_state.get_image_array().shape
    return filter_data.reshape(image_shape) if filter_data.ndim == 1 else filter_data


def create_plot_config(
    config: Dict[str, Any], filter_name: str, filter_data: np.ndarray
) -> Dict[str, Any]:
    """Create a plot configuration dictionary."""
    return {
        **config,
        "filter_data": filter_data,
        "vmin": np.min(filter_data),
        "vmax": np.max(filter_data),
        "title": filter_name,
    }


def create_ui_placeholders(
    tab: Any, selected_filters: List[str]
) -> Dict[str, Any]:
    """Create UI placeholders for filters and formula display."""
    with tab:
        placeholders = {"formula": st.empty()}
        columns = st.columns(len(selected_filters) or 1)

        for i, filter_name in enumerate(selected_filters):
            filter_key = filter_name.lower().replace(" ", "_")
            placeholders[filter_key] = columns[i].empty()
            
            # If "show_per_pixel" is enabled, add a zoomed-in placeholder
            if session_state.get_session_state("show_per_pixel", False):
                placeholders[f"zoomed_{filter_key}"] = (
                    columns[i]
                    .expander(f"Zoomed-in {filter_name}", expanded=False)
                    .empty()
                )

    return placeholders


def get_zoomed_image_section(
    image: np.ndarray, center_x_coord: int, center_y_coord: int, zoom_size: int
) -> Tuple[np.ndarray, Tuple[int, int]]:
    """Get a zoomed-in section of the image centered at the specified pixel."""
    half_zoom = zoom_size // 2
    
    # Ensure that the zoomed section stays within the image boundaries
    top = max(0, center_y_coord - half_zoom)
    bottom = min(image.shape[0], center_y_coord + half_zoom + 1)
    left = max(0, center_x_coord - half_zoom)
    right = min(image.shape[1], center_x_coord + half_zoom + 1)

    zoomed_section = image[top:bottom, left:right]
    center_coords_in_zoom = (center_x_coord - left, center_y_coord - top)

    return zoomed_section, center_coords_in_zoom