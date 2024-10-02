## Image Rendering ##
from typing import Dict, Any, List, Tuple
import src.session_state as session_state
import streamlit as st
import numpy as np
from src.images import create_shared_config

def create_technique_config(
    technique: str, tab: st.delta_generator.DeltaGenerator
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

    config = {
        **shared_config,
        "results": result_image,
        "ui_placeholders": create_ui_placeholders(tab, selected_filters),
        "selected_filters": selected_filters,
        "kernel": {
            "size": shared_config["kernel_size"],
            "outline_color": "red",
            "outline_width": 1,
            "grid_line_color": "red",
            "grid_line_style": ":",
            "grid_line_width": 1,
            "center_pixel_color": "green",
            "center_pixel_opacity": 0.5,
        },
        "search_window": {
            "outline_color": "blue",
            "outline_width": 2,
            "size": shared_config["search_window_size"],
        },
        "pixel_value": {
            "text_color": "white",
            "font_size": 8,
        },
        "zoom": False,
        "processable_area": shared_config["processable_area"],
        "last_processed_pixel": result_image.get("last_processed_pixel", (0, 0)),
    }
    return config


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
    data = [(plot_config, config["ui_placeholders"][filter_key], False)]

    if config["show_per_pixel_processing"]:
        data.append((plot_config, config["ui_placeholders"], True))

    return data


def prepare_filter_data(filter_data: np.ndarray) -> np.ndarray:
    """Ensure filter data is 2D."""
    return (
        filter_data.reshape(session_state.get_image_array().shape)
        if filter_data.ndim == 1
        else filter_data
    )


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
    tab: st.delta_generator.DeltaGenerator, selected_filters: List[str]
) -> Dict[str, Any]:
    """Create UI placeholders for filters and formula."""
    with tab:
        placeholders = {"formula": st.empty()}
        columns = st.columns(max(1, len(selected_filters)))

        for i, filter_name in enumerate(selected_filters):
            filter_key = filter_name.lower().replace(" ", "_")
            placeholders[filter_key] = columns[i].empty()
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
    top, bottom = (
        max(0, center_y_coord - half_zoom),
        min(image.shape[0], center_y_coord + half_zoom + 1),
    )
    left, right = (
        max(0, center_x_coord - half_zoom),
        min(image.shape[1], center_x_coord + half_zoom + 1),
    )

    return image[top:bottom, left:right], (center_x_coord - left, center_y_coord - top)
