import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from typing import Any, Dict, List, Tuple, Optional, Callable
from contextlib import contextmanager
import json
from src.draw.overlay import add_overlays
from src.draw.formula import display_formula_details
from session_state import (
    get_filter_selection, get_kernel_size, get_search_window_size,
    get_color_map, get_image_array, set_viz_config,
    get_show_per_pixel_processing, handle_processing_error
)
from src.math.nlm import check_function_arguments

def visualize_image_and_results(viz_params: Dict[str, Any]) -> None:
    filter_options = {**viz_params['results'].get('filter_data', {}), "Original Image": get_image_array()}
    selected_filters = get_filter_selection(viz_params['technique'])

    for filter_name in selected_filters:
        filter_data = filter_options.get(filter_name)
        if filter_data is None:
            handle_processing_error(f"Data for {filter_name} is not available.")
            continue

        filter_data = reshape_filter_data(filter_data, get_image_array().shape)
        
        for plot_type in ["main", "zoomed"]:
            plot_key = f"{'zoomed_' if plot_type == 'zoomed' else ''}{filter_name.lower().replace(' ', '_')}"
            if plot_key in viz_params['ui_placeholders'] and (plot_type != "zoomed" or get_show_per_pixel_processing()):
                plot_config = create_plot_config(viz_params, filter_name, filter_data, plot_type)
                create_and_display_plot(plot_config, filter_data, viz_params['ui_placeholders'][plot_key])

    display_formula_details(viz_params)

def reshape_filter_data(filter_data: np.ndarray, image_shape: Tuple[int, int]) -> np.ndarray:
    return filter_data.reshape(image_shape) if filter_data.ndim == 1 else filter_data

def create_plot_config(viz_params: Dict[str, Any], filter_name: str, filter_data: np.ndarray, plot_type: str) -> Dict[str, Any]:
    config = {
        **viz_params,
        "vmin": None if filter_name == "Original Image" else np.min(filter_data),
        "vmax": None if filter_name == "Original Image" else np.max(filter_data),
        "zoom": (plot_type == "zoomed"),
        "show_kernel": (viz_params['show_per_pixel_processing'] if plot_type == "main" else True),
        "show_per_pixel_processing": (plot_type == "zoomed"),
        "title": f"Zoomed-In {filter_name}" if plot_type == "zoomed" else filter_name
    }
    
    if config['zoom']:
        filter_data, new_center_x, new_center_y = get_zoomed_image_section(filter_data, config)
        config['last_processed_pixel'] = (new_center_x, new_center_y)
    
    return config

def create_and_display_plot(config: Dict[str, Any], filter_data: np.ndarray, placeholder: Any) -> None:
    fig, ax = plt.subplots(1, 1, figsize=config.get('figure_size', (8, 8)))
    ax.imshow(filter_data, vmin=config.get('vmin'), vmax=config.get('vmax'), cmap=get_color_map())
    ax.set_title(config.get('title', ''))
    ax.axis("off")
    add_overlays(ax, filter_data, config)
    fig.tight_layout(pad=2)
    placeholder.pyplot(fig)
    plt.close(fig)

def get_zoomed_image_section(image: np.ndarray, config: Dict[str, Any]) -> Tuple[np.ndarray, int, int]:
    last_processed_pixel = config.get('last_processed_pixel', config.get('processing_end', (0, 0)))
    kernel_size = config['kernel']['size']
    half_zoom = kernel_size // 2
    center_x, center_y = last_processed_pixel
    top, bottom = max(0, center_y - half_zoom), min(image.shape[0], center_y + half_zoom + 1)
    left, right = max(0, center_x - half_zoom), min(image.shape[1], center_x + half_zoom + 1)
    return image[top:bottom, left:right], center_x - left, center_y - top

def create_technique_ui_and_config(technique: str, tab: st.delta_generator.DeltaGenerator, show_per_pixel_processing: bool, result_image: Optional[Dict[str, Any]], image_array: np.ndarray) -> Optional[Dict[str, Any]]:
    with tab:
        selected_filters = get_filter_selection(technique)
        ui_placeholders = create_ui_placeholders(technique, show_per_pixel_processing, selected_filters)

    if result_image is None:
        handle_processing_error(f"No results available for {technique}. Please ensure the processing completed successfully.")
        return None

    kernel_size = get_kernel_size()
    if kernel_size is None:
        handle_processing_error("Kernel size is not set. Please ensure all parameters are properly initialized.")
        return None

    height, width = image_array.shape
    half_kernel = kernel_size // 2

    result_array = get_result_array(result_image, technique, image_array)
    y, x = np.random.randint(half_kernel, height - half_kernel), np.random.randint(half_kernel, width - half_kernel)
    kernel_matrix = get_kernel_matrix(image_array, x, y, half_kernel, kernel_size)

    config = {
        'technique': technique,
        'results': {'filter_data': result_image if isinstance(result_image, dict) else {technique: result_array}},
        'image_array': image_array,
        'ui_placeholders': ui_placeholders,
        'show_per_pixel_processing': show_per_pixel_processing,
        'kernel': {
            'size': kernel_size,
            'center': (y, x),
            'kernel_matrix': kernel_matrix
        },
        'last_processed_pixel': (y, x),
        'original_pixel_value': float(image_array[y, x]),
        'processed_pixel_value': float(result_array[y - half_kernel, x - half_kernel]),
        'search_window': {'size': get_search_window_size()},
    }

    # Check arguments for the appropriate processing function
    processing_function = get_processing_function(technique)
    args = create_processing_args(technique, config)
    try:
        check_function_arguments(processing_function, args, technique)
    except (ValueError, TypeError) as e:
        error_message = (
            f"Error in argument checking for {technique} technique:\n\n"
            f"{str(e)}\n\n"
            f"Config used:\n{json.dumps(config, indent=2, default=str)}\n\n"
            f"Args created:\n{args}"
        )
        handle_processing_error(error_message)
        return None

    set_viz_config(config)
    return config

def get_processing_function(technique: str) -> Callable:
    if technique == 'nlm':
        from src.math.nlm import process_nlm_pixel
        return process_nlm_pixel
    elif technique == 'speckle':
        from src.math.nlm import process_speckle_pixel
        return process_speckle_pixel
    else:
        raise ValueError(f"Unknown technique: {technique}")

def create_processing_args(technique: str, config: Dict[str, Any]) -> Tuple:
    y, x = config['last_processed_pixel']
    image_array = config['image_array']
    height, width = image_array.shape
    kernel_size = config['kernel']['size']
    
    if technique == 'nlm':
        return (y, x, image_array, kernel_size, 
                config['search_window']['size'], config.get('filter_strength', 1.0), 
                config.get('use_full_image', False), height, width)
    elif technique == 'speckle':
        return (y, x, image_array, kernel_size, 
                config['kernel']['center'], height, width, 
                width - kernel_size + 1)
    else:
        raise ValueError(f"Unknown technique: {technique}")

def create_ui_placeholders(technique: str, show_per_pixel_processing: bool, selected_filters: List[str]) -> Dict[str, Any]:
    ui_placeholders = {"formula": st.empty(), "original_image": st.empty()}
    
    columns = st.columns(len(selected_filters))
    for i, filter_name in enumerate(selected_filters):
        filter_key = filter_name.lower().replace(" ", "_")
        ui_placeholders[filter_key] = columns[i].empty()
        if show_per_pixel_processing:
            ui_placeholders[f"zoomed_{filter_key}"] = columns[i].expander(f"Zoomed-in {filter_name}", expanded=False).empty()

    if show_per_pixel_processing:
        ui_placeholders["zoomed_kernel"] = st.empty()

    return ui_placeholders

def get_result_array(result_image: Dict[str, Any], technique: str, image_array: np.ndarray) -> np.ndarray:
    if isinstance(result_image, dict):
        result_array = result_image.get('filter_data', {}).get(technique, image_array)
    else:
        result_array = result_image

    if result_array.ndim == 1:
        result_array = result_array.reshape(image_array.shape)

    return result_array

@contextmanager 
def create_processing_status():
    with st.status("Processing image...", expanded=True) as status:
        progress_bar = status.progress(0)
        yield status, progress_bar

def get_kernel_matrix(image_array: np.ndarray, end_x: int, end_y: int, half_kernel: int, kernel_size: int) -> np.ndarray:
    height, width = image_array.shape
    kernel_matrix = image_array[
        max(0, end_y - half_kernel):min(height, end_y + half_kernel + 1),
        max(0, end_x - half_kernel):min(width, end_x + half_kernel + 1)
    ].copy()

    if kernel_matrix.shape != (kernel_size, kernel_size):
        kernel_matrix = np.pad(
            kernel_matrix,
            (
                (max(0, half_kernel - end_y), max(0, end_y + half_kernel + 1 - height)),
                (max(0, half_kernel - end_x), max(0, end_x + half_kernel + 1 - width))
            ),
            mode="constant", constant_values=0
        )

    return np.array(kernel_matrix)