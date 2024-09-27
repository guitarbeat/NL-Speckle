"""
This module provides plotting functionalities using Matplotlib and Streamlit.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from src.formula import display_analysis_formula
from src.nlm import NLMResult
from src.speckle import SpeckleResult
from src.overlay import KernelConfig, add_overlays, VisualizationConfig, SearchWindowConfig

# Constants for Image Visualization
SPECKLE_CONTRAST = "Speckle Contrast"
ORIGINAL_IMAGE = "Original Image"
NON_LOCAL_MEANS = "Non-Local Means"

DEFAULT_SPECKLE_VIEW = [SPECKLE_CONTRAST, ORIGINAL_IMAGE]
DEFAULT_NLM_VIEW = [NON_LOCAL_MEANS, ORIGINAL_IMAGE]

@dataclass
class PixelCoordinates:
    x: int
    y: int

@dataclass
class ImageArray:
    data: np.ndarray

def generate_plot_key(filter_name: str, plot_type: str) -> str:
    base_key = filter_name.lower().replace(" ", "_")
    return f"zoomed_{base_key}" if plot_type == "zoomed" else base_key

def create_process_params(
    analysis_params: Dict[str, Any], technique: str, technique_params: Dict[str, Any]
) -> Dict[str, Any]:
    common_params = {
        "kernel_size": st.session_state.get("kernel_size", 3),
        "pixels_to_process": analysis_params.get("pixels_to_process", 0),
        "total_pixels": analysis_params.get("total_pixels", 0),
        "show_per_pixel_processing": analysis_params.get("show_per_pixel_processing", False),
    }

    if technique == "nlm":
        common_params.update({
            "search_window_size": analysis_params.get("search_window_size"),
            "filter_strength": analysis_params.get("filter_strength"),
        })

    return {
        "image_array": analysis_params.get("image_array", np.array([])),
        "technique": technique,
        "analysis_params": {**technique_params, **common_params},
        "show_per_pixel_processing": analysis_params.get("show_per_pixel_processing", False),
    }

def visualize_filter_and_zoomed(
    filter_name: str, filter_data: np.ndarray, viz_config: VisualizationConfig
):
    for plot_type in ["main", "zoomed"]:
        plot_key = generate_plot_key(filter_name, plot_type)

        if plot_key not in viz_config.ui_placeholders or (
            plot_type == "zoomed" and not viz_config.show_per_pixel_processing
        ):
            continue

        config = update_visualization_config(viz_config, filter_data, filter_name, plot_type)
        config.title = f"Zoomed-In {filter_name}" if plot_type == "zoomed" else filter_name

        visualize_image(filter_data, viz_config.ui_placeholders[plot_key], config=config)

def update_visualization_config(
    viz_config: VisualizationConfig,
    filter_data: np.ndarray,
    filter_name: str,
    plot_type: str,
) -> VisualizationConfig:
    updated_config = {
        "vmin": None if filter_name == "Original Image" else np.min(filter_data),
        "vmax": None if filter_name == "Original Image" else np.max(filter_data),
        "zoom": (plot_type == "zoomed"),
        "show_kernel": (viz_config.show_per_pixel_processing if plot_type == "main" else True),
        "show_per_pixel_processing": (plot_type == "zoomed"),
        "image_array": viz_config.image_array,
        "analysis_params": viz_config.analysis_params,
        "results": viz_config.results,
        "ui_placeholders": viz_config.ui_placeholders,
        "last_processed_pixel": viz_config.last_processed_pixel,
        "original_pixel_value": viz_config.original_pixel_value,
        "technique": viz_config.technique,
        "kernel": viz_config.kernel,
        "search_window": viz_config.search_window,
        "pixel_value": viz_config.pixel_value,
    }

    return VisualizationConfig(**updated_config)

def create_image_plot(plot_image: np.ndarray, config: VisualizationConfig) -> plt.Figure:
    fig, ax = plt.subplots(1, 1, figsize=config.figure_size)
    ax.imshow(plot_image, vmin=config.vmin, vmax=config.vmax, cmap=st.session_state.color_map)
    ax.set_title(config.title)
    ax.axis("off")

    add_overlays(ax, plot_image, config)
    fig.tight_layout(pad=2)
    return fig

def prepare_filter_options_and_parameters(
    results: Any, last_processed_pixel: Tuple[int, int]
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    end_x, end_y = last_processed_pixel
    filter_options = results.get_filter_data()
    specific_params = {
        "kernel_size": results.kernel_size,
        "pixels_processed": results.pixels_processed,
        "total_pixels": results.kernel_size**2,
    }

    for filter_name, filter_data in filter_options.items():
        if isinstance(filter_data, np.ndarray) and filter_data.size > 0:
            specific_params[filter_name.lower().replace(" ", "_")] = filter_data[end_y, end_x]

    if hasattr(results, "filter_strength"):
        specific_params |= {
            "filter_strength": results.filter_strength,
            "search_window_size": results.search_window_size,
        }
    elif hasattr(results, "start_pixel_mean"):
        specific_params |= {
            "start_pixel_mean": results.start_pixel_mean,
            "start_pixel_std_dev": results.start_pixel_std_dev,
            "start_pixel_speckle_contrast": results.start_pixel_speckle_contrast,
        }

    return filter_options, specific_params

def prepare_comparison_images() -> Optional[Dict[str, np.ndarray]]:
    comparison_images = {
        "Unprocessed Image": st.session_state.get("analysis_params", {}).get("image_array", np.array([]))
    }

    for result_key in ["speckle_results", "nlm_results"]:
        results = st.session_state.get(result_key)
        if results is not None:
            comparison_images |= results.get_filter_data()

    return comparison_images if len(comparison_images) > 1 else None

def get_zoomed_image_section(
    image: np.ndarray, center_x: int, center_y: int, kernel_size: int
) -> Tuple[np.ndarray, int, int]:
    half_zoom = kernel_size // 2
    top = max(0, center_y - half_zoom)
    bottom = min(image.shape[0], top + kernel_size)
    left = max(0, center_x - half_zoom)
    right = min(image.shape[1], left + kernel_size)

    zoomed_image = image[top:bottom, left:right]
    new_center_x = center_x - left
    new_center_y = center_y - top

    return zoomed_image, new_center_x, new_center_y

def visualize_image(
    image: np.ndarray, placeholder, *, config: VisualizationConfig
) -> None:
    try:
        if config.zoom:
            try:
                image, new_center_x, new_center_y = get_zoomed_image_section(
                    image,
                    config.last_processed_pixel[0],
                    config.last_processed_pixel[1],
                    config.kernel.size,
                )
                config.last_processed_pixel = (new_center_x, new_center_y)
            except Exception as e:
                placeholder.error(f"An error occurred while zooming the image: {e}. Please check the logs for details.")
                return

        try:
            fig = create_image_plot(image, config)
            placeholder.pyplot(fig)
            plt.close(fig)
        except Exception as e:
            placeholder.error(f"An error occurred while creating the image plot: {e}. Please check the logs for details.")
    except (ValueError, TypeError, KeyError) as e:
        placeholder.error(f"An error occurred while visualizing the image: {e}. Please check the logs for details.")

def get_filter_options(technique: str) -> List[str]:
    if technique == "speckle":
        return ["Original Image"] + SpeckleResult.get_filter_options()
    elif technique == "nlm":
        return ["Original Image"] + NLMResult.get_filter_options()
    else:
        return []

def create_filter_selection(technique: str, filter_options: List[str]) -> List[str]:
    default_selection = DEFAULT_SPECKLE_VIEW if technique == "speckle" else DEFAULT_NLM_VIEW

    selected_filters = st.multiselect(
        "Select views to display",
        filter_options,
        default=default_selection,
        key=f"{technique}_filter_selection",
    )

    st.session_state[f"{technique}_selected_filters"] = selected_filters
    return selected_filters

def create_filter_views(
    selected_filters: List[str],
    ui_placeholders: Dict[str, Any],
    show_per_pixel_processing: bool,
) -> None:
    columns = st.columns(len(selected_filters))

    for i, filter_name in enumerate(selected_filters):
        filter_key = filter_name.lower().replace(" ", "_")
        ui_placeholders[filter_key] = columns[i].empty()

        if show_per_pixel_processing:
            ui_placeholders[f"zoomed_{filter_key}"] = (
                columns[i].expander(f"Zoomed-in {filter_name}", expanded=False).empty()
            )

def create_technique_ui_elements(
    technique: str, tab: Any, show_per_pixel_processing: bool
) -> Dict[str, Any]:
    if not technique or not isinstance(technique, str):
        raise ValueError("Technique must be a non-empty string.")

    with tab:
        ui_placeholders = {"formula": st.empty(), "original_image": st.empty()}

        filter_options = get_filter_options(technique)

        if selected_filters := create_filter_selection(technique, filter_options):
            create_filter_views(selected_filters, ui_placeholders, show_per_pixel_processing)
        else:
            st.warning("No views selected. Please select at least one view to display.")

        if show_per_pixel_processing:
            ui_placeholders["zoomed_kernel"] = st.empty()

    return ui_placeholders

def visualize_analysis_results(viz_params: VisualizationConfig) -> None:
    filter_options, specific_params = prepare_filter_options_and_parameters(
        viz_params.results, viz_params.last_processed_pixel
    )
    filter_options["Original Image"] = viz_params.image_array.data

    selected_filters = st.session_state.get(f"{viz_params.technique}_selected_filters", [])
    for filter_name in selected_filters:
        if filter_name in filter_options:
            filter_data = filter_options[filter_name]
            visualize_filter_and_zoomed(filter_name, filter_data, viz_params)

    last_processed_x, last_processed_y = viz_params.last_processed_pixel
    specific_params.update({
        "Centered at": f"({last_processed_x}, {last_processed_y})",
        "image_height": viz_params.image_array.data.shape[0],
        "image_width": viz_params.image_array.data.shape[1],
        "half_kernel": viz_params.kernel.size // 2,
        "valid_height": viz_params.image_array.data.shape[0] - viz_params.kernel.size + 1,
        "valid_width": viz_params.image_array.data.shape[1] - viz_params.kernel.size + 1,
        "search_window_size": viz_params.search_window.size,
    })

    if viz_params.technique == "nlm":
        specific_params["nlm_value"] = viz_params.results.nonlocal_means[last_processed_y, last_processed_x]
    else:  # speckle
        specific_params.update({
            "mean": viz_params.results.mean_filter[last_processed_y, last_processed_x],
            "std": viz_params.results.std_dev_filter[last_processed_y, last_processed_x],
            "sc": viz_params.results.speckle_contrast_filter[last_processed_y, last_processed_x],
        })

    display_analysis_formula(
        specific_params,
        viz_params.ui_placeholders,
        viz_params.technique,
        last_processed_x,
        last_processed_y,
        viz_params.kernel.size,
        viz_params.kernel.kernel_matrix,
        viz_params.original_pixel_value,
    )

def run_technique(
    technique: str, tab: Any, analysis_params: Dict[str, Any], nl_speckle_result: Any
) -> None:
    technique_params = st.session_state.get(f"{technique}_params", {})
    show_per_pixel_processing = analysis_params.get("show_per_pixel_processing", False)

    ui_placeholders = create_technique_ui_elements(technique, tab, show_per_pixel_processing)
    st.session_state[f"{technique}_placeholders"] = ui_placeholders

    process_params = create_process_params(analysis_params, technique, technique_params)

    try:
        if technique == "nlm":
            results = nl_speckle_result.nlm_result
        elif technique == "speckle":
            results = nl_speckle_result.speckle_result
        else:
            raise ValueError(f"Unknown technique: {technique}")

        st.session_state[f"{technique}_results"] = results

        viz_config = create_visualization_config(
            process_params["image_array"],
            technique,
            analysis_params,
            results,
            ui_placeholders,
            show_per_pixel_processing,
        )

        visualize_analysis_results(viz_config)

    except (ValueError, TypeError, KeyError) as e:
        st.error(f"Error for {technique}: {str(e)}. Please check the logs for details.")

def create_visualization_config(
    image_array: np.ndarray,
    technique: str,
    analysis_params: Dict[str, Any],
    results: Union[NLMResult, SpeckleResult],
    ui_placeholders: Dict[str, Any],
    show_per_pixel_processing: bool,
) -> VisualizationConfig:
    half_kernel = results.kernel_size // 2
    height, width = image_array.shape
    end_x, end_y = results.processing_end_coord

    y_start = max(0, end_y - half_kernel)
    y_end = min(height, end_y + half_kernel + 1)
    x_start = max(0, end_x - half_kernel)
    x_end = min(width, end_x + half_kernel + 1)
    kernel_matrix = image_array[y_start:y_end, x_start:x_end].copy()

    if kernel_matrix.shape != (results.kernel_size, results.kernel_size):
        pad_top = max(0, half_kernel - end_y)
        pad_bottom = max(0, end_y + half_kernel + 1 - height)
        pad_left = max(0, half_kernel - end_x)
        pad_right = max(0, end_x + half_kernel + 1 - width)
        kernel_matrix = np.pad(
            kernel_matrix,
            ((pad_top, pad_bottom), (pad_left, pad_right)),
            mode="constant",
            constant_values=0
        )

    config_params = {
        "image_array": ImageArray(data=image_array),
        "technique": technique,
        "results": results,
        "ui_placeholders": ui_placeholders,
        "show_per_pixel_processing": show_per_pixel_processing,
        "processing_end": results.processing_end_coord,
        "kernel": KernelConfig(
            size=results.kernel_size,
            origin=(half_kernel, half_kernel),
            kernel_matrix=kernel_matrix,
        ),
        "search_window": SearchWindowConfig(
            size=results.search_window_size if technique == "nlm" else None,
            use_full_image=analysis_params.get("use_full_image", False),
        ),
        "last_processed_pixel": results.processing_end_coord,
        "pixels_to_process": results.pixels_processed,
        "original_pixel_value": image_array[end_y, end_x],
    }

    return VisualizationConfig(**config_params)
