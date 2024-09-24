"""
This module provides plotting functionalities using Matplotlib and Streamlit.
"""


from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from src.overlay import add_overlays, VisualizationConfig, KernelConfig, SearchWindowConfig
from src.formula import display_analysis_formula
from src.nlm import NLMResult
from src.processing import (
    ProcessParams,
    configure_process_params,
    extract_kernel_from_image,
    process_image,
)
from src.speckle import SpeckleResult

# Constants for Image Visualization
DEFAULT_SPECKLE_VIEW = ["Speckle Contrast", "Original Image"]
DEFAULT_NLM_VIEW = ["Non-Local Means", "Original Image"]


def generate_plot_key(filter_name: str, plot_type: str) -> str:
    """Generate a key for identifying plots based on filter name and plot
    type."""
    base_key = filter_name.lower().replace(" ", "_")
    return f"zoomed_{base_key}" if plot_type == "zoomed" else base_key


@dataclass
class PixelCoordinates:
    """Represents the pixel's (x, y) coordinates."""

    x: int
    y: int


@dataclass
class ImageArray:
    """Container for the image data as a numpy array."""

    data: np.ndarray


def create_process_params(
    analysis_params: Dict[str, Any], technique: str, technique_params: Dict[str, Any]
) -> ProcessParams:
    """Create process parameters based on analysis and technique.

    Args:
        analysis_params (Dict[str, Any]): Parameters for analysis. technique
        (str): The technique to be used. technique_params (Dict[str, Any]):
        Parameters specific to the technique.

    Returns:
        ProcessParams: The created process parameters.
    """
    common_params = {
        "kernel_size": st.session_state.get("kernel_size", 3),
        "pixels_to_process": analysis_params.get("pixels_to_process", 0),
        "total_pixels": analysis_params.get("total_pixels", 0),
        "show_per_pixel_processing": analysis_params.get(
            "show_per_pixel_processing", False
        ),
    }

    if technique == "nlm":
        common_params |= {
            "search_window_size": analysis_params.get("search_window_size"),
            "filter_strength": analysis_params.get("filter_strength"),
        }

    return ProcessParams(
        image_array=analysis_params.get(
            "image_array", ImageArray(np.array([]))),
        technique=technique,
        analysis_params=technique_params | common_params,
        show_per_pixel_processing=analysis_params.get(
            "show_per_pixel_processing", False
        ),
    )


def visualize_filter_and_zoomed(
    filter_name: str, filter_data: np.ndarray, viz_config: VisualizationConfig
):
    """Visualize the main and zoomed versions of a filter."""
    for plot_type in ["main", "zoomed"]:
        plot_key = generate_plot_key(filter_name, plot_type)

        # Skip unnecessary visualizations
        if plot_key not in viz_config.ui_placeholders or (
            plot_type == "zoomed" and not viz_config.show_per_pixel_processing
        ):
            continue

        # Create updated config for zoomed view
        config = update_visualization_config(
            viz_config, filter_data, filter_name, plot_type
        )
        title = f"Zoomed-In {filter_name}" if plot_type == "zoomed" else filter_name

        # Update the config with the title instead of passing it separately
        config.title = title

        visualize_image(
            filter_data, viz_config.ui_placeholders[plot_key], config=config
        )


def update_visualization_config(
    viz_config: VisualizationConfig,
    filter_data: np.ndarray,
    filter_name: str,
    plot_type: str,
) -> VisualizationConfig:
    """Update the VisualizationConfig object for zoomed or main view."""
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


def create_image_plot(
    plot_image: np.ndarray, config: VisualizationConfig
) -> plt.Figure:
    """Creates an image plot from the given image and configuration.

    Args:
        plot_image (np.ndarray): The image data to plot. config
        (VisualizationConfig): The configuration for visualization.

    Returns:
        plt.Figure: The created plot figure.
    """
    fig, ax = plt.subplots(1, 1, figsize=config.figure_size)
    ax.imshow(plot_image, vmin=config.vmin,
              vmax=config.vmax, cmap=st.session_state.color_map)
    ax.set_title(config.title)
    ax.axis("off")

    add_overlays(ax, plot_image, config)
    fig.tight_layout(pad=2)
    return fig


def prepare_filter_options_and_parameters(
    results: Any, last_processed_pixel: Tuple[int, int]
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """
    Prepare filter options and specific parameters based on the analysis
    results.

    Args:
        results (Any): The analysis results object. last_processed_pixel
        (Tuple[int, int]): Coordinates of the last processed pixel.

    Returns:
        Tuple[Dict[str, np.ndarray], Dict[str, Any]]: Filter options and
        specific parameters.
    """
    end_x, end_y = last_processed_pixel
    filter_options = results.get_filter_data()
    specific_params = {
        "kernel_size": results.kernel_size,
        "pixels_processed": results.pixels_processed,
        "total_pixels": results.kernel_size**2,
    }

    for filter_name, filter_data in filter_options.items():
        if isinstance(filter_data, np.ndarray) and filter_data.size > 0:
            specific_params[filter_name.lower().replace(" ", "_")] = filter_data[
                end_y, end_x
            ]

    # Include any additional attributes if available
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
    """
    Prepare images for comparison from different analysis results. Returns:
        Optional[Dict[str, np.ndarray]]: A dictionary of image names and their
        corresponding arrays, or None.
    """
    comparison_images = {
        "Unprocessed Image": st.session_state.get("analysis_params", {}).get(
            "image_array", np.array([])
        )
    }

    for result_key in ["speckle_results", "nlm_results"]:
        results = st.session_state.get(result_key)
        if results is not None:
            comparison_images |= results.get_filter_data()

    return comparison_images if len(comparison_images) > 1 else None


def get_zoomed_image_section(
    image: np.ndarray, center_x: int, center_y: int, kernel_size: int
) -> Tuple[np.ndarray, int, int]:
    """
    Extract a zoomed section of the image.

    Args:
        image (np.ndarray): The original image. center_x (int): X-coordinate of
        the zoom center. center_y (int): Y-coordinate of the zoom center.
        kernel_size (int): Size of the zoomed section.

    Returns:
        Tuple[np.ndarray, int, int]: Zoomed image section and new center
        coordinates.
    """
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
    """
    Visualize an image with optional zooming and overlays.

    Args:
        image (np.ndarray): The image array to visualize. placeholder: Streamlit
        placeholder to display the plot. config (VisualizationConfig):
        Configuration for visualization options.
    """
    try:
        # Optional zooming on the image
        if config.zoom:
            try:
                image, new_center_x, new_center_y = get_zoomed_image_section(
                    image,
                    config.last_processed_pixel[0],  # x coordinate
                    config.last_processed_pixel[1],  # y coordinate
                    config.kernel.size,
                )
                config.last_processed_pixel = (new_center_x, new_center_y)
            except Exception as e:
                placeholder.error(
                    f"An error occurred while zooming the image: {
                        e}. Please check the logs for details."
                )
                return

        try:
            fig = create_image_plot(image, config)
            placeholder.pyplot(fig)  # Pass the figure object to pyplot
            # Ensure figure is closed after rendering to free up memory
            plt.close(fig)
        except Exception as e:
            placeholder.error(
                f"An error occurred while creating the image plot: {
                    e}. Please check the logs for details."
            )
    except (ValueError, TypeError, KeyError) as e:
        placeholder.error(
            f"An error occurred while visualizing the image: {
                e}. Please check the logs for details."
        )


def get_filter_options(technique: str) -> List[str]:
    """
    Get filter options based on the image processing technique.

    Args:
        technique (str): Image processing technique (e.g., 'nlm', 'speckle').

    Returns:
        List[str]: A list of filter options for the technique.
    """
    if technique == "speckle":
        return ["Original Image"] + SpeckleResult.get_filter_options()
    elif technique == "nlm":
        return ["Original Image"] + NLMResult.get_filter_options()
    else:
        return []


def create_filter_selection(technique: str, filter_options: List[str]) -> List[str]:
    """
    Create and return a filter selection UI element.

    Args:
        technique (str): Image processing technique. filter_options (List[str]):
        List of available filter options.

    Returns:
        List[str]: List of selected filters from the UI.
    """
    # Default views for different techniques
    default_selection = (
        DEFAULT_SPECKLE_VIEW if technique == "speckle" else DEFAULT_NLM_VIEW
    )

    # Create multi-select widget for filter views
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
    """
    Create views for the selected filters in the UI.

    Args:
        selected_filters (List[str]): List of selected filters to display.
        ui_placeholders (Dict[str, Any]): Dictionary of placeholders to update.
        show_per_pixel_processing (bool): Whether to display per-pixel
        processing views.
    """
    columns = st.columns(len(selected_filters))

    for i, filter_name in enumerate(selected_filters):
        filter_key = filter_name.lower().replace(" ", "_")
        ui_placeholders[filter_key] = columns[
            i
        ].empty()  # Create a placeholder for the filter

        # If per-pixel processing is enabled, create zoomed-in views
        if show_per_pixel_processing:
            ui_placeholders[f"zoomed_{filter_key}"] = (
                columns[i].expander(
                    f"Zoomed-in {filter_name}", expanded=False).empty()
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
            create_filter_views(
                selected_filters, ui_placeholders, show_per_pixel_processing
            )
        else:
            st.warning(
                "No views selected. Please select at least one view to display.")

        if show_per_pixel_processing:
            ui_placeholders["zoomed_kernel"] = st.empty()

    return ui_placeholders


def visualize_analysis_results(viz_params: VisualizationConfig) -> None:
    """
    Visualize analysis results based on the provided parameters.

    Args:
        viz_params (VisualizationConfig): Visualization parameters including
        results, image array, etc.
    """
    filter_options, specific_params = prepare_filter_options_and_parameters(
        viz_params.results, viz_params.last_processed_pixel
    )
    filter_options["Original Image"] = viz_params.image_array.data

    selected_filters = st.session_state.get(
        f"{viz_params.technique}_selected_filters", []
    )
    for filter_name in selected_filters:
        if filter_name in filter_options:
            filter_data = filter_options[filter_name]
            visualize_filter_and_zoomed(filter_name, filter_data, viz_params)

    last_processed_x, last_processed_y = viz_params.last_processed_pixel
    specific_params['Centered at'] = f"({last_processed_x}, {last_processed_y})"
    # adding 'image_height' and 'image_width' to specific_params
    specific_params['image_height'] = viz_params.image_array.data.shape[0]
    specific_params['image_width'] = viz_params.image_array.data.shape[1]
    last_processed_x, last_processed_y = viz_params.last_processed_pixel
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


def run_technique(technique: str, tab: Any, analysis_params: Dict[str, Any]) -> None:
    technique_params = st.session_state.get(f"{technique}_params", {})
    show_per_pixel_processing = analysis_params.get(
        "show_per_pixel_processing", False)

    ui_placeholders = create_technique_ui_elements(
        technique, tab, show_per_pixel_processing
    )
    st.session_state[f"{technique}_placeholders"] = ui_placeholders

    process_params = create_process_params(
        analysis_params, technique, technique_params)

    try:
        configure_process_params(technique, process_params, technique_params)
        
        _, results = process_image(process_params)
        st.session_state[f"{technique}_results"] = results

        viz_config = create_visualization_config(
            process_params.image_array,
            technique,
            analysis_params,
            results,
            ui_placeholders,
            show_per_pixel_processing,
        )

        visualize_analysis_results(viz_config)

    except (ValueError, TypeError, KeyError) as e:
        st.error(f"Error for {technique}: {
                 str(e)}. Please check the logs for details.")


def create_visualization_config(
    image_array: np.ndarray,
    technique: str,
    analysis_params: Dict[str, Any],
    results: Union[SpeckleResult, NLMResult],
    ui_placeholders: Dict[str, Any],
    show_per_pixel_processing: bool,
) -> VisualizationConfig:
    last_processed_x, last_processed_y = results.get_last_processed_coordinates()
    kernel_matrix, original_pixel_value, kernel_size = extract_kernel_from_image(
        image_array,
        last_processed_x,
        last_processed_y,
        analysis_params.get("kernel_size", 3),
    )

    config_params = {
        "image_array": ImageArray(image_array),
        "technique": technique,
        "analysis_params": analysis_params,
        "results": results,
        "last_processed_pixel": (last_processed_x, last_processed_y),
        "original_pixel_value": original_pixel_value,
        "show_per_pixel_processing": show_per_pixel_processing,
        "ui_placeholders": ui_placeholders,
        "kernel": KernelConfig(kernel_matrix=kernel_matrix, size=kernel_size),
    }

    if isinstance(results, NLMResult):
        config_params["search_window"] = SearchWindowConfig(
            size=results.search_window_size,
            use_full_image=analysis_params.get("use_full_image", False)
        )
    else:
        config_params["search_window"] = SearchWindowConfig()

    return VisualizationConfig(**config_params)
