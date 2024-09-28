"""
This module provides plotting functionalities using Matplotlib and Streamlit.
"""

from dataclasses import dataclass
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from src.math.formula import display_analysis_formula, prepare_variables
from src.classes import SpeckleResult, NLMResult, VisualizationConfig, SearchWindowConfig, KernelVisualizationConfig
from src.draw.overlay import add_overlays

# Constants for Image Visualization
SPECKLE_CONTRAST, ORIGINAL_IMAGE, NON_LOCAL_MEANS = "Speckle Contrast", "Original Image", "Non-Local Means"
DEFAULT_SPECKLE_VIEW, DEFAULT_NLM_VIEW = [SPECKLE_CONTRAST, ORIGINAL_IMAGE], [NON_LOCAL_MEANS, ORIGINAL_IMAGE]

@dataclass
class ImageArray:
    data: np.ndarray

def visualize_image(image: np.ndarray, placeholder, *, config: VisualizationConfig) -> None:
    try:
        if config.zoom:
            image, new_center_x, new_center_y = get_zoomed_image_section(
                image, config.last_processed_pixel[0], config.last_processed_pixel[1], config.kernel.size
            )
            config.last_processed_pixel = (new_center_x, new_center_y)

        fig, ax = plt.subplots(1, 1, figsize=config.figure_size)
        ax.imshow(image, vmin=config.vmin, vmax=config.vmax, cmap=st.session_state.color_map)
        ax.set_title(config.title)
        ax.axis("off")
        add_overlays(ax, image, config)
        fig.tight_layout(pad=2)
        placeholder.pyplot(fig)
        plt.close(fig)
    except Exception as e:
        placeholder.error(f"An error occurred while visualizing the image: {e}. Please check the logs for details.")

def get_zoomed_image_section(image: np.ndarray, center_x: int, center_y: int, kernel_size: int):
    half_zoom = kernel_size // 2
    top, bottom = max(0, center_y - half_zoom), min(image.shape[0], center_y + half_zoom + 1)
    left, right = max(0, center_x - half_zoom), min(image.shape[1], center_x + half_zoom + 1)
    return image[top:bottom, left:right], center_x - left, center_y - top

def create_technique_ui(technique: str, tab: st.delta_generator.DeltaGenerator, show_per_pixel_processing: bool):
    with tab:
        ui_placeholders = {"formula": st.empty(), "original_image": st.empty()}
        filter_options = ["Original Image"] + (SpeckleResult.get_filter_options() if technique == "speckle" else NLMResult.get_filter_options())
        default_selection = DEFAULT_SPECKLE_VIEW if technique == "speckle" else DEFAULT_NLM_VIEW
        selected_filters = st.multiselect("Select views to display", filter_options, default=default_selection, key=f"{technique}_filter_selection")
        st.session_state[f"{technique}_selected_filters"] = selected_filters

        if selected_filters:
            columns = st.columns(len(selected_filters))
            for i, filter_name in enumerate(selected_filters):
                filter_key = filter_name.lower().replace(" ", "_")
                ui_placeholders[filter_key] = columns[i].empty()
                if show_per_pixel_processing:
                    ui_placeholders[f"zoomed_{filter_key}"] = columns[i].expander(f"Zoomed-in {filter_name}", expanded=False).empty()
        else:
            st.warning("No views selected. Please select at least one view to display.")

        if show_per_pixel_processing:
            ui_placeholders["zoomed_kernel"] = st.empty()

    return ui_placeholders

def visualize_analysis_results(viz_params: VisualizationConfig) -> None:
    filter_options = viz_params.results.filter_data
    filter_options["Original Image"] = viz_params.image_array.data
    selected_filters = st.session_state.get(f"{viz_params.technique}_selected_filters", [])

    for filter_name in selected_filters:
        if filter_name in filter_options:
            filter_data = filter_options[filter_name]
            for plot_type in ["main", "zoomed"]:
                plot_key = f"{'zoomed_' if plot_type == 'zoomed' else ''}{filter_name.lower().replace(' ', '_')}"
                if plot_key in viz_params.ui_placeholders and (plot_type != "zoomed" or viz_params.show_per_pixel_processing):
                    config = VisualizationConfig(
                        **{**viz_params.__dict__,
                           "vmin": None if filter_name == "Original Image" else np.min(filter_data),
                           "vmax": None if filter_name == "Original Image" else np.max(filter_data),
                           "zoom": (plot_type == "zoomed"),
                           "show_kernel": (viz_params.show_per_pixel_processing if plot_type == "main" else True),
                           "show_per_pixel_processing": (plot_type == "zoomed"),
                           "title": f"Zoomed-In {filter_name}" if plot_type == "zoomed" else filter_name}
                    )
                    visualize_image(filter_data, viz_params.ui_placeholders[plot_key], config=config)

    last_x, last_y = viz_params.last_processed_pixel
    specific_params = {
        "kernel_size": viz_params.results.kernel_size,
        "pixels_processed": viz_params.results.pixels_processed,
        "total_pixels": viz_params.results.kernel_size**2,
        "x": last_x,
        "y": last_y,
        "image_height": viz_params.image_array.data.shape[0],
        "image_width": viz_params.image_array.data.shape[1],
        "half_kernel": viz_params.kernel.size // 2,
        "valid_height": viz_params.image_array.data.shape[0] - viz_params.kernel.size + 1,
        "valid_width": viz_params.image_array.data.shape[1] - viz_params.kernel.size + 1,
        "search_window_size": viz_params.search_window.size,
        "kernel_matrix": viz_params.kernel.kernel_matrix,
        "original_value": viz_params.original_pixel_value,
        "analysis_type": viz_params.technique,
    }

    if viz_params.technique == "nlm":
        specific_params.update({
            "filter_strength": viz_params.results.filter_strength,
            "search_window_size": viz_params.results.search_window_size,
            "nlm_value": viz_params.results.nonlocal_means[last_y, last_x],
        })
    else:  # speckle
        specific_params.update({
            "mean": viz_params.results.mean_filter[last_y, last_x],
            "std": viz_params.results.std_dev_filter[last_y, last_x],
            "sc": viz_params.results.speckle_contrast_filter[last_y, last_x],
        })

    specific_params = prepare_variables(specific_params, viz_params.technique)

    display_analysis_formula(
        specific_params,
        viz_params.ui_placeholders,
        viz_params.technique,
        last_x,
        last_y,
        viz_params.kernel.size,
        viz_params.kernel.kernel_matrix,
        viz_params.original_pixel_value,
    )

def run_technique(technique: str, tab: st.delta_generator.DeltaGenerator) -> None:
    show_per_pixel_processing = st.session_state.get("show_per_pixel", False)

    ui_placeholders = create_technique_ui(technique, tab, show_per_pixel_processing)
    st.session_state[f"{technique}_placeholders"] = ui_placeholders

    try:
        if f"{technique}_result" not in st.session_state or st.session_state[f"{technique}_result"] is None:
            st.warning(f"{technique.upper()} processing not complete. Please wait.")
            return

        results = st.session_state[f"{technique}_result"]
        
        if "image_array" not in st.session_state:
            st.error("Image array not found in session state. Please upload an image first.")
            return

        image_array = st.session_state.image_array
        height, width = image_array.shape

        half_kernel = results.kernel_size // 2
        end_x, end_y = results.processing_end_coord

        end_x = min(max(end_x, half_kernel), width - half_kernel - 1)
        end_y = min(max(end_y, half_kernel), height - half_kernel - 1)

        kernel_matrix = image_array[
            max(0, end_y - half_kernel):min(height, end_y + half_kernel + 1),
            max(0, end_x - half_kernel):min(width, end_x + half_kernel + 1)
        ].copy()

        if kernel_matrix.shape != (results.kernel_size, results.kernel_size):
            kernel_matrix = np.pad(
                kernel_matrix,
                (
                    (max(0, half_kernel - end_y), max(0, end_y + half_kernel + 1 - height)),
                    (max(0, half_kernel - end_x), max(0, end_x + half_kernel + 1 - width))
                ),
                mode="constant", constant_values=0
            )

        # Ensure kernel_matrix is a numpy array
        if not isinstance(kernel_matrix, np.ndarray):
            kernel_matrix = np.array(kernel_matrix)

        viz_config = VisualizationConfig(
            image_array=ImageArray(data=image_array),
            technique=technique,
            results=results,
            ui_placeholders=ui_placeholders,
            show_per_pixel_processing=show_per_pixel_processing,
            processing_end=(end_x, end_y),
            kernel=KernelVisualizationConfig(
                size=results.kernel_size,
                origin=(half_kernel, half_kernel),
                kernel_matrix=kernel_matrix,
            ),
            search_window=SearchWindowConfig(
                size=getattr(results, 'search_window_size', None) if technique == "nlm" else None,
                use_full_image=st.session_state.get("use_whole_image", False),
            ),
            last_processed_pixel=(end_x, end_y),
            pixels_to_process=st.session_state.get("exact_pixel_count", results.pixels_processed),
            original_pixel_value=float(image_array[end_y, end_x]),
        )

        visualize_analysis_results(viz_config)

    except Exception as e:
        st.error(f"Error for {technique}: {str(e)}. Please check the logs for details.")
        import traceback
        st.error(f"Traceback: {traceback.format_exc()}")

def prepare_comparison_images():
    comparison_images = {
        "Unprocessed Image": st.session_state.get("image_array", np.array([]))
    }

    for result_key in ["speckle_result", "nlm_result"]:
        if results := st.session_state.get(result_key):
            comparison_images.update(results.filter_data)

    return comparison_images if len(comparison_images) > 1 else None
