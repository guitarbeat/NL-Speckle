"""
Utility functions and classes for image processing and comparison.

Includes:
- ImageComparison: Handles image comparison in Streamlit.
- FilterResult: Abstract base class for filtering techniques.
- Point and Dimensions: Named tuples for coordinates and dimensions.
- ProcessingDetails: Dataclass for storing image processing details.
- calculate_processing_details: Calculates processing details for an image.
"""

# Import necessary modules
from src.plotting import run_technique, VisualizationConfig
from dataclasses import dataclass
from typing import Tuple
import numpy as np
import streamlit as st
from typing import Dict, Any, List
import matplotlib.pyplot as plt
from streamlit_image_comparison import image_comparison
import json  # Add this import for structured logging
import logging

# --- Type Aliases ---


def setup_and_run_analysis_techniques(analysis_params: Dict[str, Any]) -> None:
    """Set up and run analysis techniques based on the provided parameters."""
    techniques: List[str] = st.session_state.get("techniques", [])
    tabs: List[Any] = st.session_state.get("tabs", [])

    for technique, tab in zip(techniques, tabs):
        if tab is not None:
            with tab:
                run_technique(technique, tab, analysis_params)


def update_session_state(technique: str, pixels_to_process: int, results: Any) -> None:
    """
    Update session state with processing results.

    Args:
        technique (str): Image processing technique.
        pixels_to_process (int): Number of processed pixels.
        results (Any): The result of the image processing.
    """
    st.session_state.update(
        {"processed_pixels": pixels_to_process, f"{technique}_results": results}
    )


def create_visualization_config(
    image_array: np.ndarray,
    technique: str,
    analysis_params: Dict[str, Any],
    results: Any,
    last_processed_pixel: Tuple[int, int],
    kernel_matrix: np.ndarray,
    kernel_size: int,
    original_pixel_value: float,
    show_per_pixel_processing: bool,
) -> VisualizationConfig:
    """Utility to create a VisualizationConfig object."""
    return VisualizationConfig(
        vmin=None,
        vmax=None,
        zoom=False,
        show_kernel=show_per_pixel_processing,
        show_per_pixel_processing=show_per_pixel_processing,
        search_window_size=analysis_params.get("search_window_size"),
        use_full_image=analysis_params.get("use_whole_image", False),
        image_array=image_array,
        analysis_params=analysis_params,
        results=results,
        ui_placeholders=st.session_state.get(f"{technique}_placeholders", {}),
        last_processed_pixel=(last_processed_pixel[0], last_processed_pixel[1]),
        kernel_size=kernel_size,
        kernel_matrix=kernel_matrix,
        original_pixel_value=original_pixel_value,
        color_map=st.session_state.get("color_map", "gray"),
        title=f"{technique.upper()} Analysis Result",
        figure_size=(8, 8),
        technique=technique,  # Keep this line
    )


# Configure logging with structured format
class JsonFormatter(logging.Formatter):
    """Formatter that outputs logs in JSON format."""

    def format(self, record):
        log_obj = {
            "level": record.levelname,
            "message": record.getMessage(),
            "time": self.formatTime(record),
            "function": record.funcName,
            "line": record.lineno,
            "filename": record.filename,
        }
        return json.dumps(log_obj)


# --- Common Error Handling ---
def handle_error(message: str):
    """Handle errors and display an error message in Streamlit."""
    st.error(f"{message}. Please check the logs.")


# --- Image Comparison Class ---
@dataclass
class ImageComparison:
    """Class for handling image comparison functionality."""

    @staticmethod
    def handle(tab, cmap_name, images):
        """Handle image comparison in a Streamlit tab."""
        try:
            with tab:
                st.header("Image Comparison")
                if not images:
                    st.warning("No images available for comparison.")
                    return
                available_images = list(images.keys())

                def get_image_choices():
                    col1, col2 = st.columns(2)
                    image_choice_1 = col1.selectbox(
                        "Select first image to compare:",
                        [""] + available_images,
                        index=0,
                    )
                    image_choice_2 = col2.selectbox(
                        "Select second image to compare:",
                        [""] + available_images,
                        index=0,
                    )
                    return image_choice_1, image_choice_2

                image_choice_1, image_choice_2 = get_image_choices()

                if image_choice_1 and image_choice_2:
                    img1, img2 = images[image_choice_1], images[image_choice_2]

                    def display():
                        if image_choice_1 == image_choice_2:

                            def normalize_and_colorize(images, cmap_names):
                                colored_images = []
                                for img, cmap_name in zip(images, cmap_names):
                                    normalized = (
                                        (img - np.min(img))
                                        / (np.max(img) - np.min(img))
                                        if np.max(img) - np.min(img) != 0
                                        else img
                                    )
                                    colored = plt.get_cmap(cmap_name)(normalized)[
                                        :, :, :3
                                    ]
                                    colored_images.append(
                                        (colored * 255).astype(np.uint8)
                                    )
                                return colored_images

                            def process_normalized_images(normalized_images):
                                if not isinstance(normalized_images, list):
                                    st.error("Expected a list of normalized images.")
                                    return None, None

                                if len(normalized_images) != 2:
                                    st.error(f"Got {len(normalized_images)}.")
                                    return None, None

                                return normalized_images[0], normalized_images[1]

                            normalized_images = normalize_and_colorize(
                                [img1, img2], [cmap_name] * 2
                            )
                            img1_uint8, img2_uint8 = process_normalized_images(
                                normalized_images
                            )

                            image_comparison(
                                img1=img1_uint8,
                                img2=img2_uint8,
                                label1=image_choice_1,
                                label2=image_choice_2,
                            )
                            st.subheader("Selected Images")
                            st.image(
                                [img1_uint8, img2_uint8],
                                caption=[image_choice_1, image_choice_2],
                            )
                        else:
                            st.error(
                                "Please select two different images for comparison."
                            )

                            def display_difference_map():
                                diff_map = np.abs(img1 - img2)
                                display_diff = normalize_and_colorize(
                                    [diff_map], [cmap_name]
                                )[0]
                                st.image(
                                    display_diff,
                                    caption="Difference Map",
                                    use_column_width=True,
                                )

                            display_difference_map()

                    display()
                else:
                    st.info("Select two images to compare.")
        except (KeyError, ValueError, TypeError) as e:
            handle_error(f"Error while handling image comparison: {e}")


# --- Named Tuples for Coordinates and Dimensions ---
