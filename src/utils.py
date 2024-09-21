"""
Utility functions and classes for image processing and comparison.
"""

# Import necessary modules
from src.plotting import run_technique, VisualizationConfig
import numpy as np
import streamlit as st
from typing import Dict, Any, List, Tuple
import matplotlib.pyplot as plt
from streamlit_image_comparison import image_comparison

def setup_and_run_analysis_techniques(analysis_params: Dict[str, Any]) -> None:
    """Set up and run analysis techniques based on the provided parameters."""
    techniques: List[str] = st.session_state.get("techniques", [])
    tabs: List[Any] = st.session_state.get("tabs", [])

    for technique, tab in zip(techniques, tabs):
        if tab is not None:
            with tab:
                run_technique(technique, tab, analysis_params)

def update_session_state(technique: str, pixels_to_process: int, results: Any) -> None:
    """Update session state with processing results."""
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
        technique=technique,
    )

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
                image_choice_1, image_choice_2 = ImageComparison.get_image_choices(available_images)

                if image_choice_1 and image_choice_2:
                    img1, img2 = images[image_choice_1], images[image_choice_2]

                    if image_choice_1 == image_choice_2:
                        ImageComparison.display_same_image_comparison(img1, img2, image_choice_1, image_choice_2, cmap_name)
                    else:
                        st.error("Please select two different images for comparison.")
                        ImageComparison.display_difference_map(img1, img2, cmap_name)
                else:
                    st.info("Select two images to compare.")
        except (KeyError, ValueError, TypeError) as e:
            st.error(f"Error in image comparison: {str(e)}")
    @staticmethod
    def get_image_choices(available_images):
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
    
    @staticmethod
    def normalize_and_colorize(images, cmap_names):
        colored_images = []
        for img, cmap_name in zip(images, cmap_names):
            normalized = (img - np.min(img)) / (np.max(img) - np.min(img)) if np.max(img) - np.min(img) != 0 else img
            colored = plt.get_cmap(cmap_name)(normalized)[:, :, :3]
            colored_images.append((colored * 255).astype(np.uint8))
        return colored_images

    @staticmethod
    def process_normalized_images(normalized_images):
        if not isinstance(normalized_images, list):
            st.error("Expected a list of normalized images.")
            return None, None

        if len(normalized_images) != 2:
            st.error(f"Got {len(normalized_images)} images. Expected 2.")
            return None, None

        return normalized_images[0], normalized_images[1]

    @staticmethod
    def display_same_image_comparison(img1, img2, image_choice_1, image_choice_2, cmap_name):
        normalized_images = ImageComparison.normalize_and_colorize([img1, img2], [cmap_name] * 2)
        img1_uint8, img2_uint8 = ImageComparison.process_normalized_images(normalized_images)

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

    @staticmethod
    def display_difference_map(img1, img2, cmap_name):
        diff_map = np.abs(img1 - img2)
        display_diff = ImageComparison.normalize_and_colorize([diff_map], [cmap_name])[0]
        st.image(
            display_diff,
            caption="Difference Map",
            use_column_width=True,
        )