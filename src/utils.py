"""
Utility functions and classes for image processing and comparison.
"""

from typing import Any, Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from streamlit_image_comparison import image_comparison

from src.decor import log_action

# Import necessary modules
from src.plotting import VisualizationConfig


@log_action
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


@log_action
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
                image_choice_1, image_choice_2 = ImageComparison.get_image_choices(
                    available_images
                )

                if image_choice_1 and image_choice_2:
                    img1, img2 = images[image_choice_1], images[image_choice_2]

                    if image_choice_1 == image_choice_2:
                        ImageComparison.display_same_image_comparison(
                            img1, img2, image_choice_1, image_choice_2, cmap_name
                        )
                    else:
                        st.error("Please select two different images for comparison.")
                        ImageComparison.display_difference_map(img1, img2, cmap_name)
                else:
                    st.info("Select two images to compare.")
        except (KeyError, ValueError, TypeError) as e:
            st.error(f"Error in image comparison: {str(e)}")

    @staticmethod
    def get_image_choices(available_images):
        """Provides a user interface for selecting two images to compare.
        Parameters:
            - available_images (list): A list of image filenames available for selection.
        Returns:
            - tuple: A pair of selected image filenames.
        Processing Logic:
            - The function uses Streamlit to create two columns for dual dropdown menus.
            - Dropdowns are populated with the available images plus an empty option at the beginning."""
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
        """Normalize and map a list of images to specified color maps.
        Parameters:
            - images (list): A list of image arrays to be normalized and colored.
            - cmap_names (list): A list of colormap names corresponding to each image.
        Returns:
            - list: A list of images that have been normalized and colorized.
        Processing Logic:
            - Normalization is done based on the min and max values of the input image.
            - Avoids division by zero by checking if the max and min of the image are not equal.
            - The normalized image is then color-mapped using the specified color map name.
            - The color-mapped image is converted to a uint8 format suitable for image representation."""
        colored_images = []
        for img, cmap_name in zip(images, cmap_names):
            normalized = (
                (img - np.min(img)) / (np.max(img) - np.min(img))
                if np.max(img) - np.min(img) != 0
                else img
            )
            colored = plt.get_cmap(cmap_name)(normalized)[:, :, :3]
            colored_images.append((colored * 255).astype(np.uint8))
        return colored_images

    @staticmethod
    def process_normalized_images(normalized_images):
        """Processes a pair of normalized images.
        Parameters:
            - normalized_images (list): A list containing exactly two normalized images.
        Returns:
            - tuple: A tuple containing the two images from the list if there are exactly two images, else returns (None, None).
        Processing Logic:
            - The function checks whether the input is a list and contains exactly two elements.
            - If the input is not a list or does not have two elements, an error is displayed and (None, None) is returned."""
        if not isinstance(normalized_images, list):
            st.error("Expected a list of normalized images.")
            return None, None

        if len(normalized_images) != 2:
            st.error(f"Got {len(normalized_images)} images. Expected 2.")
            return None, None

        return normalized_images[0], normalized_images[1]

    @staticmethod
    def display_same_image_comparison(
        img1, img2, image_choice_1, image_choice_2, cmap_name
    ):
        """Displays two images side by side for visual comparison.
        Parameters:
            - img1 (np.array): The first image to compare.
            - img2 (np.array): The second image to compare.
            - image_choice_1 (str): Label for the first image.
            - image_choice_2 (str): Label for the second image.
            - cmap_name (str): The colormap to be used for colorizing the images.
        Returns:
            - None: The function does not return any value, it only exhibits the side by side comparison of images.
        Processing Logic:
            - Normalizes and colorizes the input images using the specified colormap.
            - Converts the normalized images to a uint8 format suitable for displaying.
            - Uses an image comparison plot function to show the images side by side with labels."""
        normalized_images = ImageComparison.normalize_and_colorize(
            [img1, img2], [cmap_name] * 2
        )
        img1_uint8, img2_uint8 = ImageComparison.process_normalized_images(
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

    @staticmethod
    def display_difference_map(img1, img2, cmap_name):
        """Displays the difference between two images with a specified colormap.
        Parameters:
            - img1 (array-like): The first image to compare.
            - img2 (array-like): The second image to compare.
            - cmap_name (str): The name of the colormap to use for displaying the difference.
        Returns:
            - None: The function does not return anything but displays the difference map in Streamlit.
        Processing Logic:
            - The absolute difference between img1 and img2 is calculated.
            - The difference is normalized and colorized according to the specified colormap.
            - The resulting difference map is displayed in Streamlit with the caption "Difference Map"."""
        diff_map = np.abs(img1 - img2)
        display_diff = ImageComparison.normalize_and_colorize([diff_map], [cmap_name])[
            0
        ]
        st.image(
            display_diff,
            caption="Difference Map",
            use_column_width=True,
        )
