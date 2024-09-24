"""
Utility functions and classes for image processing and comparison.
"""

from typing import Any, Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from streamlit_image_comparison import image_comparison


# Import necessary modules
from src.plotting import VisualizationConfig


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
        last_processed_pixel=(
            last_processed_pixel[0], last_processed_pixel[1]),
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
                        st.error(
                            "Please select two different images for comparison.")
                        ImageComparison.display_difference_map(
                            img1, img2, cmap_name)
                else:
                    st.info("Select two images to compare.")
        except (KeyError, ValueError, TypeError) as e:
            st.error(f"Error in image comparison: {str(e)}")

    @staticmethod
    def get_image_choices(available_images):
        """Retrieve two user-selected image names from a dropdown list for comparison.
        Parameters:
            - available_images (list of str): List of image names that can be selected by the user.
        Returns:
            - tuple of (str, str): A tuple consisting of the names of the two images selected for comparison.
        Processing Logic:
            - Streamlit columns are used to create a side-by-side dropdown selection for the two images.
            - The first element in the dropdown is left intentionally blank to allow deselection of an image.
            - Both dropdowns use the same list of available images for selection consistency."""
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
        """Normalize and apply a colormap to a list of images.
        Parameters:
            - images (list): A list of images (assumed to be numpy arrays).
            - cmap_names (list): A list of colormap names corresponding to each image.
        Returns:
            - list: A list of colorized and normalized images as numpy arrays in uint8 format.
        Processing Logic:
            - Normalization is performed by finding the range of the pixel intensities and scaling the values between 0 and 1.
            - If the max and min pixel intensities are equal in an image, normalization is bypassed to prevent division by zero.
            - The normalized image is then mapped to a specified colormap.
            - The color-mapped normalized image is scaled to the range of 0 to 255 and converted to uint8 type."""
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
            - normalized_images (list): List containing two normalized images.
        Returns:
            - tuple: A tuple containing the first and second image if the proper conditions are met, otherwise (None, None).
        Processing Logic:
            - The function enforces that the input must be a list containing exactly two elements.
            - It uses Streamlit's error reporting to communicate any issues with the input."""
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
        """Display a side-by-side comparison of two images with given labels and colormap.
        Parameters:
            - img1: Image data for the first image to be compared.
            - img2: Image data for the second image to be compared.
            - image_choice_1 (str): Label for the first image to be displayed.
            - image_choice_2 (str): Label for the second image to be displayed.
            - cmap_name (str): Name of the colormap to be used for displaying the images.
        Returns:
            - None: This function does not return any value. It is intended for visualization purposes.
        Processing Logic:
            - Normalizes and colorizes both images using the specified colormap.
            - Converts the normalized images to uint8 format for display.
            - Calls the image_comparison function to visualize the images side by side with their respective labels.
            - Uses Streamlit (st) to display the subheader and the images."""
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
        """Displays a difference map between two images using a specified colormap.
        Parameters:
            - img1 (ndarray): First input image for comparison.
            - img2 (ndarray): Second input image for comparison.
            - cmap_name (str): Name of the colormap used to colorize the difference.
        Returns:
            - None: The function does not return a value but displays the difference map in a UI.
        Processing Logic:
            - Compute the absolute difference between two input images.
            - Normalize and colorize the result, based on the given colormap name.
            - Display the colorized difference map using the Streamlit UI component."""
        diff_map = np.abs(img1 - img2)
        display_diff = ImageComparison.normalize_and_colorize([diff_map], [cmap_name])[
            0
        ]
        st.image(
            display_diff,
            caption="Difference Map",
            use_column_width=True,
        )
