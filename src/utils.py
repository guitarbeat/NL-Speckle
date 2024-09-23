"""
Utility functions and classes for image processing and comparison.
"""

# Import necessary modules
from src.plotting import VisualizationConfig
import numpy as np
import streamlit as st
from typing import Dict, Any, Tuple
import matplotlib.pyplot as plt
from streamlit_image_comparison import image_comparison
from src.decor import log_action



@log_action
def update_session_state(technique: str, pixels_to_process: int, results: Any) -> None:
    """Update session state with processing results."""
    st.session_state.update(
        {"processed_pixels": pixels_to_process, f"{technique}_results": results}
    )

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
        """Selects two images from a list of available images using a UI.
        Parameters:
            - available_images (list): A list of image names to be displayed for selection.
        Returns:
            - tuple: A tuple containing the names of the first and second selected images.
        Processing Logic:
            - Utilizes UI columns for a side-by-side selection.
            - Initializes both select boxes with an empty string as the first option."""
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
        """Return the first two elements from a list of normalized images if the list contains exactly two elements.
        Parameters:
            - normalized_images (list): A list of normalized images to process.
        Returns:
            - tuple: A tuple containing the first and second image from the list if there are exactly two images, otherwise (None, None).
        Processing Logic:
            - The function enforces that the input is a list and contains exactly two items by checking its type and length."""
        if not isinstance(normalized_images, list):
            st.error("Expected a list of normalized images.")
            return None, None

        if len(normalized_images) != 2:
            st.error(f"Got {len(normalized_images)} images. Expected 2.")
            return None, None

        return normalized_images[0], normalized_images[1]

    @staticmethod
    def display_same_image_comparison(img1, img2, image_choice_1, image_choice_2, cmap_name):
        """Displays a side-by-side comparison of two images with specified color mappings.
        Parameters:
            - img1 (np.ndarray): First image to compare.
            - img2 (np.ndarray): Second image to compare.
            - image_choice_1 (str): Label for the first image.
            - image_choice_2 (str): Label for the second image.
            - cmap_name (str): Name of the colormap to apply to both images.
        Returns:
            - None: This function does not return a value; it renders the images directly to a display.
        Processing Logic:
            - Normalize color map and colorize both images using the specified colormap.
            - Convert the normalized images to uint8 format before displaying.
            - Display the images with their corresponding labels side by side for comparison."""
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
        """Displays the difference between two images using a specific colormap.
        Parameters:
            - img1 (np.array): The first image array to compare.
            - img2 (np.array): The second image array to compare.
            - cmap_name (str): The name of the colormap used to colorize the difference.
        Returns:
            - None: This function does not return anything, it directly displays the image.
        Processing Logic:
            - Calculate the absolute difference between the two input images as diff_map.
            - Use the ImageComparison.normalize_and_colorize method to apply the colormap to the diff_map.
            - Use the st.image function from streamlit to display the resulting difference image."""
        diff_map = np.abs(img1 - img2)
        display_diff = ImageComparison.normalize_and_colorize([diff_map], [cmap_name])[0]
        st.image(
            display_diff,
            caption="Difference Map",
            use_column_width=True,
        )