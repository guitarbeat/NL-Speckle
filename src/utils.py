"""
Utility functions and classes for image processing and comparison.
"""

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from streamlit_image_comparison import image_comparison

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
        diff_map = np.abs(img1 - img2)
        display_diff = ImageComparison.normalize_and_colorize([diff_map], [cmap_name])[
            0
        ]
        st.image(
            display_diff,
            caption="Difference Map",
            use_column_width=True,
        )
