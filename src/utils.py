"""
Utility functions and classes for image processing and comparison.
"""

import numpy as np
import streamlit as st
from streamlit_image_comparison import image_comparison
import matplotlib.pyplot as plt

class ImageComparison:
    """Class for handling image comparison functionality."""

    @staticmethod
    def handle(tab, cmap_name, images):
        with tab:
            st.header("Image Comparison")
            
            if not images:
                st.warning("No images available for comparison.")
                return

            available_images = list(images.keys())
            # st.write(f"Debug: Available images: {available_images}")

            image_choice_1, image_choice_2 = ImageComparison._get_image_choices(available_images)
            
            if image_choice_1 and image_choice_2:
                if image_choice_1 != image_choice_2:
                    ImageComparison._compare_images(images, image_choice_1, image_choice_2, cmap_name)
                else:
                    st.error("Please select two different images for comparison.")
            else:
                st.info("Select two images to compare.")

    @staticmethod
    def _get_image_choices(available_images):
        col1, col2 = st.columns(2)
        image_choice_1 = col1.selectbox(
            "Select first image to compare:",
            options=[""] + available_images,
            index=0,
            key="image_choice_1"
        )
        
        default_index = min(1, len(available_images))
        image_choice_2 = col2.selectbox(
            "Select second image to compare:",
            options=[""] + available_images,
            index=default_index,
            key="image_choice_2"
        )
        
        # st.write(f"Debug: Selected images: '{image_choice_1}', '{image_choice_2}'")
        return image_choice_1, image_choice_2

    @staticmethod
    def _compare_images(images, image_choice_1, image_choice_2, cmap_name):
        try:
            img1, img2 = images[image_choice_1], images[image_choice_2]
            # st.write(f"Debug: Image shapes: {img1.shape}, {img2.shape}")
            
            normalized_images = ImageComparison._normalize_and_colorize([img1, img2], cmap_name)
            if normalized_images:
                img1_uint8, img2_uint8 = normalized_images
                
                image_comparison(
                    img1=img1_uint8,
                    img2=img2_uint8,
                    label1=image_choice_1,
                    label2=image_choice_2,
                )
                
                st.subheader("Selected Images")
                st.image([img1_uint8, img2_uint8], caption=[image_choice_1, image_choice_2])
                
                ImageComparison._display_difference_map(img1, img2, cmap_name)
            else:
                st.error("Error processing images for comparison.")
        except Exception as e:
            st.error(f"Error in image comparison: {str(e)}")
            st.exception(e)

    @staticmethod
    def _normalize_and_colorize(images, cmap_name):
        try:
            normalized_images = []
            for img in images:
                if np.max(img) == np.min(img):
                    normalized = img
                else:
                    normalized = (img - np.min(img)) / (np.max(img) - np.min(img))
                colored = plt.get_cmap(cmap_name)(normalized)[:, :, :3]
                normalized_images.append((colored * 255).astype(np.uint8))
            return normalized_images
        except Exception as e:
            st.error(f"Error in normalizing and colorizing images: {str(e)}")
            return None

    @staticmethod
    def _display_difference_map(img1, img2, cmap_name):
        try:
            diff_map = np.abs(img1 - img2)
            normalized_diff = ImageComparison._normalize_and_colorize([diff_map], cmap_name)
            if normalized_diff:
                st.image(normalized_diff[0], caption="Difference Map")
            else:
                st.error("Error creating difference map.")
        except Exception as e:
            st.error(f"Error in creating difference map: {str(e)}")