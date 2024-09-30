"""
Utility functions for image processing and comparison.
"""

import numpy as np
from streamlit_image_comparison import image_comparison
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image
from typing import Optional, Tuple, Dict, List
from session_state import prepare_comparison_images, get_color_map, handle_processing_error
import streamlit as st
def handle_image_comparison(tab):
    with tab:
        st.header("Image Comparison")
        
        comparison_images = prepare_comparison_images()
        if not comparison_images:
            handle_processing_error("No images available for comparison.")
            return

        available_images = list(comparison_images.keys())
        image_choice_1, image_choice_2 = get_image_choices(available_images)
        
        if image_choice_1 and image_choice_2:
            if image_choice_1 != image_choice_2:
                compare_images(comparison_images, image_choice_1, image_choice_2)
            else:
                st.error("Please select two different images for comparison.")
        else:
            st.info("Select two images to compare.")

def get_image_choices(available_images: List[str]) -> Tuple[str, str]:
    col1, col2 = st.columns(2)
    image_choice_1 = col1.selectbox("Select first image to compare:", options=[""] + available_images, index=0)
    image_choice_2 = col2.selectbox("Select second image to compare:", options=[""] + available_images, index=min(1, len(available_images)))
    return image_choice_1, image_choice_2

def compare_images(images: Dict[str, np.ndarray], image_choice_1: str, image_choice_2: str):
    try:
        img1, img2 = images[image_choice_1], images[image_choice_2]
        normalized_images = normalize_and_colorize([img1, img2])
        if not normalized_images:
            handle_processing_error("Error processing images for comparison.")
            return

        img1_uint8, img2_uint8 = normalized_images
        
        image_comparison(img1=img1_uint8, img2=img2_uint8, label1=image_choice_1, label2=image_choice_2)
        
        st.subheader("Selected Images")
        st.image([img1_uint8, img2_uint8], caption=[image_choice_1, image_choice_2])
        
        diff_map = display_difference_map(img1, img2)
        
        st.subheader("Download Images")
        col1, col2, col3 = st.columns(3)
        with col1:
            create_download_button(img1_uint8, f"{image_choice_1}.png")
        with col2:
            create_download_button(img2_uint8, f"{image_choice_2}.png")
        with col3:
            if diff_map is not None:
                create_download_button(diff_map, "difference_map.png")
    except Exception as e:
        handle_processing_error(f"Error in image comparison: {str(e)}")
        st.exception(e)

def create_download_button(image: np.ndarray, filename: str):
    buf = BytesIO()
    Image.fromarray(image).save(buf, format="PNG")
    st.download_button(label=f"Download {filename}", data=buf.getvalue(), file_name=filename, mime="image/png")

def display_difference_map(img1: np.ndarray, img2: np.ndarray) -> Optional[np.ndarray]:
    try:
        diff_map = np.abs(img1 - img2)
        normalized_diff = normalize_and_colorize([diff_map])
        if normalized_diff:
            diff_map_uint8 = normalized_diff[0]
            st.image(diff_map_uint8, caption="Difference Map")
            return diff_map_uint8
        st.error("Error creating difference map.")
    except Exception as e:
        st.error(f"Error in creating difference map: {str(e)}")
    return None

def normalize_and_colorize(images: List[np.ndarray]) -> Optional[List[np.ndarray]]:
    try:
        color_map = get_color_map()
        return [
            (plt.get_cmap(color_map)(
                img if np.max(img) == np.min(img) else (img - np.min(img)) / (np.max(img) - np.min(img))
            )[:, :, :3] * 255).astype(np.uint8)
            for img in images
        ]
    except Exception as e:
        st.error(f"Error in normalizing and colorizing images: {str(e)}")
        return None