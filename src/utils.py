"""
Utility functions for image processing and comparison.
"""

import numpy as np
import streamlit as st
from streamlit_image_comparison import image_comparison
import matplotlib.pyplot as plt

def handle_image_comparison(tab, images):
    with tab:
        st.header("Image Comparison")
        
        if not images:
            st.warning("No images available for comparison.")
            return

        available_images = list(images.keys())
        image_choice_1, image_choice_2 = get_image_choices(available_images)
        
        if image_choice_1 and image_choice_2:
            if image_choice_1 != image_choice_2:
                compare_images(images, image_choice_1, image_choice_2)
            else:
                st.error("Please select two different images for comparison.")
        else:
            st.info("Select two images to compare.")

def get_image_choices(available_images):
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
    
    return image_choice_1, image_choice_2

def compare_images(images, image_choice_1, image_choice_2):
    try:
        img1, img2 = images[image_choice_1], images[image_choice_2]
        
        normalized_images = normalize_and_colorize([img1, img2])
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
            
            display_difference_map(img1, img2)
        else:
            st.error("Error processing images for comparison.")
    except Exception as e:
        st.error(f"Error in image comparison: {str(e)}")
        st.exception(e)

def normalize_and_colorize(images):
    try:
        normalized_images = []
        for img in images:
            if np.max(img) == np.min(img):
                normalized = img
            else:
                normalized = (img - np.min(img)) / (np.max(img) - np.min(img))
            colored = plt.get_cmap(st.session_state.color_map)(normalized)[:, :, :3]
            normalized_images.append((colored * 255).astype(np.uint8))
        return normalized_images
    except Exception as e:
        st.error(f"Error in normalizing and colorizing images: {str(e)}")
        return None

def display_difference_map(img1, img2):
    try:
        diff_map = np.abs(img1 - img2)
        normalized_diff = normalize_and_colorize([diff_map])
        if normalized_diff:
            st.image(normalized_diff[0], caption="Difference Map")
        else:
            st.error("Error creating difference map.")
    except Exception as e:
        st.error(f"Error in creating difference map: {str(e)}")