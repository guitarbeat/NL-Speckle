"""
Utility functions for image processing and comparison.
"""

import numpy as np
from streamlit_image_comparison import image_comparison
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image
from typing import Optional, Tuple, Dict, List
import streamlit as st
from src.session_state import get_color_map, handle_processing_error, get_technique_result

def prepare_comparison_images() -> Dict[str, np.ndarray]:
    """Prepare images from various techniques for comparison."""
    try:
        techniques = ["Speckle", "NL-Means"]
        return {f"{t} Output": res['output'] for t in techniques if (res := get_technique_result(t)) and 'output' in res} | \
               {"Original Image": get_technique_result("Original")} if get_technique_result("Original") else {}
    except Exception as e:
        st.error(f"Error preparing comparison images: {e}")
        return {}

def handle_image_comparison(tab):
    techniques = {
        'Speckle': ('speckle', 'speckle_contrast_filter'),
        'NL-Means': ('nlm', 'nonlocal_means')
    }
    
    images_to_compare = []
    for technique, (result_key, image_key) in techniques.items():
        result = get_technique_result(result_key)
        if result is not None and image_key in result:
            images_to_compare.append((technique, result[image_key]))
    
    if not images_to_compare:
        tab.warning("No processed images available for comparison.")
        return
    
    if len(images_to_compare) == 1:
        tab.write(f"Displaying processed image: {images_to_compare[0][0]}")
        display_image(tab, images_to_compare[0][1], images_to_compare[0][0])
    else:
        tab.write("Comparing processed images:")
        cols = tab.columns(len(images_to_compare))
        for col, (technique, image_data) in zip(cols, images_to_compare):
            display_image(col, image_data, technique)

def display_image(container, image_data, caption):
    if isinstance(image_data, np.ndarray):
        # If it's a numpy array, convert to PIL Image
        image = Image.fromarray((image_data * 255).astype(np.uint8))
    elif isinstance(image_data, Image.Image):
        # If it's already a PIL Image, use it directly
        image = image_data
    else:
        container.warning(f"Unsupported image format for {caption}")
        return

    container.image(image, caption=caption, use_column_width=True)

def get_image_choices(available_images: List[str]) -> Tuple[str, str]:
    """Allow user to choose two images for comparison."""
    try:
        col1, col2 = st.columns(2)
        return (col1.selectbox("Select first image:", options=[""] + available_images, index=0),
                col2.selectbox("Select second image:", options=[""] + available_images, index=min(1, len(available_images))))
    except Exception as e:
        st.error(f"Error getting image choices: {e}")
        return "", ""

def compare_images(images: Dict[str, np.ndarray], img1_label: str, img2_label: str) -> None:
    """Perform image comparison and display results."""
    try:
        img1, img2 = images[img1_label], images[img2_label]
        normalized_images = normalize_and_colorize([img1, img2])
        if not normalized_images:
            return handle_processing_error("Error processing images for comparison.")

        img1_uint8, img2_uint8 = normalized_images
        image_comparison(img1=img1_uint8, img2=img2_uint8, label1=img1_label, label2=img2_label)
        st.subheader("Selected Images")
        st.image([img1_uint8, img2_uint8], caption=[img1_label, img2_label])

        if diff_map := display_difference_map(img1, img2):
            st.subheader("Download Images")
            for img, label in zip([img1_uint8, img2_uint8, diff_map], [f"{img1_label}.png", f"{img2_label}.png", "difference_map.png"]):
                create_download_button(img, label)
    except Exception as e:
        handle_processing_error(f"Error in image comparison: {e}")

def create_download_button(image: np.ndarray, filename: str) -> None:
    """Generate a download button for the given image."""
    try:
        buf = BytesIO()
        Image.fromarray(image).save(buf, format="PNG")
        st.download_button(label=f"Download {filename}", data=buf.getvalue(), file_name=filename, mime="image/png")
    except Exception as e:
        st.error(f"Error creating download button: {e}")

def display_difference_map(img1: np.ndarray, img2: np.ndarray) -> Optional[np.ndarray]:
    """Compute and display the difference map between two images."""
    try:
        diff_map = np.abs(img1 - img2)
        if normalized_diff := normalize_and_colorize([diff_map]):
            diff_map_uint8 = normalized_diff[0]
            st.image(diff_map_uint8, caption="Difference Map")
            return diff_map_uint8
        st.error("Error creating difference map.")
    except Exception as e:
        st.error(f"Error creating difference map: {e}")
    return None

def normalize_and_colorize(images: List[np.ndarray]) -> Optional[List[np.ndarray]]:
    """Normalize and apply color map to images."""
    try:
        color_map = get_color_map()
        return [(plt.get_cmap(color_map)((img - np.min(img)) / (np.ptp(img) + 1e-6))[:, :, :3] * 255).astype(np.uint8) for img in images]
    except Exception as e:
        st.error(f"Error normalizing and colorizing images: {e}")
        return None
