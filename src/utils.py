"""
Utility functions for image processing and comparison.
"""

import numpy as np
import streamlit as st
from streamlit_image_comparison import image_comparison
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image
from typing import Optional, Tuple

###############################################################################
#                             Image Preparation                               #
###############################################################################

def prepare_comparison_images():
    comparison_images = {
        "Unprocessed Image": st.session_state.get("image_array", np.array([]))
    }

    for result_key in ["speckle_result", "nlm_result"]:
        if results := st.session_state.get(result_key):
            comparison_images.update(results.filter_data)

    return comparison_images if len(comparison_images) > 1 else None


###############################################################################
#                           Processing Calculations                           #
###############################################################################

def calculate_processing_end(width: int, height: int, kernel_size: int, pixel_count: int) -> Tuple[int, int]:
    valid_width = width - kernel_size + 1
    if valid_width <= 0:
        raise ValueError("Invalid valid_width calculated. Check image dimensions and kernel_size.")
    end_y, end_x = divmod(pixel_count - 1, valid_width)
    half_kernel = kernel_size // 2
    return (min(end_x + half_kernel, width - 1), min(end_y + half_kernel, height - 1))


###############################################################################
#                              Input Validation                               #
###############################################################################

def validate_input(image: np.ndarray, kernel_size: int, pixel_count: int, 
                   search_window_size: Optional[int] = None, filter_strength: Optional[float] = None):
    if not isinstance(image, np.ndarray) or image.ndim != 2:
        raise ValueError("Image must be a 2D numpy array.")
    if not isinstance(kernel_size, int) or kernel_size <= 0 or kernel_size > min(image.shape) or kernel_size % 2 == 0:
        raise ValueError("Invalid kernel_size. Must be a positive odd integer not larger than the smallest image dimension.")
    if not isinstance(pixel_count, int) or pixel_count <= 0:
        raise ValueError("pixel_count must be a positive integer.")
    if search_window_size is not None and (not isinstance(search_window_size, int) or search_window_size <= 0):
        raise ValueError("search_window_size must be a positive integer.")
    if filter_strength is not None and (not isinstance(filter_strength, (int, float)) or filter_strength <= 0):
        raise ValueError("filter_strength must be a positive number.")
    if image.dtype != np.float32:
        raise ValueError("Image must be of dtype float32.")


###############################################################################
#                            Image Comparison UI                              #
###############################################################################

def handle_image_comparison(tab):
    with tab:
        st.header("Image Comparison")
        
        # Prepare comparison images
        comparison_images = {
            "Unprocessed Image": st.session_state.get("image_array", np.array([]))
        }

        for result_key in ["speckle_result", "nlm_result"]:
            if results := st.session_state.get(result_key):
                comparison_images.update(results.filter_data)

        if len(comparison_images) <= 1:
            st.warning("No images available for comparison.")
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
            
            diff_map = display_difference_map(img1, img2)
            
            # Add download buttons
            st.subheader("Download Images")
            col1, col2, col3 = st.columns(3)
            with col1:
                create_download_button(img1_uint8, f"{image_choice_1}.png")
            with col2:
                create_download_button(img2_uint8, f"{image_choice_2}.png")
            with col3:
                if diff_map is not None:
                    create_download_button(diff_map, "difference_map.png")
        else:
            st.error("Error processing images for comparison.")
    except Exception as e:
        st.error(f"Error in image comparison: {str(e)}")
        st.exception(e)

###############################################################################
#                           Image Download Utilities                          #
###############################################################################

def create_download_button(image, filename):
    buf = BytesIO()
    Image.fromarray(image).save(buf, format="PNG")
    st.download_button(
        label=f"Download {filename}",
        data=buf.getvalue(),
        file_name=filename,
        mime="image/png"
    )

###############################################################################
#                         Image Processing Utilities                          #
###############################################################################

def display_difference_map(img1, img2):
    try:
        diff_map = np.abs(img1 - img2)
        normalized_diff = normalize_and_colorize([diff_map])
        if normalized_diff:
            diff_map_uint8 = normalized_diff[0]
            st.image(diff_map_uint8, caption="Difference Map")
            return diff_map_uint8
        else:
            st.error("Error creating difference map.")
            return None
    except Exception as e:
        st.error(f"Error in creating difference map: {str(e)}")
        return None

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