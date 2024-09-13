import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from streamlit_image_comparison import image_comparison

# TODO: Consider splitting this function into smaller, more focused functions
def calculate_processing_details(image, kernel_size, max_pixels=None):
    """
    Calculate processing details for kernel-based image processing algorithms.
    """
    height, width = image.shape
    half_kernel = kernel_size // 2
    valid_height = height - kernel_size + 1
    valid_width = width - kernel_size + 1
    total_valid_pixels = valid_height * valid_width
    pixels_to_process = total_valid_pixels if max_pixels is None else min(max_pixels, total_valid_pixels)

    first_x = first_y = half_kernel
    
    last_pixel = pixels_to_process - 1
    last_y = (last_pixel // valid_width) + half_kernel
    last_x = (last_pixel % valid_width) + half_kernel

    # TODO: Consider using a dataclass or named tuple for better structure
    return {
        'height': height,
        'width': width,
        'first_x': first_x,
        'first_y': first_y,
        'last_x': int(last_x),
        'last_y': int(last_y),
        'pixels_to_process': pixels_to_process,
        'valid_height': valid_height,
        'valid_width': valid_width
    }


def prepare_comparison_images():
    """
    Prepare a dictionary of images for comparison visualization.
    """
    speckle_results = st.session_state.get("speckle_results")
    nlm_results = st.session_state.get("nlm_results")
    analysis_params = st.session_state.analysis_params

    if speckle_results is not None and nlm_results is not None:
        return {
            'Unprocessed Image': analysis_params['image_np'],
            'Standard Deviation': speckle_results.std_dev_filter,
            'Speckle Contrast': speckle_results.speckle_contrast_filter,
            'Mean Filter': speckle_results.mean_filter,
            'NL-Means Image': nlm_results.processed_image
        }
    else:
        return None

# TODO: Consider splitting this function into UI and logic components
def handle_image_comparison(tab, cmap_name, images):
    """
    Handle the image comparison UI and functionality.
    """
    with tab:
        st.header("Image Comparison")
        if not images:
            st.warning("No images available for comparison.")
            return
        
        available_images = list(images.keys())
        col1, col2 = st.columns(2)
        image_choice_1 = col1.selectbox('Select first image to compare:', [''] + available_images, index=0)
        image_choice_2 = col2.selectbox('Select second image to compare:', [''] + available_images, index=0)
        
        if image_choice_1 and image_choice_2:
            img1, img2 = images[image_choice_1], images[image_choice_2]
            
            # TODO: Extract this image normalization logic into a separate function
            img1_uint8, img2_uint8 = map(lambda img: (plt.get_cmap(cmap_name)((img - np.min(img)) / (np.max(img) - np.min(img)))[:, :, :3] * 255).astype(np.uint8), [img1, img2])
            
            if image_choice_1 != image_choice_2:
                image_comparison(img1=img1_uint8, img2=img2_uint8, label1=image_choice_1, label2=image_choice_2, make_responsive=True)
                st.subheader("Selected Images")
                st.image([img1_uint8, img2_uint8], caption=[image_choice_1, image_choice_2])
            else:
                st.error("Please select two different images for comparison.")
                st.image(np.abs(img1 - img2), caption="Difference Map", use_column_width=True)
        else:
            st.info("Select two images to compare.")