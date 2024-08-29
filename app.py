
import streamlit as st
import matplotlib.pyplot as plt
# Project Structure:
# .
# ├── app.py               # Main application file, contains Streamlit app structure and high-level logic.
# ├── helpers.py           # Helper functions for plotting and displaying results.
# ├── speckle_lib          # Functions for speckle contrast and non-local means calculations.
# ├── ui_components.py     # Streamlit UI components and layouts.
# └── config.py            # Configuration variables and constants and image loading and processing.

from config import set_page_config, configure_sidebar,get_image_download_link
from ui_components import (
    display_original_image_section,
    display_speckle_contrast_formula,
    display_speckle_contrast_section,
    display_image_comparison,
    display_image_comparison_error)
from speckle_lib import handle_speckle_contrast_calculation
from helpers import display_speckle_contrast_process


set_page_config()

st.title("Speckle Contrast Analysis")
st.markdown("""
Welcome to the Interactive Speckle Contrast Analysis tool! This application allows you to visualize and analyze speckle contrast in images.

**How to use this tool:**
1. Choose an image source (preloaded or upload your own) in the sidebar.
2. Adjust the processing parameters and visualization settings.
3. Explore the speckle contrast analysis results in the tabs below.
4. Use the animation controls to see the analysis process in action.

Let's get started!
""")

def main():
    image, kernel_size, stride, cmap, animation_speed, image_np = configure_sidebar()
    max_pixels = st.slider("Pixels to process", 1, image.width * image.height, image.width * image.height)
    tabs = st.tabs(["Speckle Contrast Calculation", "Non-Local Means Method", "Speckle Contrast Comparison"])

    std_dev_image, speckle_contrast_image, mean_image = handle_speckle_contrast_tab(tabs[0], image_np, kernel_size, stride, max_pixels, animation_speed, cmap)
    handle_non_local_means_tab(tabs[1])
    handle_speckle_contrast_comparison_tab(tabs[2], cmap, std_dev_image, speckle_contrast_image, mean_image)

def handle_speckle_contrast_tab(tab, image_np, kernel_size, stride, max_pixels, animation_speed, cmap):
    with tab:
        st.header("Speckle Contrast Formula", divider="rainbow")
        formula_placeholder = st.empty()
        display_speckle_contrast_formula(formula_placeholder)
        col1, col2 = st.columns(2)

        with col1:
            original_image_placeholder, mean_filter_placeholder, zoomed_kernel_placeholder, zoomed_mean_placeholder = display_original_image_section()
        with col2:
            speckle_contrast_placeholder, std_dev_filter_placeholder, zoomed_sc_placeholder, zoomed_std_placeholder = display_speckle_contrast_section()

        std_dev_image, speckle_contrast_image, mean_image = handle_speckle_contrast_calculation(
            max_pixels, image_np, kernel_size, stride, 
            original_image_placeholder, mean_filter_placeholder, 
            std_dev_filter_placeholder, speckle_contrast_placeholder, 
            zoomed_kernel_placeholder, zoomed_mean_placeholder, 
            zoomed_std_placeholder, zoomed_sc_placeholder, 
            formula_placeholder, animation_speed, cmap
        )
        save_results_section(std_dev_image, speckle_contrast_image, mean_image)
        display_speckle_contrast_process()

        # Return the images for use in the comparison tab
        return std_dev_image, speckle_contrast_image, mean_image

def save_results_section(std_dev_image, speckle_contrast_image, mean_image):
    with st.expander("Save Results"):
        if std_dev_image is not None and speckle_contrast_image is not None:
            get_image_download_link(std_dev_image, "std_dev_filter.png", "Download Std Dev Filter")
            get_image_download_link(speckle_contrast_image, "speckle_contrast.png", "Download Speckle Contrast Image")
            get_image_download_link(mean_image, "mean_filter.png", "Download Mean Filter")
        else:
            st.error("No results to save. Please generate images by running the analysis.")

def handle_non_local_means_tab(tab):
    with tab:
        st.header("Non-Local Means Method")
        st.write("Coming soon...")

def handle_speckle_contrast_comparison_tab(tab, cmap_name, std_dev_image, speckle_contrast_image, mean_image):
    with tab:
        st.header("Speckle Contrast Comparison")
        available_images = ['Standard Deviation', 'Speckle Contrast', 'Mean Filter']
        col1, col2 = st.columns(2)
        with col1:
            image_choice_1 = st.selectbox('Select first image to compare:', [''] + available_images, index=0)
        with col2:
            image_choice_2 = st.selectbox('Select second image to compare:', [''] + available_images, index=0)
        # Convert cmap_name to the actual colormap function
        cmap = plt.get_cmap(cmap_name)

        if image_choice_1 and image_choice_2 and image_choice_1 != image_choice_2:
            img1, img2 = get_images_to_compare(image_choice_1, image_choice_2, std_dev_image, speckle_contrast_image, mean_image)
            display_image_comparison(img1, img2, image_choice_1, image_choice_2, cmap)
        else:
            display_image_comparison_error(image_choice_1, image_choice_2)

def get_images_to_compare(image_choice_1, image_choice_2, std_dev_image, speckle_contrast_image, mean_image):
    images_to_compare = {
        'Standard Deviation': std_dev_image,
        'Speckle Contrast': speckle_contrast_image,
        'Mean Filter': mean_image
    }
    return images_to_compare[image_choice_1], images_to_compare[image_choice_2]

if __name__ == "__main__":
    main()
