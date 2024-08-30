
import streamlit as st
import matplotlib.pyplot as plt
import streamlit_nested_layout
import numpy as np
import time
# Project Structure:
# .
# ├── app.py               # Main application file, contains Streamlit app structure and high-level logic.
# ├── helpers.py           # Helper functions for plotting and displaying results.
# ├── speckle_lib          # Functions for speckle contrast and non-local means calculations.
# ├── ui_components.py     # Streamlit UI components and layouts.
# └── config.py            # Configuration variables and constants and image loading and processing.

from config import set_page_config, configure_sidebar,get_image_download_link
from ui_components import (
    create_section,
    display_speckle_contrast_formula,
    display_image_comparison,
    display_image_comparison_error)
from speckle_lib import handle_speckle_contrast_calculation
from helpers import display_speckle_contrast_process
# from nlm_lib import handle_non_local_means_tab

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
 
    image, kernel_size, search_window_size, filter_strength, stride, cmap, animation_speed, image_np = configure_sidebar()


    max_pixels = st.slider("Pixels to process", 1, image.width * image.height, image.width * image.height)
    tabs = st.tabs(["Speckle Contrast Calculation", "Non-Local Means Denoising", "Speckle Contrast Comparison"])

    # Tab 1
    std_dev_image, speckle_contrast_image, mean_image, image_np = handle_speckle_contrast_tab(tabs[0], image_np, kernel_size, stride, max_pixels, animation_speed, cmap)

    # Tab 2
    handle_non_local_means_tab(tabs[1], image_np, kernel_size, stride, search_window_size, filter_strength, max_pixels, animation_speed, cmap)

    # Tab 3
    handle_speckle_contrast_comparison_tab(tabs[2], cmap, std_dev_image, speckle_contrast_image, mean_image, image_np)


def save_results_section(std_dev_image, speckle_contrast_image, mean_image):
    with st.expander("Save Results"):
        if std_dev_image is not None and speckle_contrast_image is not None:
            get_image_download_link(std_dev_image, "std_dev_filter.png", "Download Std Dev Filter")
            get_image_download_link(speckle_contrast_image, "speckle_contrast.png", "Download Speckle Contrast Image")
            get_image_download_link(mean_image, "mean_filter.png", "Download Mean Filter")
        else:
            st.error("No results to save. Please generate images by running the analysis.")

def handle_speckle_contrast_tab(tab, image_np, kernel_size, stride, max_pixels, animation_speed, cmap):
    with tab:
        st.header("Speckle Contrast Formula", divider="rainbow")
        formula_placeholder = st.empty()
        display_speckle_contrast_formula(formula_placeholder)
        col1, col2 = st.columns(2)

        with col1:
            original_image_placeholder, zoomed_kernel_placeholder = create_section("Original Image with Current Kernel", expanded_main=True, expanded_zoomed=False)
            mean_filter_placeholder, zoomed_mean_placeholder = create_section("Mean Filter", expanded_main=False, expanded_zoomed=False)
        with col2:
            speckle_contrast_placeholder, zoomed_sc_placeholder = create_section("Speckle Contrast", expanded_main=True, expanded_zoomed=False)
            std_dev_filter_placeholder, zoomed_std_placeholder = create_section("Standard Deviation Filter", expanded_main=False, expanded_zoomed=False)
            
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
        return std_dev_image, speckle_contrast_image, mean_image, image_np  # Include the original image

def handle_speckle_contrast_comparison_tab(tab, cmap_name, std_dev_image, speckle_contrast_image, mean_image, original_image):
    with tab:
        st.header("Speckle Contrast Comparison")
        available_images = ['Unprocessed Image', 'Standard Deviation', 'Speckle Contrast', 'Mean Filter']  # Include unprocessed image
        col1, col2 = st.columns(2)
        with col1:
            image_choice_1 = st.selectbox('Select first image to compare:', [''] + available_images, index=0)
        with col2:
            image_choice_2 = st.selectbox('Select second image to compare:', [''] + available_images, index=0)
        # Convert cmap_name to the actual colormap function
        cmap = plt.get_cmap(cmap_name)

        if image_choice_1 and image_choice_2 and image_choice_1 != image_choice_2:
            img1, img2 = get_images_to_compare(image_choice_1, image_choice_2, std_dev_image, speckle_contrast_image, mean_image, original_image)
            display_image_comparison(img1, img2, image_choice_1, image_choice_2, cmap)
        else:
            display_image_comparison_error(image_choice_1, image_choice_2)

def get_images_to_compare(image_choice_1, image_choice_2, std_dev_image, speckle_contrast_image, mean_image, original_image):
    images_to_compare = {
        'Unprocessed Image': original_image,  # Include unprocessed image
        'Standard Deviation': std_dev_image,
        'Speckle Contrast': speckle_contrast_image,
        'Mean Filter': mean_image
    }
    return images_to_compare[image_choice_1], images_to_compare[image_choice_2]






def handle_non_local_means_tab(tab, image_np, kernel_size, stride, search_window_size, filter_strength, max_pixels, animation_speed, cmap):
    with tab:
        st.header("Non-Local Means Denoising", divider="rainbow")

        formula_placeholder = st.empty()
        display_nlm_formula(formula_placeholder, 0, 0, kernel_size, search_window_size, filter_strength)


        # save_results_section(weights_image, nlm_image)
        # display_nlm_process()
  
        # Return the images for use in the comparison tab
        return 

def display_nlm_formula(formula_placeholder, p_x, p_y, window_size, search_size, filter_strength):
    """Display the formula for Non-Local Means denoising for a specific pixel."""
    with formula_placeholder.container():
        st.latex(r'''
        u(p_{%d,%d}) = \frac{1}{C(p_{%d,%d})} \sum_{q \in S} v(q) \cdot f(p_{%d,%d}, q)
        ''' % (p_x, p_y, p_x, p_y, p_x, p_y))

        st.latex(r'''
        f(p_{%d,%d}, q) = \exp\left(-\frac{\|N(p_{%d,%d}) - N(q)\|^2_2}{(%f)^2}\right)
        ''' % (p_x, p_y, p_x, p_y, filter_strength))

        st.latex(r'''
        C(p_{%d,%d}) = \sum_{q \in S} f(p_{%d,%d}, q)
        ''' % (p_x, p_y, p_x, p_y))

        st.markdown(f"""
        Where:
        - $u(p_{{{p_x},{p_y}}})$ is the filtered value at pixel $({p_x}, {p_y})$
        - $v(q)$ is the original value of a pixel $q$ in the search window
        - $f(p_{{{p_x},{p_y}}}, q)$ is the weight function based on similarity between neighborhoods
        - $S$ is the search window of size {search_size}x{search_size} centered at $({p_x}, {p_y})$
        - $N(p)$ is the neighborhood of size {window_size}x{window_size} around pixel $p$
        - **$h$ (filtering parameter) = {filter_strength:.2f}**: Controls the decay of the exponential function. 
          A larger $h$ makes the weights less sensitive to differences between neighborhoods.
        - $C(p_{{{p_x},{p_y}}})$ is the normalizing factor
        """)



def display_weights_image_section():
    st.subheader("Weights Image")
    weights_image_placeholder = st.empty()
    return weights_image_placeholder

def display_nlm_section():
    st.subheader("Non-Local Means Result")
    nlm_image_placeholder = st.empty()
    return nlm_image_placeholder

def handle_nlm_calculation(max_pixels, image_np, kernel_size, stride, search_window_size, filter_strength,
                           original_image_placeholder, weights_image_placeholder, nlm_image_placeholder,
                           formula_placeholder, animation_speed, cmap):
    # Resize image if necessary
    if image_np.size > max_pixels:
        image_np = resize_image(image_np, max_pixels)

    # Display original image
    display_image(original_image_placeholder, image_np, "Original Image", cmap)

    # Initialize progress bar
    progress_bar = st.progress(0)

    # Initialize output images
    weights_image = np.zeros_like(image_np, dtype=np.float32)
    nlm_image = np.zeros_like(image_np, dtype=np.float32)

    # Perform NLM calculation
    total_pixels = image_np.shape[0] * image_np.shape[1]
    for i in range(0, image_np.shape[0] - kernel_size + 1, stride):
        for j in range(0, image_np.shape[1] - kernel_size + 1, stride):
            # Calculate weights and NLM value for the current pixel
            weights, nlm_value = calculate_nlm(image_np, i, j, kernel_size, search_window_size, filter_strength)
            
            # Update weights and NLM images
            weights_image[i:i+kernel_size, j:j+kernel_size] += weights
            nlm_image[i:i+kernel_size, j:j+kernel_size] += nlm_value
            
            # Update progress
            progress = (i * image_np.shape[1] + j) / total_pixels
            progress_bar.progress(progress)
            
            # Update displays
            if (i * image_np.shape[1] + j) % animation_speed == 0:
                display_image(weights_image_placeholder, weights_image, "Weights Image", cmap)
                display_image(nlm_image_placeholder, nlm_image, "NLM Image", cmap)
                display_nlm_formula(formula_placeholder, i, j, kernel_size, search_window_size, filter_strength)
                time.sleep(0.1)

    # Normalize and finalize images
    weights_image /= np.max(weights_image)
    nlm_image /= np.max(nlm_image)

    # Display final images
    display_image(weights_image_placeholder, weights_image, "Final Weights Image", cmap)
    display_image(nlm_image_placeholder, nlm_image, "Final NLM Image", cmap)

    return weights_image, nlm_image

def calculate_nlm(image, i, j, kernel_size, search_window_size, filter_strength):
    # Implement the NLM calculation for a single pixel
    # This is a placeholder and needs to be implemented
    return np.random.rand(kernel_size, kernel_size), np.random.rand(kernel_size, kernel_size)

def display_image(placeholder, image, title, cmap):
    placeholder.image(image, caption=title, use_column_width=True, clamp=True, channels="GRAY", output_format="PNG")


if __name__ == "__main__":
    main()

