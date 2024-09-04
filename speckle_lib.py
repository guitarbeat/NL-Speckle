import numpy as np
import time
import streamlit as st
from helpers import create_plot, display_kernel_view
from typing import Tuple
import io
from PIL import Image
import matplotlib.pyplot as plt
# ---------------------------- Core Calculations ---------------------------- #

@st.cache_data
def calculate_speckle(image: np.ndarray, kernel_size: int, stride: int, max_pixels: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, int, float, float, float]:
    output_height = (image.shape[0] - kernel_size) // stride + 1
    output_width = (image.shape[1] - kernel_size) // stride + 1
    total_pixels = min(max_pixels, output_height * output_width)

    mean_filter = np.zeros((output_height, output_width))
    std_dev_filter = np.zeros((output_height, output_width))
    sc_filter = np.zeros((output_height, output_width))

    for pixel in range(total_pixels):
        row, col = divmod(pixel, output_width)
        top_left_y, top_left_x = row * stride, col * stride

        local_window = image[top_left_y:top_left_y + kernel_size, top_left_x:top_left_x + kernel_size]
        
        # Calculate local statistics
        local_mean = np.mean(local_window)
        local_std = np.std(local_window)
        speckle_contrast = local_std / local_mean if local_mean != 0 else 0

        mean_filter[row, col] = local_mean
        std_dev_filter[row, col] = local_std
        sc_filter[row, col] = speckle_contrast

    last_x, last_y = top_left_x, top_left_y
    return mean_filter, std_dev_filter, sc_filter, last_x, last_y, local_mean, local_std, speckle_contrast

# ---------------------------- Streamlit Handlers ---------------------------- #

def handle_image_analysis(tab: st.tabs, image_np: np.ndarray, kernel_size: int, stride: int, max_pixels: int, animation_speed: float, cmap: str, technique: str = "speckle") -> Tuple[np.ndarray, ...]:
    with tab:
        st.header(f"{technique.capitalize()} Analysis", divider="rainbow")
        
        # Create placeholders
        placeholders = create_placeholders(technique)
        
        # Create sections
        create_sections(placeholders, technique)
        
        # Calculation and visualization loop
        results = run_analysis_loop(image_np, kernel_size, stride, max_pixels, animation_speed, cmap, technique, placeholders)
        
        # Save Results Section
        create_save_section(results, technique)

        return results

def create_placeholders(technique: str) -> dict:
    placeholders = {
        'formula': st.empty(),
        'original_image': None,
        'zoomed_kernel': None,
    }
    if technique == "speckle":
        placeholders.update({
            'mean_filter': None,
            'zoomed_mean_filter': None,
            'standard_deviation_filter': None,
            'zoomed_standard_deviation_filter': None,
            'speckle_contrast': None,
            'zoomed_speckle_contrast': None
        })
    # Add placeholders for other techniques here
    return placeholders

def create_sections(placeholders: dict, technique: str):
    def create_section(title: str, expanded_main: bool = False, expanded_zoomed: bool = False):
        with st.expander(title, expanded=expanded_main):
            main_placeholder = st.empty()
            with st.expander(f"Zoomed-in {title.split()[0]}", expanded=expanded_zoomed):
                zoomed_placeholder = st.empty()
        return main_placeholder, zoomed_placeholder
    
    col1, col2 = st.columns(2)
    with col1:
        placeholders['original_image'], placeholders['zoomed_kernel'] = create_section("Original Image with Current Kernel", expanded_main=True, expanded_zoomed=False)
        if technique == "speckle":
            placeholders['mean_filter'], placeholders['zoomed_mean_filter'] = create_section("Mean Filter", expanded_main=False, expanded_zoomed=False)
    with col2:
        if technique == "speckle":
            placeholders['speckle_contrast'], placeholders['zoomed_speckle_contrast'] = create_section("Speckle Contrast", expanded_main=True, expanded_zoomed=False)
            placeholders['standard_deviation_filter'], placeholders['zoomed_standard_deviation_filter'] = create_section("Standard Deviation Filter", expanded_main=False, expanded_zoomed=False)
    # Add sections for other techniques here

def run_analysis_loop(image_np: np.ndarray, kernel_size: int, stride: int, max_pixels: int, animation_speed: float, cmap: str, technique: str, placeholders: dict) -> Tuple[np.ndarray, ...]:
    for i in range(1, max_pixels + 1) if st.session_state.is_animating else [max_pixels]:
        if technique == "speckle":
            results = calculate_speckle(image_np, kernel_size, stride, i)
            mean_filter, std_dev_filter, sc_filter, last_x, last_y, last_mean, last_std, last_sc = results
            display_speckle_contrast_formula(placeholders['formula'], last_x, last_y, last_std, last_mean, last_sc)
        # Add calculations for other techniques here
        
        update_visualizations(image_np, kernel_size, last_x, last_y, results, cmap, technique, placeholders, stride)
        
        if not st.session_state.is_animating:
            break
        
        time.sleep(animation_speed)
    
    return results

def update_visualizations(image_np: np.ndarray, kernel_size: int, last_x: int, last_y: int, results: Tuple[np.ndarray, ...], cmap: str, technique: str, placeholders: dict, stride: int):
    fig_original = create_plot(
        image_np, [], last_x, last_y, kernel_size,
        ["Original Image with Current Kernel"], cmap=cmap, 
        search_window=None, figsize=(5, 5)
    )
    placeholders['original_image'].pyplot(fig_original)

    zoomed_kernel = image_np[last_y : last_y + kernel_size, last_x : last_x + kernel_size]
    display_kernel_view(zoomed_kernel, image_np, "Zoomed-In Kernel", placeholders['zoomed_kernel'], cmap)

    if technique == "speckle":
        mean_filter, std_dev_filter, sc_filter = results[:3]
        for filter_name, filter_data in [
            ("Mean Filter", mean_filter),
            ("Standard Deviation Filter", std_dev_filter),
            ("Speckle Contrast", sc_filter)
        ]:
            display_filter(filter_name, filter_data, last_x, last_y, cmap, placeholders, stride)
    # Add visualizations for other techniques here

def display_filter(filter_name: str, filter_data: np.ndarray, last_x: int, last_y: int, cmap: str, placeholders: dict, stride: int):
    fig_full, ax_full = plt.subplots()
    ax_full.imshow(filter_data, cmap=cmap)
    ax_full.set_title(filter_name)
    ax_full.axis('off')
    placeholders[f'{filter_name.lower().replace(" ", "_")}'].pyplot(fig_full)
    
    zoom_size = 1
    zoomed_data = filter_data[last_y // stride : last_y // stride + zoom_size, last_x // stride : last_x // stride + zoom_size]
    fig_zoom, ax_zoom = plt.subplots()
    ax_zoom.imshow(zoomed_data, cmap=cmap)
    ax_zoom.set_title(f"Zoomed-In {filter_name}")
    ax_zoom.axis('off')
    for i, row in enumerate(zoomed_data):
        for j, val in enumerate(row):
            ax_zoom.text(j, i, f"{val:.2f}", ha="center", va="center", color="red", fontsize=10)
    fig_zoom.tight_layout(pad=2)
    placeholders[f'zoomed_{filter_name.lower().replace(" ", "_")}'].pyplot(fig_zoom)

def create_save_section(results: Tuple[np.ndarray, ...], technique: str):
    with st.expander("Save Results"):
        if technique == "speckle":
            std_dev_filter, sc_filter, mean_filter = results[:3]
            if std_dev_filter is not None and sc_filter is not None:
                for image, filename, button_text in [
                    (std_dev_filter, "std_dev_filter.png", "Download Std Dev Filter"),
                    (sc_filter, "speckle_contrast.png", "Download Speckle Contrast Image"),
                    (mean_filter, "mean_filter.png", "Download Mean Filter")
                ]:
                    create_download_button(image, filename, button_text)
            else:
                st.error("No results to save. Please generate images by running the analysis.")
        # Add save options for other techniques here

def create_download_button(image: np.ndarray, filename: str, button_text: str):
    img_buffer = io.BytesIO()
    Image.fromarray((255 * image).astype(np.uint8)).save(img_buffer, format='PNG')
    img_buffer.seek(0)
    st.download_button(label=button_text, data=img_buffer, file_name=filename, mime="image/png")

# ---------------------------- Information ---------------------------- #

def display_speckle_contrast_formula(formula_placeholder: st.empty, x: int, y: int, std: float, mean: float, sc: float):
    """Display the speckle contrast formula."""
    with formula_placeholder.container():
        st.latex(r'SC_{{({}, {})}} = \frac{{\sigma}}{{\mu}} = \frac{{{:.3f}}}{{{:.3f}}} = {:.3f}'.format(x, y, std, mean, sc))
