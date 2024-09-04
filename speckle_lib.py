import numpy as np
import time
import streamlit as st
from helpers import update_plot, display_data_and_zoomed_view, display_kernel_view
from ui_components import create_section, save_results_section
from typing import Tuple, Dict, Optional

# ---------------------------- Core Calculations ---------------------------- #

def calculate_local_statistics(local_window: np.ndarray) -> Tuple[float, float, float]:
    """Calculate the mean, standard deviation, and speckle contrast for a local window."""
    local_mean = np.mean(local_window)
    local_std = np.std(local_window)
    speckle_contrast = local_std / local_mean if local_mean != 0 else 0
    return local_mean, local_std, speckle_contrast

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
        local_mean, local_std, speckle_contrast = calculate_local_statistics(local_window)

        mean_filter[row, col] = local_mean
        std_dev_filter[row, col] = local_std
        sc_filter[row, col] = speckle_contrast

    last_x, last_y = top_left_x, top_left_y
    return mean_filter, std_dev_filter, sc_filter, last_x, last_y, local_mean, local_std, speckle_contrast

# ---------------------------- Streamlit Handlers ---------------------------- #

def handle_speckle_contrast_calculation(
    max_pixels: int, image_np: np.ndarray, kernel_size: int, stride: int, 
    placeholders: Dict[str, st.empty], animation_speed: float, cmap: str, search_window: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    for i in range(1, max_pixels + 1) if st.session_state.is_animating else [max_pixels]:
        mean_filter, std_dev_filter, sc_filter, last_x, last_y, last_mean, last_std, last_sc = calculate_speckle(
            image_np, kernel_size, stride, i
        )

        update_all_displays(
            image_np, kernel_size, cmap, search_window,
            last_x, last_y, stride, mean_filter, std_dev_filter, sc_filter,
            last_std, last_mean, last_sc, placeholders
        )
        
        if not st.session_state.is_animating:
            break
        
        time.sleep(animation_speed)

    return std_dev_filter, sc_filter, mean_filter

def update_all_displays(
    image_np: np.ndarray, kernel_size: int, cmap: str, search_window: Optional[int],
    last_x: int, last_y: int, stride: int, mean_filter: np.ndarray, std_dev_filter: np.ndarray, sc_filter: np.ndarray,
    last_std: float, last_mean: float, last_sc: float, placeholders: Dict[str, st.empty]
):
    """Update all displays including formula and visualizations."""
    display_speckle_contrast_formula(placeholders['formula'], last_x, last_y, last_std, last_mean, last_sc)
    
    update_visualizations(
        image_np, kernel_size, cmap, search_window, 
        last_x, last_y, stride, mean_filter, std_dev_filter, sc_filter,
        placeholders
    )

def update_visualizations(
    image_np: np.ndarray, kernel_size: int, cmap: str, search_window: Optional[int], 
    last_x: int, last_y: int, stride: int, mean_filter: np.ndarray, std_dev_filter: np.ndarray, sc_filter: np.ndarray,
    placeholders: Dict[str, st.empty]
):
    """Update all the visualizations based on the current state."""
    fig_original = update_plot(
        image_np, [], last_x, last_y, kernel_size,
        ["Original Image with Current Kernel"], cmap=cmap, 
        search_window=search_window, figsize=(5, 5)
    )
    placeholders['original_image'].pyplot(fig_original)

    zoomed_kernel = image_np[last_y : last_y + kernel_size, last_x : last_x + kernel_size]
    display_kernel_view(zoomed_kernel, image_np, "Zoomed-In Kernel", placeholders['zoomed_kernel'], cmap)

    for filter_name, filter_data in [
        ("Mean Filter", mean_filter),
        ("Standard Deviation Filter", std_dev_filter),
        ("Speckle Contrast", sc_filter)
    ]:
        display_data_and_zoomed_view(
            filter_data, image_np, last_x, last_y, stride, filter_name,
            placeholders[f'{filter_name.lower().replace(" ", "_")}'],
            placeholders[f'zoomed_{filter_name.lower().replace(" ", "_")}'],
            cmap
        )

def handle_speckle_contrast_tab(tab: st.tabs, image_np: np.ndarray, kernel_size: int, stride: int, max_pixels: int, animation_speed: float, cmap: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with tab:
        st.header("Speckle Contrast Formula", divider="rainbow")
        placeholders = create_placeholders()
        
        std_dev_image, speckle_contrast_image, mean_image = handle_speckle_contrast_calculation(
            max_pixels, image_np, kernel_size, stride, 
            placeholders, animation_speed, cmap
        )
        save_results_section(std_dev_image, speckle_contrast_image, mean_image)

        return std_dev_image, speckle_contrast_image, mean_image, image_np

def create_placeholders() -> Dict[str, st.empty]:
    """Create and return a dictionary of all required placeholders."""
    placeholders = {
        'formula': st.empty(),
        'original_image': None,
        'zoomed_kernel': None,
        'mean_filter': None,
        'zoomed_mean_filter': None,
        'standard_deviation_filter': None,
        'zoomed_standard_deviation_filter': None,
        'speckle_contrast': None,
        'zoomed_speckle_contrast': None
    }
    
    col1, col2 = st.columns(2)
    with col1:
        placeholders['original_image'], placeholders['zoomed_kernel'] = create_section("Original Image with Current Kernel", expanded_main=True, expanded_zoomed=False)
        placeholders['mean_filter'], placeholders['zoomed_mean_filter'] = create_section("Mean Filter", expanded_main=False, expanded_zoomed=False)
    with col2:
        placeholders['speckle_contrast'], placeholders['zoomed_speckle_contrast'] = create_section("Speckle Contrast", expanded_main=True, expanded_zoomed=False)
        placeholders['standard_deviation_filter'], placeholders['zoomed_standard_deviation_filter'] = create_section("Standard Deviation Filter", expanded_main=False, expanded_zoomed=False)
    
    return placeholders

# ---------------------------- Information ---------------------------- #

def display_speckle_contrast_formula(formula_placeholder: st.empty, x: int, y: int, std: float, mean: float, sc: float):
    """Display the speckle contrast formula."""
    with formula_placeholder.container():
        st.latex(r'SC_{{({}, {})}} = \frac{{\sigma}}{{\mu}} = \frac{{{:.3f}}}{{{:.3f}}} = {:.3f}'.format(x, y, std, mean, sc))
