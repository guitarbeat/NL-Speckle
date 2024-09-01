import numpy as np
import time
import streamlit as st
from helpers import update_plot, display_data_and_zoomed_view, display_kernel_view
from ui_components import create_section, save_results_section

# ---------------------------- Core Calculations ---------------------------- #
import numpy as np

def calculate_local_statistics(local_window: np.ndarray) -> tuple:
    """Calculate the mean, standard deviation, and speckle contrast for a local window."""
    local_mean = np.mean(local_window)
    local_std = np.std(local_window)
    speckle_contrast = local_std / local_mean if local_mean != 0 else 0
    return local_mean, local_std, speckle_contrast

def calculate_speckle(image: np.ndarray, kernel_size: int, stride: int, max_pixels: int, cache: dict) -> tuple:
    """Calculate the mean, standard deviation, and speckle contrast for an image."""

    output_height = (image.shape[0] - kernel_size) // stride + 1
    output_width = (image.shape[1] - kernel_size) // stride + 1

    total_pixels = min(max_pixels, output_height * output_width)

    mean_filter = np.zeros((output_height, output_width))
    std_dev_filter = np.zeros((output_height, output_width))
    sc_filter = np.zeros((output_height, output_width))

    cache_key = (image.shape, kernel_size, stride)
    if cache_key not in cache:
        cache[cache_key] = {}

    for pixel in range(total_pixels):
        row, col = divmod(pixel, output_width)
        top_left_y, top_left_x = row * stride, col * stride

        if (row, col) in cache[cache_key]:
            local_mean, local_std, speckle_contrast = cache[cache_key][(row, col)]
        else:
            local_window = image[top_left_y:top_left_y + kernel_size, top_left_x:top_left_x + kernel_size]
            local_mean, local_std, speckle_contrast = calculate_local_statistics(local_window)
            cache[cache_key][(row, col)] = (local_mean, local_std, speckle_contrast)

        mean_filter[row, col] = local_mean
        std_dev_filter[row, col] = local_std
        sc_filter[row, col] = speckle_contrast

    last_x, last_y = top_left_x, top_left_y
    return mean_filter, std_dev_filter, sc_filter, last_x, last_y, local_mean, local_std, speckle_contrast

# ---------------------------- Streamlit Handlers ---------------------------- #

def handle_speckle_contrast_calculation(
    max_pixels, image_np, kernel_size, stride, 
    original_image_placeholder, mean_filter_placeholder, 
    std_dev_filter_placeholder, speckle_contrast_placeholder, 
    zoomed_kernel_placeholder, zoomed_mean_placeholder, 
    zoomed_std_placeholder, zoomed_sc_placeholder, 
    formula_placeholder, animation_speed, cmap, search_window=None
):
    """Handle the speckle contrast calculation and update Streamlit placeholders."""
    for i in range(1, max_pixels + 1) if st.session_state.is_animating else [max_pixels]:
        # Calculate statistics
        mean_filter, std_dev_filter, sc_filter, last_x, last_y, last_mean, last_std, last_sc = calculate_speckle(
            image_np, kernel_size, stride, i, st.session_state.cache
        )

        # Display results
        display_speckle_contrast_formula(formula_placeholder, last_x, last_y, last_std, last_mean, last_sc)

        # Update visualizations
        update_visualizations(
            image_np, kernel_size, cmap, search_window, 
            last_x, last_y, stride, mean_filter, std_dev_filter, sc_filter,
            original_image_placeholder, zoomed_kernel_placeholder,
            mean_filter_placeholder, zoomed_mean_placeholder,
            std_dev_filter_placeholder, zoomed_std_placeholder,
            speckle_contrast_placeholder, zoomed_sc_placeholder
        )
        
        if not st.session_state.is_animating:
            break
        
        time.sleep(animation_speed)

    # Return the final images for use in other tabs
    return std_dev_filter, sc_filter, mean_filter

def update_visualizations(
    image_np, kernel_size, cmap, search_window, 
    last_x, last_y, stride, mean_filter, std_dev_filter, sc_filter,
    original_image_placeholder, zoomed_kernel_placeholder,
    mean_filter_placeholder, zoomed_mean_placeholder,
    std_dev_filter_placeholder, zoomed_std_placeholder,
    speckle_contrast_placeholder, zoomed_sc_placeholder
):
    """Update all the visualizations based on the current state."""
    fig_original = update_plot(
        image_np, [], last_x, last_y, kernel_size,
        ["Original Image with Current Kernel"], cmap=cmap, 
        search_window=search_window, figsize=(5, 5)
    )
    original_image_placeholder.pyplot(fig_original)

    zoomed_kernel = image_np[last_y : last_y + kernel_size, last_x : last_x + kernel_size]
    display_kernel_view(zoomed_kernel, image_np, "Zoomed-In Kernel", zoomed_kernel_placeholder, cmap)

    display_data_and_zoomed_view(mean_filter, image_np, last_x, last_y, stride, "Mean Filter", mean_filter_placeholder, zoomed_mean_placeholder, cmap)
    display_data_and_zoomed_view(std_dev_filter, image_np, last_x, last_y, stride, "Standard Deviation Filter", std_dev_filter_placeholder, zoomed_std_placeholder, cmap)
    display_data_and_zoomed_view(sc_filter, image_np, last_x, last_y, stride, "Speckle Contrast", speckle_contrast_placeholder, zoomed_sc_placeholder, cmap)

def handle_speckle_contrast_tab(tab, image_np, kernel_size, stride, max_pixels, animation_speed, cmap):
    with tab:
        st.header("Speckle Contrast Formula", divider="rainbow")
        formula_placeholder = st.empty()
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

        return std_dev_image, speckle_contrast_image, mean_image, image_np
    
# ---------------------------- Information ---------------------------- #

def display_speckle_contrast_formula(formula_placeholder, x, y, std, mean, sc):
    """Display the speckle contrast formula."""
    with formula_placeholder.container():
        st.latex(r'SC_{{({}, {})}} = \frac{{\sigma}}{{\mu}} = \frac{{{:.3f}}}{{{:.3f}}} = {:.3f}'.format(x, y, std, mean, sc))

def display_speckle_contrast_process():
    """Display information about the speckle contrast calculation process."""
    with st.expander("View Speckle Contrast Calculation Process", expanded=False):
        st.markdown("### Speckle Contrast Calculation Process")
        st.markdown("""
        1. **Sliding Window (Kernel) Approach**: Analyze the image using a sliding window.
        2. **Moving the Kernel**: The stride parameter determines the step size.
        3. **Local Statistics Calculation**: For each kernel position, calculate local statistics.
        4. **Understanding the Function**:
            - `local_window`: The current image section under the kernel.
            - `local_mean`: The average pixel intensity.
            - `local_std`: The standard deviation of pixel intensities.
            - `speckle_contrast`: Calculated as the ratio of standard deviation to mean.
        5. **Building the Speckle Contrast Image**: Generate images for mean, standard deviation, and speckle contrast.
        6. **Visualization**: Display the images in the expandable sections above.
        """)
        st.code('''
    def calculate_local_statistics(local_window: np.ndarray) -> tuple:
        local_mean = np.mean(local_window)
        local_std = np.std(local_window)
        speckle_contrast = local_std / local_mean if local_mean != 0 else 0
        return local_mean, local_std, speckle_contrast
        ''', language="python")
        st.markdown("### How It's Used in the Main Calculation")
        st.code('''
    def calculate_speckle(image: np.ndarray, kernel_size: int, stride: int, max_pixels: int, cache: dict) -> tuple:
        for pixel in range(total_pixels):
            row, col = divmod(pixel, output_width)
            local_mean, local_std, speckle_contrast = get_cached_statistics(row, col, cache_key, cache, image, kernel_size, stride)
        ''', language="python")
        st.markdown("""
        This snippet shows the main calculation loop, extracting the local window, passing it to `calculate_local_statistics`, and storing the results in the output images.
        """)

