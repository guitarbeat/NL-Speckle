import streamlit as st
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

