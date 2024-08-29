import matplotlib.patches as patches
import matplotlib.pyplot as plt
import streamlit as st

# ---------------------------- Utility Functions ---------------------------- #

def clear_axes(axs):
    """Clear all axes."""
    for ax in axs:
        ax.clear()

def configure_axes(ax, title, image=None, cmap="viridis", show_colorbar=False, vmin=None, vmax=None):
    """Configure individual axes with image and title."""
    if image is not None:
        ax.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.axis("off")
    if show_colorbar:
        plt.colorbar(ax=ax, fraction=0.046, pad=0.04)

def add_kernel_rectangle(ax, last_x, last_y, kernel_size):
    """Add a rectangle to the axes to visualize the kernel area."""
    ax.add_patch(patches.Rectangle(
        (last_x - 0.5, last_y - 0.5),
        kernel_size, kernel_size,
        linewidth=2, edgecolor="r",
        facecolor="none"
    ))


def display_speckle_contrast_process():
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
    def calculate_statistics(image: np.ndarray, kernel_size: int, stride: int, max_pixels: int, cache: dict) -> tuple:
        for pixel in range(total_pixels):
            row, col = divmod(pixel, output_width)
            top_left_y, top_left_x = row * stride, col * stride
            local_window = image[top_left_y:top_left_y + kernel_size, top_left_x:top_left_x + kernel_size]
            local_mean, local_std, speckle_contrast = calculate_local_statistics(local_window)
        ''', language="python")
        st.markdown("""
        This snippet shows the main calculation loop, extracting the local window, passing it to `calculate_local_statistics`, and storing the results in the output images.
        """)
