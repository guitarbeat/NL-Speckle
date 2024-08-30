import numpy as np
import time
import matplotlib.pyplot as plt
from helpers import clear_axes, configure_axes, add_kernel_rectangle
import streamlit as st

# ---------------------------- Library Functions ---------------------------- #

def calculate_local_speckle(local_window: np.ndarray) -> tuple:
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
            local_mean, local_std, speckle_contrast = calculate_local_speckle(local_window)
            cache[cache_key][(row, col)] = (local_mean, local_std, speckle_contrast)

        mean_filter[row, col] = local_mean
        std_dev_filter[row, col] = local_std
        sc_filter[row, col] = speckle_contrast

    last_x, last_y = top_left_x, top_left_y
    return mean_filter, std_dev_filter, sc_filter, last_x, last_y, local_mean, local_std, speckle_contrast


def display_speckle_contrast_formula(placeholder, x, y, std, mean, sc):
    """Display the formula for speckle contrast."""
    placeholder.latex(
        r"SC_{{({}, {})}} = \frac{{\sigma}}{{\mu}} = \frac{{{:.3f}}}{{{:.3f}}} = {:.3f}".format(x, y, std, mean, sc)
    )

def handle_speckle_contrast_calculation(
    max_pixels, image_np, kernel_size, stride, 
    original_image_placeholder, mean_filter_placeholder, 
    std_dev_filter_placeholder, speckle_contrast_placeholder, 
    zoomed_kernel_placeholder, zoomed_mean_placeholder, 
    zoomed_std_placeholder, zoomed_sc_placeholder, 
    formula_placeholder, animation_speed, cmap
):
    """Handle the speckle contrast calculation and update Streamlit placeholders."""
    for i in range(1, max_pixels + 1) if st.session_state.is_animating else [max_pixels]:
        # Calculate statistics
        mean_filter, std_dev_filter, sc_filter, last_x, last_y, last_mean, last_std, last_sc = calculate_speckle(
            image_np, kernel_size, stride, i, st.session_state.cache
        )

        # Display results
        display_speckle_contrast_formula(formula_placeholder, last_x, last_y, last_std, last_mean, last_sc)
        

        fig_original, axs_original = plt.subplots(1, 1, figsize=(5, 5))
        original_image_placeholder.pyplot(
            update_plot(fig_original, [axs_original], image_np, [mean_filter], last_x, last_y, kernel_size, ["Original Image", "Mean Filter"], cmap)
        )

        zoomed_kernel_placeholder.pyplot(
            plot_zoomed_views(
                [image_np[last_y : last_y + kernel_size, last_x : last_x + kernel_size]],
                ["Zoomed-In Kernel"],
                cmap
            )
        )
        
        # display_filter_and_zoomed_view(mean_filter, last_x, last_y, stride, "Mean Filter", mean_filter_placeholder, zoomed_mean_placeholder, cmap)
        display_data_and_zoomed_view(mean_filter, last_x, last_y, stride, "Mean Filter", mean_filter_placeholder, zoomed_mean_placeholder, cmap)

        # display_filter_and_zoomed_view(std_dev_filter, last_x, last_y, stride, "Standard Deviation Filter", std_dev_filter_placeholder, zoomed_std_placeholder, cmap)
        display_data_and_zoomed_view(std_dev_filter, last_x, last_y, stride, "Standard Deviation Filter", std_dev_filter_placeholder, zoomed_std_placeholder, cmap)

        # display_filter_and_zoomed_view(sc_filter, last_x, last_y, stride, "Speckle Contrast", speckle_contrast_placeholder, zoomed_sc_placeholder, cmap)
        display_data_and_zoomed_view(sc_filter, last_x, last_y, stride, "Speckle Contrast", speckle_contrast_placeholder, zoomed_sc_placeholder, cmap)
        
        if not st.session_state.is_animating:
            break
        
        time.sleep(animation_speed)

    # Return the final images for use in other tabs
    return std_dev_filter, sc_filter, mean_filter

# ---------------------------- Plotting Functions ---------------------------- #

def display_data_and_zoomed_view(data, last_x, last_y, stride, title, data_placeholder, zoomed_placeholder, cmap="viridis", zoom_size=1):
    """
    Display data and its zoomed-in view.
    
    Parameters:
    - data: 2D numpy array of the data to display
    - last_x, last_y: Coordinates of the last processed point
    - stride: Step size between processed points
    - title: Title for the plot
    - data_placeholder: Streamlit placeholder for the full data plot
    - zoomed_placeholder: Streamlit placeholder for the zoomed view
    - cmap: Colormap to use (default: "viridis")
    - zoom_size: Size of the zoomed area (default: 1)
    """
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.imshow(data, cmap=cmap)
    ax.set_title(title)
    ax.axis("off")
    data_placeholder.pyplot(fig)
    
    zoomed_data = data[last_y // stride : last_y // stride + zoom_size, 
                       last_x // stride : last_x // stride + zoom_size]
    zoomed_placeholder.pyplot(
        plot_zoomed_views(
            [zoomed_data],
            ["Zoomed-In " + title],
            cmap
        )
    )

def update_plot(fig, axs, main_image, overlays, last_x, last_y, kernel_size, titles, cmap="viridis"):
    """
    Update a plot with multiple subplots.
    
    Parameters:
    - fig: matplotlib figure object
    - axs: list of matplotlib axes objects
    - main_image: 2D numpy array of the main image
    - overlays: list of 2D numpy arrays for additional overlays
    - last_x, last_y: Coordinates of the last processed point
    - kernel_size: Size of the processing kernel
    - titles: list of titles for each subplot
    - cmaps: list of colormaps for each subplot (optional)
    """

    
    clear_axes(axs)
    
    configure_axes(axs[0], "Original Image with Current Kernel", main_image, cmap=cmap, vmin=0, vmax=1)
    add_kernel_rectangle(axs[0], last_x, last_y, kernel_size)
    
    for ax, filter_data, title in zip(axs[1:], overlays, titles[1:]):
        configure_axes(ax, title, filter_data, cmap=cmap)

    fig.tight_layout(pad=2)
    return fig



def plot_zoomed_views(zoomed_data, titles, cmap="viridis", fontsize=10, text_color="red"):
    """
    Plot zoomed-in views with values annotated.
    
    Parameters:
    - zoomed_data: list of 2D numpy arrays to display
    - titles: list of titles for each zoomed view
    - cmap: colormap to use (default: "viridis")
    - fontsize: font size for annotations (default: 10)
    - text_color: color of the annotations (default: "red")
    """
    zoom_fig, zoom_axs = plt.subplots(1, len(zoomed_data), figsize=(5 * len(zoomed_data), 5))
    zoom_axs = zoom_axs if isinstance(zoom_axs, np.ndarray) else [zoom_axs]
    
    for ax, data, title in zip(zoom_axs, zoomed_data, titles):
        im = ax.imshow(data, cmap=cmap)
        ax.set_title(title, fontsize=12)
        ax.axis("off")
        plt.colorbar(im, ax=ax)
        for i, row in enumerate(data):
            for j, val in enumerate(row):
                ax.text(j, i, f"{val:.3f}", ha="center", va="center", color=text_color, fontsize=fontsize)
    
    zoom_fig.tight_layout(pad=2)
    return zoom_fig

def add_kernel_rectangle(ax, x, y, size, color='red', linewidth=2):
    """
    Add a rectangle to represent the kernel on the plot.
    
    Parameters:
    - ax: matplotlib axis object
    - x, y: Coordinates of the top-left corner of the rectangle
    - size: Size of the rectangle
    - color: Color of the rectangle (default: 'red')
    - linewidth: Width of the rectangle's border (default: 2)
    """
    from matplotlib.patches import Rectangle
    rect = Rectangle((x, y), size, size, fill=False, edgecolor=color, linewidth=linewidth)
    ax.add_patch(rect)