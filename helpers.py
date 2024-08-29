import io
import time
import numpy as np
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from PIL import Image
import streamlit as st
from utils import calculate_statistics

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

def plot_zoomed_views(zoomed_data, titles, cmap):
    """Plot zoomed-in views with values annotated."""
    zoom_fig, zoom_axs = plt.subplots(1, len(zoomed_data), figsize=(20, 5))
    zoom_axs = zoom_axs if isinstance(zoom_axs, (list, np.ndarray)) else [zoom_axs]
    
    for ax, data, title in zip(zoom_axs, zoomed_data, titles):
        ax.imshow(data, cmap=cmap, vmin=0, vmax=1)
        ax.set_title(title, fontsize=12)
        ax.axis("off")
        for i, row in enumerate(data):
            for j, val in enumerate(row):
                ax.text(j, i, f"{val:.3f}", ha="center", va="center", color="red", fontsize=10)
    
    zoom_fig.tight_layout(pad=2)
    return zoom_fig

def display_speckle_contrast_formula(placeholder, x, y, std, mean, sc):
    """Display the formula for speckle contrast."""
    placeholder.latex(
        r"SC_{{({}, {})}} = \frac{{\sigma}}{{\mu}} = \frac{{{:.3f}}}{{{:.3f}}} = {:.3f}".format(x, y, std, mean, sc)
    )

# ---------------------------- Plotting Functions ---------------------------- #

def update_plot(fig, axs, image_np, filters, last_x, last_y, kernel_size, cmap="viridis"):
    """Update the plot with the original image and calculated filters."""
    clear_axes(axs)
    configure_axes(axs[0], "Original Image with Current Kernel", image_np, cmap=cmap, vmin=0, vmax=1)
    add_kernel_rectangle(axs[0], last_x, last_y, kernel_size)
    
    for ax, filter_data, title in zip(axs[1:], filters, ["Mean Filter", "Standard Deviation Filter", "Speckle Contrast"]):
        configure_axes(ax, title, filter_data, cmap=cmap)
    
    fig.tight_layout(pad=2)
    return fig

def _display_filter_and_zoomed_view(filter_data, last_x, last_y, stride, title, filter_placeholder, zoomed_placeholder, cmap):
    """Helper function to display filter and zoomed-in view."""
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    configure_axes(ax, title, filter_data, cmap=cmap)
    filter_placeholder.pyplot(fig)
    
    zoomed_placeholder.pyplot(
        plot_zoomed_views(
            [filter_data[last_y // stride : last_y // stride + 1, last_x // stride : last_x // stride + 1]],
            ["Zoomed-In " + title],
            cmap
        )
    )

# ---------------------------- Streamlit Interaction Functions ---------------------------- #

def toggle_animation():
    """Toggle the animation state in Streamlit."""
    st.session_state.is_animating = not st.session_state.is_animating

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
        mean_filter, std_dev_filter, sc_filter, last_x, last_y, last_mean, last_std, last_sc = calculate_statistics(
            image_np, kernel_size, stride, i, st.session_state.cache
        )

        # Display results
        display_speckle_contrast_formula(formula_placeholder, last_x, last_y, last_std, last_mean, last_sc)
        
        fig_original, axs_original = plt.subplots(1, 1, figsize=(5, 5))
        original_image_placeholder.pyplot(
            update_plot(fig_original, [axs_original], image_np, [mean_filter], last_x, last_y, kernel_size, cmap)
        )
        
        zoomed_kernel_placeholder.pyplot(
            plot_zoomed_views(
                [image_np[last_y : last_y + kernel_size, last_x : last_x + kernel_size]],
                ["Zoomed-In Kernel"],
                cmap
            )
        )
        
        _display_filter_and_zoomed_view(mean_filter, last_x, last_y, stride, "Mean Filter", mean_filter_placeholder, zoomed_mean_placeholder, cmap)
        _display_filter_and_zoomed_view(std_dev_filter, last_x, last_y, stride, "Standard Deviation Filter", std_dev_filter_placeholder, zoomed_std_placeholder, cmap)
        _display_filter_and_zoomed_view(sc_filter, last_x, last_y, stride, "Speckle Contrast", speckle_contrast_placeholder, zoomed_sc_placeholder, cmap)
        
        if not st.session_state.is_animating:
            break
        
        time.sleep(animation_speed)

    # Return the final images for use in other tabs
    return std_dev_filter, sc_filter, mean_filter
