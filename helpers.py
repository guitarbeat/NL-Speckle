import matplotlib.pyplot as plt
import matplotlib.patches as patches
import streamlit as st
import time
from utils import calculate_statistics
from streamlit_image_comparison import image_comparison
from PIL import Image
import io

def clear_axes(axs):
    """Clear all axes in the provided list."""
    for ax in axs:
        ax.clear()

def add_kernel_rectangle(ax, last_x, last_y, kernel_size):
    """Add a rectangle to highlight the kernel on the given axis."""
    rect = patches.Rectangle(
        (last_x - 0.5, last_y - 0.5),
        kernel_size,
        kernel_size,
        linewidth=2, edgecolor='r', facecolor='none'
    )
    ax.add_patch(rect)

def configure_axes(ax, title, image=None, cmap='viridis', show_colorbar=False, vmin=None, vmax=None):
    """Configure the axis with an image and a title."""
    if image is not None:
        ax.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.axis('off')
    if show_colorbar:
        plt.colorbar(ax=ax, fraction=0.046, pad=0.04)

def update_plot(fig, axs, image_np, filters, last_x, last_y, kernel_size, cmap='viridis'):
    """Update the plot with the original image and filter outputs."""
    clear_axes(axs)
    
    configure_axes(axs[0], 'Original Image with Current Kernel', image_np, cmap=cmap, vmin=0, vmax=1)
    add_kernel_rectangle(axs[0], last_x, last_y, kernel_size)

    filter_titles = ['Mean Filter', 'Standard Deviation Filter', 'Speckle Contrast']
    for ax, filter_data, title in zip(axs[1:], filters, filter_titles):
        configure_axes(ax, title, filter_data, cmap=cmap)
    
    fig.tight_layout(pad=2)
    return fig

def toggle_animation():
    """Toggle the animation state in Streamlit's session state."""
    st.session_state.is_animating = not st.session_state.is_animating

def plot_zoomed_views(zoomed_data, titles, cmap):
    """Plot zoomed-in views of the kernel and filter outputs."""
    num_plots = len(zoomed_data)
    zoom_fig, zoom_axs = plt.subplots(1, num_plots, figsize=(20, 5))
    
    # Ensure zoom_axs is always a list, even if there's only one plot
    if num_plots == 1:
        zoom_axs = [zoom_axs]

    for ax, data, title in zip(zoom_axs, zoomed_data, titles):
        ax.imshow(data, cmap=cmap, vmin=0, vmax=1)
        ax.set_title(title, fontsize=12)
        ax.axis('off')

        # Display pixel values on the image
        num_rows, num_cols = data.shape
        for i in range(num_rows):
            for j in range(num_cols):
                pixel_value = data[i, j]
                ax.text(j, i, f"{pixel_value:.3f}", ha="center", va="center", color="red", fontsize=10)

    zoom_fig.tight_layout(pad=2)
    return zoom_fig



def display_speckle_contrast_formula(placeholder, x, y, std, mean, sc):
    """Display the speckle contrast formula in a Streamlit placeholder."""
    placeholder.latex(
        r'SC_{{({}, {})}} = \frac{{\sigma}}{{\mu}} = \frac{{{:.3f}}}{{{:.3f}}} = {:.3f}'.format(x, y, std, mean, sc)
    )


def handle_speckle_contrast_calculation(max_pixels, image_np, kernel_size, stride, 
                                        original_image_placeholder, mean_filter_placeholder, std_dev_filter_placeholder, speckle_contrast_placeholder, 
                                        zoomed_kernel_placeholder, zoomed_mean_placeholder, zoomed_std_placeholder, zoomed_sc_placeholder, 
                                        formula_placeholder, animation_speed, cmap):
    """Handle the main speckle contrast calculation and display."""
    for i in range(1, max_pixels + 1) if st.session_state.is_animating else [max_pixels]:
        mean_filter, std_dev_filter, sc_filter, last_x, last_y, last_mean, last_std, last_sc = calculate_statistics(
            image_np, kernel_size, stride, i, st.session_state.cache
        )

        display_speckle_contrast_formula(formula_placeholder, last_x, last_y, last_std, last_mean, last_sc)

        # Original Image
        fig_original, axs_original = plt.subplots(1, 1, figsize=(5, 5))
        fig_original = update_plot(fig_original, [axs_original], image_np, [mean_filter], last_x, last_y, kernel_size, cmap=cmap)
        original_image_placeholder.pyplot(fig_original)

        # Zoomed Kernel
        zoomed_kernel = image_np[last_y:last_y + kernel_size, last_x:last_x + kernel_size]
        zoomed_fig_kernel = plot_zoomed_views([zoomed_kernel], ['Zoomed-In Kernel'], cmap)
        zoomed_kernel_placeholder.pyplot(zoomed_fig_kernel)

        # Mean Filter
        fig_mean, axs_mean = plt.subplots(1, 1, figsize=(5, 5))
        configure_axes(axs_mean, 'Mean Filter', mean_filter, cmap=cmap)
        mean_filter_placeholder.pyplot(fig_mean)

        # Zoomed Mean
        zoomed_mean = mean_filter[last_y // stride:last_y // stride + 1, last_x // stride:last_x // stride + 1]
        zoomed_fig_mean = plot_zoomed_views([zoomed_mean], ['Zoomed-In Mean'], cmap)
        zoomed_mean_placeholder.pyplot(zoomed_fig_mean)

        # Standard Deviation Filter
        fig_std, axs_std = plt.subplots(1, 1, figsize=(5, 5))
        configure_axes(axs_std, 'Standard Deviation Filter', std_dev_filter, cmap=cmap)
        std_dev_filter_placeholder.pyplot(fig_std)

        # Zoomed Std Dev
        zoomed_std = std_dev_filter[last_y // stride:last_y // stride + 1, last_x // stride:last_x // stride + 1]
        zoomed_fig_std = plot_zoomed_views([zoomed_std], ['Zoomed-In Std Dev'], cmap)
        zoomed_std_placeholder.pyplot(zoomed_fig_std)

        # Speckle Contrast
        fig_sc, axs_sc = plt.subplots(1, 1, figsize=(5, 5))
        configure_axes(axs_sc, 'Speckle Contrast', sc_filter, cmap=cmap)
        speckle_contrast_placeholder.pyplot(fig_sc)

        # Zoomed Speckle Contrast
        zoomed_sc = sc_filter[last_y // stride:last_y // stride + 1, last_x // stride:last_x // stride + 1]
        zoomed_fig_sc = plot_zoomed_views([zoomed_sc], ['Zoomed-In Speckle Contrast'], cmap)
        zoomed_sc_placeholder.pyplot(zoomed_fig_sc)

        if not st.session_state.is_animating:
            break

        time.sleep(animation_speed)


def create_comparison_tab(sc_filter, cmap):
    st.header("Speckle Contrast Comparison")
    
    # Create the original speckle contrast image
    fig, ax = plt.subplots()
    im = ax.imshow(sc_filter, cmap=cmap)
    plt.colorbar(im)
    ax.axis('off')
    
    # Save the original image to a byte stream
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    original_image = Image.open(buf)
    
    # Create the inverted speckle contrast image
    fig, ax = plt.subplots()
    im = ax.imshow(1 - sc_filter, cmap=cmap)
    plt.colorbar(im)
    ax.axis('off')
    
    # Save the inverted image to a byte stream
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    inverted_image = Image.open(buf)
    
    # Use the image_comparison component
    image_comparison(
        img1=original_image,
        img2=inverted_image,
        label1="Original Speckle Contrast",
        label2="Inverted Speckle Contrast",
        width=700,
        starting_position=50,
        show_labels=True,
        make_responsive=True,
        in_memory=True,
    )