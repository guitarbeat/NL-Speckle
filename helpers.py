import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

#------------------- Utility Functions -------------------#

def clip_indices(index, min_val, max_val):
    """Clip indices to ensure they stay within valid ranges."""
    return max(min(index, max_val - 1), min_val)

def create_search_window(x, y, window_size, image_shape):
    """Create a search window based on the given parameters."""
    if window_size == "full":
        return 0, 0, image_shape[1], image_shape[0]
    else:
        half_window = window_size // 2
        x_start = max(0, x - half_window)
        y_start = max(0, y - half_window)
        x_end = min(image_shape[1], x + half_window)
        y_end = min(image_shape[0], y + half_window)
        return x_start, y_start, x_end, y_end

def normalize_image(img):
    """Normalize image to range [0, 1]."""
    return (img - np.min(img)) / (np.max(img) - np.min(img))

def apply_colormap(img, cmap):
    """Apply colormap to normalized image."""
    return (cmap(img)[:, :, :3] * 255).astype(np.uint8)

def apply_colormap_to_images(img1, img2, cmap):
    """Apply colormap to two images."""
    img1_normalized = normalize_image(img1)
    img2_normalized = normalize_image(img2)
    return apply_colormap(img1_normalized, cmap), apply_colormap(img2_normalized, cmap)

#------------------- Plotting Functions -------------------#

def configure_axes(ax, title, image, cmap="viridis", vmin=None, vmax=None):
    """Configure a single axis with an image and title."""
    if vmin is None or vmax is None:
        vmin, vmax = np.min(image), np.max(image)
    ax.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.axis('off')

def add_rectangle(ax, x, y, width, height, **kwargs):
    """Add a rectangle to the given axis, encompassing the pixels exactly, with grid lines."""
    rect = plt.Rectangle((x - 0.5, y - 0.5), width, height, **kwargs)
    ax.add_patch(rect)
    
    # Create vertical lines
    vlines = [[(x - 0.5 + i, y - 0.5), (x - 0.5 + i, y + height - 0.5)] for i in range(1, width)]
    
    # Create horizontal lines
    hlines = [[(x - 0.5, y - 0.5 + i), (x + width - 0.5, y - 0.5 + i)] for i in range(1, height)]
    
    # Combine all lines
    lines = vlines + hlines
    
    # Create line collection
    lc = LineCollection(lines, colors='gray', linestyles=':', linewidths=0.5)
    
    # Add line collection to the axis
    ax.add_collection(lc)

def annotate_image(ax, data, text_color="red", fontsize=10):
    """Annotate image with pixel values."""
    for i, row in enumerate(data):
        for j, val in enumerate(row):
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=text_color, fontsize=fontsize)


def create_figure(data, title, cmap, vmin=None, vmax=None, figsize=(5, 5)):
    """Create a figure with given data and settings."""
    if vmin is None or vmax is None:
        vmin, vmax = np.min(data), np.max(data)
    fig, ax = plt.subplots(figsize=figsize)
    configure_axes(ax, title, data, cmap, vmin, vmax)
    return fig, ax

def plot_image_with_overlays(ax, image, overlays, last_x, last_y, kernel_size, title, cmap="viridis", search_window=None):
    """Plot an image with optional overlays and rectangles."""
    vmin, vmax = np.min(image), np.max(image)
    configure_axes(ax, title, image, cmap=cmap, vmin=vmin, vmax=vmax)
    add_rectangle(ax, last_x, last_y, kernel_size, kernel_size, edgecolor="r", linewidth=2, facecolor="none")

    if search_window is not None:
        if isinstance(search_window, int) or search_window == "full":
            search_x_start, search_y_start, search_x_end, search_y_end = create_search_window(
                last_x, last_y, search_window, image.shape
            )
            add_rectangle(ax, search_x_start, search_y_start,
                          search_x_end - search_x_start, search_y_end - search_y_start,
                          edgecolor="g", linewidth=2, facecolor="none")

def update_plot(main_image, overlays, last_x, last_y, kernel_size, titles, cmap="viridis", search_window=None, figsize=(15, 5)):
    """Create and update a plot with the main image and overlays."""
    n_plots = 1 + len(overlays)
    fig, axs = plt.subplots(1, n_plots, figsize=figsize)
    if n_plots == 1:
        axs = [axs]

    plot_image_with_overlays(axs[0], main_image, [], last_x, last_y, kernel_size, titles[0], cmap, search_window)

    for ax, overlay, title in zip(axs[1:], overlays, titles[1:]):
        vmin, vmax = np.min(overlay), np.max(overlay)
        configure_axes(ax, title, overlay, cmap=cmap, vmin=vmin, vmax=vmax)

    fig.tight_layout(pad=2)
    return fig

#------------------- Streamlit-specific Functions -------------------#

def display_data_and_zoomed_view(data, full_data, last_x, last_y, stride, title, data_placeholder, zoomed_placeholder, cmap="viridis", zoom_size=1, fontsize=10, text_color="red"):
    """Display data and its zoomed-in view using Streamlit placeholders."""
    np.min(full_data), np.max(full_data)
    
    # Display full data
    fig_full, _ = create_figure(data, title, cmap)
    data_placeholder.pyplot(fig_full)
    
    # Display zoomed data
    zoomed_data = data[last_y // stride : last_y // stride + zoom_size, 
                       last_x // stride : last_x // stride + zoom_size]
    
    fig_zoom, ax_zoom = create_figure(zoomed_data, f"Zoomed-In {title}", cmap)
    annotate_image(ax_zoom, zoomed_data, text_color, fontsize)
    
    fig_zoom.tight_layout(pad=2)
    zoomed_placeholder.pyplot(fig_zoom)

def display_kernel_view(kernel_data, full_image_data, title, placeholder, cmap="viridis", fontsize=10, text_color="red"):
    """Display the kernel view with pixel values annotated."""
    vmin, vmax = np.min(full_image_data), np.max(full_image_data)
    fig, ax = create_figure(kernel_data, title, cmap, vmin, vmax)
    annotate_image(ax, kernel_data, text_color, fontsize)
    
    fig.tight_layout(pad=2)
    placeholder.pyplot(fig)
