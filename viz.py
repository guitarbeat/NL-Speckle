# Library imports
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np

# ----------------------------- Plotting Functions ----------------------------- #

# Creates a combined plot with various overlays and annotations
@st.cache_data(persist=True)
def create_combined_plot(plot_image, plot_x, plot_y, plot_kernel_size, 
                         title, plot_cmap="viridis", plot_search_window=None, 
                         zoom=False, vmin=None, vmax=None):
    """
    Create a combined plot with various overlays and annotations.
    """
    fig, ax = plt.subplots(1, 1)
    ax.imshow(plot_image, cmap=plot_cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.axis('off')

    ax.add_patch(plt.Rectangle((plot_x - 0.5, plot_y - 0.5), 1, 1, 
                               edgecolor="r", linewidth=0.5, facecolor="r", alpha=0.2))
    draw_kernel_overlay(ax, plot_x, plot_y, plot_kernel_size) 
    draw_search_window_overlay(ax, plot_image, plot_x, plot_y, plot_search_window)
    
    if zoom:
        draw_value_annotations(ax, plot_image)
    fig.tight_layout(pad=2)
    return fig

# Draws a kernel overlay on the given axes
def draw_kernel_overlay(ax, x, y, kernel_size):
    """
    Draw a kernel overlay on the given axes.
    """
    kx, ky = int(x - kernel_size // 2), int(y - kernel_size // 2)
    ax.add_patch(plt.Rectangle((kx - 0.5, ky - 0.5), kernel_size, kernel_size, 
                               edgecolor="r", linewidth=1, facecolor="none"))
    lines = ([[(kx + i - 0.5, ky - 0.5), (kx + i - 0.5, ky + kernel_size - 0.5)] for i in range(1, kernel_size)] +
             [[(kx - 0.5, ky + i - 0.5), (kx + kernel_size - 0.5, ky + i - 0.5)] for i in range(1, kernel_size)])
    ax.add_collection(LineCollection(lines, colors='red', linestyles=':', linewidths=0.5))

# Draws a search window overlay on the given axes
def draw_search_window_overlay(ax, image, x, y, search_window):
    """
    Draw a search window overlay on the given axes.
    """
    if search_window == "full":
        rect = plt.Rectangle((-0.5, -0.5), image.shape[1], image.shape[0], 
                             edgecolor="blue", linewidth=2, facecolor="none")
    elif isinstance(search_window, int):
        half_window = search_window // 2
        rect = plt.Rectangle((x - half_window - 0.5, y - half_window - 0.5), 
                             search_window, search_window, 
                             edgecolor="blue", linewidth=1, facecolor="none")
    else:
        return
    ax.add_patch(rect)

# Draws value annotations on the given axes for each pixel in the image
def draw_value_annotations(ax, image):
    """
    Draw value annotations on the given axes for each pixel in the image.
    """
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            ax.text(j, i, f"{image[i, j]:.2f}", ha="center", va="center", color="red", fontsize=8)

# Create a plot for the full image without overlays
def create_full_image_plot(image, cmap, vmin, vmax, title):
    """
    Create a plot for the full image without overlays.
    """
    fig = plt.figure()
    plt.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.axis('off')
    plt.title(title)
    return fig

# Extract a zoomed portion of the image centered around (x, y)
def get_zoomed_image(image, x, y, zoom_size):
    """
    Extract a zoomed portion of the image centered around (x, y).
    """
    ky = int(max(0, y - zoom_size // 2))
    kx = int(max(0, x - zoom_size // 2))
    zoomed_image = image[ky:min(image.shape[0], ky + zoom_size),
                         kx:min(image.shape[1], kx + zoom_size)]
    return zoomed_image, zoom_size // 2, zoom_size // 2

def validate_visualization_inputs(image, placeholder, x, y, kernel_size, cmap, show_full, vmin, vmax, zoom):
    """
    Validate inputs for the visualize_image function.
    """
    if not isinstance(image, np.ndarray):
        raise ValueError("Image must be a numpy array")
    if not hasattr(placeholder, 'pyplot'):
        raise ValueError("Placeholder must be a Streamlit element with a pyplot method")
    
    for param_name, param_value in [('x', x), ('y', y), ('kernel_size', kernel_size), ('vmin', vmin), ('vmax', vmax)]:
        if param_value is not None and not isinstance(param_value, (int, float, np.number)):
            raise ValueError(f"{param_name} must be a number or None, got {type(param_value)}")
    
    if not isinstance(cmap, str):
        raise ValueError("cmap must be a string")
    if not isinstance(show_full, bool) or not isinstance(zoom, bool):
        raise ValueError("show_full and zoom must be boolean values")

def visualize_image(
    image,
    placeholder,
    x,
    y,
    kernel_size,
    cmap,
    show_full,
    vmin,
    vmax,
    title,
    technique=None,
    search_window_size=None,
    zoom=False
):
    """
    Visualize an image with various options and overlays.
    """
    # Input validation
    validate_visualization_inputs(image, placeholder, x, y, kernel_size, cmap, show_full, vmin, vmax, zoom)

    if show_full and not zoom:
        fig = create_full_image_plot(image, cmap, vmin, vmax, title)
    else:
        if zoom:
            image, x, y = get_zoomed_image(image, x, y, kernel_size)
        
        fig = create_combined_plot(image, x, y, kernel_size, title, cmap, 
                                   search_window_size if technique == "nlm" else None, 
                                   zoom=zoom, vmin=vmin, vmax=vmax)
    
    placeholder.pyplot(fig)