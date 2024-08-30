
import matplotlib.pyplot as plt
import matplotlib.patches as patches

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


#------------------- Utility Functions -------------------#
def clip_indices(index, min_val, max_val):
    """Clip indices to ensure they stay within valid ranges."""
    return max(min(index, max_val - 1), min_val)

def create_search_window(x, y, search_window, image_shape):
    """
    Calculate the bounds of the search window.
    
    Parameters:
    - x, y: Coordinates of the current pixel
    - search_window: Size of the search window or 'full' for entire image
    - image_shape: Shape of the image
    
    Returns:
    - Tuple of (search_x_start, search_y_start, search_x_end, search_y_end)
    """
    if search_window == "full":
        return 0, 0, image_shape[1], image_shape[0]
    
    half_size = search_window // 2
    x_start = clip_indices(x - half_size, 0, image_shape[1])
    x_end = clip_indices(x + half_size + 1, 0, image_shape[1])
    y_start = clip_indices(y - half_size, 0, image_shape[0])
    y_end = clip_indices(y + half_size + 1, 0, image_shape[0])
    
    return x_start, y_start, x_end, y_end

def add_rectangle(ax, x_start, y_start, width, height, **kwargs):
    """
    Add a rectangle patch to the provided axes.
    
    Parameters:
    - ax: Matplotlib axes object
    - x_start, y_start: Top-left coordinates of the rectangle
    - width, height: Dimensions of the rectangle
    - **kwargs: Additional keyword arguments for the rectangle properties
    """
    rect = patches.Rectangle((x_start, y_start), width, height, **kwargs)
    ax.add_patch(rect)

def update_plot(fig, axs, main_image, overlays, last_x, last_y, kernel_size, titles, cmap="viridis", search_window=None):
    """
    Update and redraw the plot with the main image and overlays.
    
    Parameters:
    - fig: Matplotlib figure object
    - axs: List of Matplotlib axes objects
    - main_image: 2D numpy array representing the main image
    - overlays: List of 2D numpy arrays for additional overlays
    - last_x, last_y: Coordinates of the last processed pixel
    - kernel_size: Size of the kernel
    - titles: List of titles for each subplot
    - cmap: Colormap for the main image and overlays (default: 'viridis')
    - search_window: Size of the search window, 'full' for entire image, or None (default: None)
    
    Returns:
    - fig: Updated Matplotlib figure object
    """
    clear_axes(axs)

    # Configure the first axis with the main image and add the kernel rectangle
    configure_axes(axs[0], "Original Image with Current Kernel", main_image, cmap=cmap, vmin=0, vmax=1)
    add_rectangle(axs[0], last_x, last_y, kernel_size, kernel_size, edgecolor="r", linewidth=2, facecolor="none")

    # Optionally add the search window rectangle
    if search_window is not None:
        if isinstance(search_window, int) or search_window == "full":
            search_x_start, search_y_start, search_x_end, search_y_end = create_search_window(
                last_x, last_y, search_window, main_image.shape
            )
            add_rectangle(axs[0], search_x_start, search_y_start, 
                          search_x_end - search_x_start, search_y_end - search_y_start, 
                          edgecolor="g", linewidth=2, facecolor="none")

    # Configure the remaining axes with overlays
    for ax, overlay, title in zip(axs[1:], overlays, titles[1:]):
        configure_axes(ax, title, overlay, cmap=cmap)
    
    fig.tight_layout(pad=2)
    return fig

