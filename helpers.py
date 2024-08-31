
import matplotlib.pyplot as plt
# ---------------------------- Utility Functions ---------------------------- #



#------------------- Utility Functions -------------------#
def clip_indices(index, min_val, max_val):
    """Clip indices to ensure they stay within valid ranges."""
    return max(min(index, max_val - 1), min_val)

def configure_axes(ax, title, image, cmap="viridis", vmin=None, vmax=None):
    """Configure a single axis with an image and title."""
    ax.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.axis('off')

def add_rectangle(ax, x, y, width, height, **kwargs):
    """Add a rectangle to the given axis."""
    rect = plt.Rectangle((x, y), width, height, **kwargs)
    ax.add_patch(rect)

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

def plot_image_with_overlays(ax, image, overlays, last_x, last_y, kernel_size, title, cmap="viridis", search_window=None):
    """Plot an image with optional overlays and rectangles."""
    configure_axes(ax, title, image, cmap=cmap, vmin=0, vmax=1)
    add_rectangle(ax, last_x, last_y, kernel_size, kernel_size, edgecolor="r", linewidth=2, facecolor="none")

    if search_window is not None:
        if isinstance(search_window, int) or search_window == "full":
            search_x_start, search_y_start, search_x_end, search_y_end = create_search_window(
                last_x, last_y, search_window, image.shape
            )
            add_rectangle(ax, search_x_start, search_y_start,
                          search_x_end - search_x_start, search_y_end - search_y_start,
                          edgecolor="g", linewidth=2, facecolor="none")

    # for overlay in overlays:
    #     ax.imshow(overlay, cmap=cmap, alpha=0.5)

def update_plot(main_image, overlays, last_x, last_y, kernel_size, titles, cmap="viridis", search_window=None, figsize=(15, 5)):
    """
    Create and update a plot with the main image and overlays.
    
    Parameters:
    - main_image: 2D numpy array representing the main image
    - overlays: List of 2D numpy arrays for additional overlays
    - last_x, last_y: Coordinates of the last processed pixel
    - kernel_size: Size of the kernel
    - titles: List of titles for each subplot
    - cmap: Colormap for the main image and overlays (default: 'viridis')
    - search_window: Size of the search window, 'full' for entire image, or None (default: None)
    - figsize: Size of the figure (default: (15, 5))
    
    Returns:
    - fig: Matplotlib figure object
    """
    n_plots = 1 + len(overlays)
    fig, axs = plt.subplots(1, n_plots, figsize=figsize)
    if n_plots == 1:
        axs = [axs]

    # Plot the original image without overlays
    plot_image_with_overlays(axs[0], main_image, [], last_x, last_y, kernel_size, titles[0], cmap, search_window)

    for ax, overlay, title in zip(axs[1:], overlays, titles[1:]):
        configure_axes(ax, title, overlay, cmap=cmap)

    fig.tight_layout(pad=2)
    return fig