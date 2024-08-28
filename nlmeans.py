import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.animation import FuncAnimation
from mpl_toolkits.axes_grid1 import make_axes_locatable

# -------------------------------------------
# Utility Functions
# -------------------------------------------

# -- Index Management --
def clip_indices(value, min_value, max_value):
    """Ensure that the value is within the specified bounds."""
    return max(min_value, min(value, max_value))

# -- Patch Extraction and Manipulation --
def get_patch(image, x, y, patch_size):
    """
    Extract a patch from the image centered at (x, y).
    """
    half = patch_size // 2
    y_start = clip_indices(y - half, 0, image.shape[0])
    y_end = clip_indices(y + half + 1, 0, image.shape[0])
    x_start = clip_indices(x - half, 0, image.shape[1])
    x_end = clip_indices(x + half + 1, 0, image.shape[1])
    return image[y_start:y_end, x_start:x_end]

def pad_patch(patch, target_shape):
    """
    Pad a given patch to match the target dimensions.
    """
    target_height, target_width = target_shape
    return np.pad(patch, ((0, target_height - patch.shape[0]), (0, target_width - patch.shape[1])), mode='constant')

# -------------------------------------------
# Core Algorithm: Non-Local Means Filter
# -------------------------------------------

# -- Weight Calculation --
def calculate_weight(patch1, patch2, h):
    """
    Assign weights to these patches based on their similarity.
    """
    target_shape = (max(patch1.shape[0], patch2.shape[0]), max(patch1.shape[1], patch2.shape[1]))
    pad1 = pad_patch(patch1, target_shape)
    pad2 = pad_patch(patch2, target_shape)
    diff = np.sum((pad1 - pad2) ** 2)
    return np.exp(-diff / (h ** 2))

# -- Search Window Definition --
def create_search_window(x, y, search_window_size, image_shape, full_image=False):
    """
    Define the search window around the current pixel (x, y).
    If full_image is True, the search window covers the entire image.
    """
    if full_image:
        # Set the search window to cover the entire image
        search_x_start, search_y_start = 0, 0
        search_x_end, search_y_end = image_shape[1], image_shape[0]
    else:
        # Default behavior: a square centered on the pixel (x, y)
        half_search = search_window_size // 2
        search_x_start = clip_indices(x - half_search, 0, image_shape[1])
        search_x_end = clip_indices(x + half_search + 1, 0, image_shape[1])
        search_y_start = clip_indices(y - half_search, 0, image_shape[0])
        search_y_end = clip_indices(y + half_search + 1, 0, image_shape[0])
    
    return search_x_start, search_y_start, search_x_end, search_y_end

# -- Weight Calculation and Pixel Filtering --
def compute_weights_and_filtered_value(image, x, y, patch_size, search_window_size, h, full_image=False):
    """
    Compute the weights and filtered value for a given pixel (x, y).
    If full_image is True, the search window covers the entire image.
    """
    search_x_start, search_y_start, search_x_end, search_y_end = create_search_window(x, y, search_window_size, image.shape, full_image=full_image)
    weights = np.zeros_like(image)
    center_patch = get_patch(image, x, y, patch_size)
    weighted_sum = 0
    weight_sum = 0

    for sy in range(search_y_start, search_y_end):
        for sx in range(search_x_start, search_x_end):
            comparison_patch = get_patch(image, sx, sy, patch_size)
            weight = calculate_weight(center_patch, comparison_patch, h)
            weights[sy, sx] = weight
            weighted_sum += weight * image[sy, sx]
            weight_sum += weight

    filtered_value = weighted_sum / weight_sum if weight_sum > 0 else image[y, x]
    return filtered_value, weights

# -------------------------------------------
# Visualization and Animation
# -------------------------------------------

# -- Visualization Utilities --
def update_rectangle_bounds(rectangle, x_start, y_start, width, height):
    """
    Update the bounds of a rectangle for visualization.
    """
    rectangle.set_bounds(x_start - 0.5, y_start - 0.5, width, height)

# -- Visual Elements Update --
def update_visual_elements(search_window, current_pixel, current_patch, x, y, search_x_start, search_y_start, search_x_end, search_y_end, patch_size):
    """
    Update the visual elements like the search window, current pixel, and patch rectangles.
    """
    update_rectangle_bounds(search_window, search_x_start, search_y_start, search_x_end - search_x_start, search_y_end - search_y_start)
    current_pixel.set_xy((x - 0.5, y - 0.5))
    half_patch = patch_size // 2
    update_rectangle_bounds(current_patch, x - half_patch, y - half_patch, patch_size, patch_size)

# -- Plot Setup for Animation --
def setup_plot(image):
    """
    Set up the plot for the animation.
    """
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)
    
    im1 = ax1.imshow(image, cmap='gray', vmin=0, vmax=1)
    im2 = ax2.imshow(np.zeros_like(image), cmap='viridis', vmin=0, vmax=1)
    im3 = ax3.imshow(np.zeros_like(image), cmap='gray', vmin=0, vmax=1)
    
    for ax, title in zip([ax1, ax2, ax3], ['Input Image', 'Weights', 'Output Image']):
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title, fontsize=12)
    
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im2, cax=cax)
    
    return fig, (ax1, ax2, ax3), (im1, im2, im3)

# -- Frame Update for Animation --
def update_frame(frame, image, patch_size, search_window_size, h, output, search_window, current_pixel, current_patch, im2, im3, full_image=False):
    """
    Update function for each frame in the animation.
    If full_image is True, the search window covers the entire image.
    """
    y, x = divmod(frame, image.shape[1])
    filtered_value, weights = compute_weights_and_filtered_value(image, x, y, patch_size, search_window_size, h, full_image=full_image)
    output[y, x] = filtered_value
    
    search_x_start, search_y_start, search_x_end, search_y_end = create_search_window(x, y, search_window_size, image.shape, full_image=full_image)
    update_visual_elements(search_window, current_pixel, current_patch, x, y, search_x_start, search_y_start, search_x_end, search_y_end, patch_size)
    
    im2.set_data(weights)
    im3.set_data(output)
    
    return im2, im3, search_window, current_pixel, current_patch

# -- Non-Local Means Filter Animation --
def nlm_filter_animation(image, patch_size=3, search_window_size=7, h=0.1, interval=200, filename=None, full_image=False):
    """
    Create an animation of the Non-Local Means filtering process.
    If a filename is provided, the animation is saved as a GIF and the output image is returned separately.
    If full_image is True, the search window covers the entire image.
    """
    output = np.zeros_like(image)
    if filename is not None:
        fig, (ax1, ax2, ax3), (im1, im2, im3) = setup_plot(image)
        
        search_window = Rectangle((0, 0), search_window_size, search_window_size, fill=False, edgecolor='red', linewidth=2)
        current_pixel = Rectangle((0, 0), 1, 1, fill=True, facecolor='red', edgecolor='none')
        current_patch = Rectangle((0, 0), patch_size, patch_size, fill=False, edgecolor='blue', linewidth=2)
        ax1.add_patch(search_window)
        ax1.add_patch(current_pixel)
        ax1.add_patch(current_patch)
        
        anim = FuncAnimation(fig, update_frame, fargs=(image, patch_size, search_window_size, h, output, search_window, current_pixel, current_patch, im2, im3, full_image),
                             frames=image.size, interval=interval, blit=True, repeat=False)
        anim.save(filename, writer='pillow')
        plt.close(fig)
        return output  # Return the final processed output image
    else:
        # Process without creating an animation
        for frame in range(image.size):
            y, x = divmod(frame, image.shape[1])
            filtered_value, _ = compute_weights_and_filtered_value(image, x, y, patch_size, search_window_size, h, full_image=full_image)
            output[y, x] = filtered_value
        return output
