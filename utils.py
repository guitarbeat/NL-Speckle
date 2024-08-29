import numpy as np
import cv2
# ---------------------------- Library Functions ---------------------------- #

def compute_output_dimensions(image: np.ndarray, kernel_size: int, stride: int) -> tuple:
    """Calculate the output dimensions of the filtered image."""
    output_height = (image.shape[0] - kernel_size) // stride + 1
    output_width = (image.shape[1] - kernel_size) // stride + 1
    return output_height, output_width

def calculate_local_statistics(local_window: np.ndarray) -> tuple:
    """Calculate the mean, standard deviation, and speckle contrast for a local window."""
    local_mean = np.mean(local_window)
    local_std = np.std(local_window)
    speckle_contrast = local_std / local_mean if local_mean != 0 else 0
    return local_mean, local_std, speckle_contrast

def calculate_statistics(image: np.ndarray, kernel_size: int, stride: int, max_pixels: int, cache: dict) -> tuple:
    """Calculate the mean, standard deviation, and speckle contrast for an image."""
    output_height, output_width = compute_output_dimensions(image, kernel_size, stride)
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
            local_mean, local_std, speckle_contrast = calculate_local_statistics(local_window)
            cache[cache_key][(row, col)] = (local_mean, local_std, speckle_contrast)

        mean_filter[row, col] = local_mean
        std_dev_filter[row, col] = local_std
        sc_filter[row, col] = speckle_contrast

    last_x, last_y = top_left_x, top_left_y
    return mean_filter, std_dev_filter, sc_filter, last_x, last_y, local_mean, local_std, speckle_contrast

# ---------------------------- Additional Functions ---------------------------- #
# This section is left intentionally blank for future expansions or additional functions.
