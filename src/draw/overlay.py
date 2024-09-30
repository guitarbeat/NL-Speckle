"""
overlay.py: A module for adding visual overlays to image processing visualizations.

This module is designed to be imported and used in other scripts. It provides
functions for adding kernels, search windows, and pixel value overlays to
matplotlib subplots.

Usage:
    from src.draw.overlay import add_overlays, add_kernel_rectangle, ...

Do not run this file directly.
"""

import itertools
from typing import Tuple, List, Dict, Any
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
import traceback
from session_state import get_use_whole_image, get_nlm_options

def add_overlays(
    subplot: plt.Axes, image: np.ndarray, config: Dict[str, Any]
) -> None:
    try:
        if config.get('show_kernel', False):
            if config.get('title') == "Original Image":
                add_kernel_rectangle(subplot, config)
                add_kernel_grid_lines(subplot, config)

                if config.get('technique') == "nlm":
                    # Use the new function to get the use_whole_image flag
                    use_whole_image = get_use_whole_image()
                    add_search_window_overlay(subplot, image, config, use_whole_image)

            highlight_center_pixel(subplot, config)

        if config.get('zoom', False) and config.get('show_per_pixel_processing', False):
            add_pixel_value_overlay(subplot, image, config)
            add_kernel_rectangle(subplot, config)
            add_kernel_grid_lines(subplot, config)
    except Exception as e:
        print(f"Error in add_overlays: {str(e)}")
        print("Traceback:")
        traceback.print_exc()
        print(f"Config: {config}")
        raise  # Re-raise the exception after logging

def add_kernel_rectangle(subplot: plt.Axes, config: Dict[str, Any]) -> None:
    kernel_top_left = get_kernel_top_left(config)
    subplot.add_patch(
        plt.Rectangle(
            kernel_top_left,
            config['kernel']['size'],
            config['kernel']['size'],
            edgecolor=config['kernel']['outline_color'],
            linewidth=config['kernel']['outline_width'],
            facecolor="none",
        )
    )

def add_kernel_grid_lines(subplot: plt.Axes, config: Dict[str, Any]) -> None:
    """
    Add grid lines to the kernel in the subplot.

    Args:
        subplot (plt.Axes): The subplot to add the kernel grid lines to. config
        (VisualizationConfig): Configuration parameters.
    """
    grid_lines = generate_kernel_grid_lines(config)
    subplot.add_collection(
        LineCollection(
            grid_lines,
            colors=config['kernel']['grid_line_color'],
            linestyles=config['kernel']['grid_line_style'],
            linewidths=config['kernel']['grid_line_width'],
        )
    )


def highlight_center_pixel(subplot: plt.Axes, config: Dict[str, Any]) -> None:
    """
    Highlight the center pixel of the kernel in the subplot.

    Args:
        subplot (plt.Axes): The subplot to highlight the center pixel in. config
        (VisualizationConfig): Configuration parameters.
    """
    center_pixel_coords = (
        config['last_processed_pixel'][0] - 0.5,
        config['last_processed_pixel'][1] - 0.5,
    )
    subplot.add_patch(
        plt.Rectangle(
            center_pixel_coords,
            1,
            1,
            edgecolor=config['kernel']['center_pixel_color'],
            linewidth=config['kernel']['center_pixel_outline_width'],
            facecolor="none",
        )
    )


def get_kernel_top_left(config: Dict[str, Any]) -> Tuple[float, float]:
    """
    Get the top-left coordinates of the kernel rectangle.

    Args:
        config (Dict[str, Any]): Configuration parameters.

    Returns:
        Tuple[float, float]: The x and y coordinates of the top-left corner of the kernel.
    """
    last_processed_pixel = config.get('last_processed_pixel', (0, 0))
    kernel_size = config['kernel']['size']
    return (
        last_processed_pixel[0] - (kernel_size // 2) - 0.5,
        last_processed_pixel[1] - (kernel_size // 2) - 0.5
    )


def generate_kernel_grid_lines(
    config: Dict[str, Any]
) -> List[List[Tuple[float, float]]]:
    """
    Generate the grid lines for the kernel.

    Args:
        config (VisualizationConfig): Configuration parameters.

    Returns:
        List[List[Tuple[float, float]]]: The kernel grid lines.
    """
    kernel_top_left = get_kernel_top_left(config)
    kernel_size = config['kernel']['size']
    kernel_bottom_right = (
        kernel_top_left[0] + kernel_size,
        kernel_top_left[1] + kernel_size,
    )

    vertical_lines = [
        [
            (kernel_top_left[0] + i, kernel_top_left[1]),
            (kernel_top_left[0] + i, kernel_bottom_right[1]),
        ]
        for i in range(1, kernel_size)
    ]
    horizontal_lines = [
        [
            (kernel_top_left[0], kernel_top_left[1] + i),
            (kernel_bottom_right[0], kernel_top_left[1] + i),
        ]
        for i in range(1, kernel_size)
    ]
    return vertical_lines + horizontal_lines

def add_search_window_overlay(
    subplot: plt.Axes, image: np.ndarray, config: Dict[str, Any], use_whole_image: bool
) -> None:
    """
    Add search window overlay for the NLM technique to the subplot.

    Args:
        subplot (plt.Axes): The subplot to add the search window overlay to.
        image (np.ndarray): The image being plotted.
        config (Dict[str, Any]): Configuration parameters.
        use_whole_image (bool): Whether to use the full image as the search window.
    """
    if use_whole_image:
        window_left, window_top = -0.5, -0.5
        window_width, window_height = image.shape[1], image.shape[0]
    else:
        window_left, window_top, window_width, window_height = get_search_window_dims(
            image, config
        )

    subplot.add_patch(
        plt.Rectangle(
            (window_left, window_top),
            window_width,
            window_height,
            edgecolor=config['search_window']['outline_color'],
            linewidth=config['search_window']['outline_width'],
            facecolor="none",
        )
    )
    
def get_search_window_dims(
    image: np.ndarray, config: Dict[str, Any]
) -> Tuple[float, float, float, float]:

    image_height, image_width = image.shape[:2]

    if get_use_whole_image():  # Use the new function
        return -0.5, -0.5, image_width, image_height

    nlm_options = get_nlm_options()  # Use the new function
    half_window_size = nlm_options['search_window_size'] // 2
    last_processed_pixel_x, last_processed_pixel_y = config['last_processed_pixel']

    window_left = max(0, last_processed_pixel_x - half_window_size) - 0.5
    window_top = max(0, last_processed_pixel_y - half_window_size) - 0.5
    window_right = min(image_width, last_processed_pixel_x + half_window_size + 1)
    window_bottom = min(image_height, last_processed_pixel_y + half_window_size + 1)

    window_width = window_right - window_left
    window_height = window_bottom - window_top

    return window_left, window_top, window_width, window_height

# Pixel value-related functions
def add_pixel_value_overlay(
    subplot: plt.Axes, image: np.ndarray, config: Dict[str, Any]
) -> None:
    """
    Add pixel value overlay for the zoomed view to the subplot.

    Args:
        subplot (plt.Axes): The subplot to add the pixel value overlay to. image
        (np.ndarray): The image being plotted. config (VisualizationConfig):
        Configuration parameters.
    """
    image_height, image_width = image.shape[:2]
    for i, j in itertools.product(range(image_height), range(image_width)):
        subplot.text(
            j,
            i,
            f"{int(image[i, j])}",
            ha="center",
            va="center",
            color=config['pixel_value']['text_color'],
            fontsize=config['pixel_value']['font_size'],
        )
