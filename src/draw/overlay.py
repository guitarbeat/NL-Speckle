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
from typing import Tuple, List

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from src.classes import VisualizationConfig
# Main overlay function


def add_overlays(
    subplot: plt.Axes, image: np.ndarray, config: VisualizationConfig
) -> None:
    """
    Add overlays to the plot based on the technique and configuration.

    Args:
        subplot (plt.Axes): The subplot to add overlays to.
        image (np.ndarray): The image being plotted.
        config (VisualizationConfig): Configuration parameters.
    """
    if config.show_kernel:
        if config.title == "Original Image":
            add_kernel_rectangle(subplot, config)
            add_kernel_grid_lines(subplot, config)

            if config.technique == "nlm" and config.search_window.size is not None:
                add_search_window_overlay(subplot, image, config)

        # Always highlight the center pixel, regardless of the image type
        highlight_center_pixel(subplot, config)

    if config.zoom and config.show_per_pixel_processing:
        add_pixel_value_overlay(subplot, image, config)
        add_kernel_rectangle(subplot, config)
        add_kernel_grid_lines(subplot, config)


# Kernel-related functions
def add_kernel_rectangle(subplot: plt.Axes, config: VisualizationConfig) -> None:
    """
    Add the main kernel rectangle to the subplot.

    Args:
        subplot (plt.Axes): The subplot to add the kernel rectangle to. config
        (VisualizationConfig): Configuration parameters.
    """
    kernel_top_left = get_kernel_top_left(config)
    subplot.add_patch(
        plt.Rectangle(
            kernel_top_left,
            config.kernel.size,
            config.kernel.size,
            edgecolor=config.kernel.outline_color,
            linewidth=config.kernel.outline_width,
            facecolor="none",
        )
    )


def add_kernel_grid_lines(subplot: plt.Axes, config: VisualizationConfig) -> None:
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
            colors=config.kernel.grid_line_color,
            linestyles=config.kernel.grid_line_style,
            linewidths=config.kernel.grid_line_width,
        )
    )


def highlight_center_pixel(subplot: plt.Axes, config: VisualizationConfig) -> None:
    """
    Highlight the center pixel of the kernel in the subplot.

    Args:
        subplot (plt.Axes): The subplot to highlight the center pixel in. config
        (VisualizationConfig): Configuration parameters.
    """
    center_pixel_coords = (
        config.last_processed_pixel[0] - 0.5,
        config.last_processed_pixel[1] - 0.5,
    )
    subplot.add_patch(
        plt.Rectangle(
            center_pixel_coords,
            1,
            1,
            edgecolor=config.kernel.center_pixel_color,
            linewidth=config.kernel.center_pixel_outline_width,
            facecolor=config.kernel.center_pixel_color,
            alpha=0.5,
        )
    )


def get_kernel_top_left(config: VisualizationConfig) -> Tuple[float, float]:
    """
    Calculate the top-left coordinates of the kernel.

    Args:
        config (VisualizationConfig): Configuration parameters.

    Returns:
        Tuple[float, float]: The top-left coordinates of the kernel.
    """
    return (
        config.last_processed_pixel[0] - (config.kernel.size // 2) - 0.5,
        config.last_processed_pixel[1] - (config.kernel.size // 2) - 0.5,
    )


def generate_kernel_grid_lines(
    config: VisualizationConfig,
) -> List[List[Tuple[float, float]]]:
    """
    Generate the grid lines for the kernel.

    Args:
        config (VisualizationConfig): Configuration parameters.

    Returns:
        List[List[Tuple[float, float]]]: The kernel grid lines.
    """
    kernel_top_left = get_kernel_top_left(config)
    kernel_bottom_right = (
        kernel_top_left[0] + config.kernel.size,
        kernel_top_left[1] + config.kernel.size,
    )

    vertical_lines = [
        [
            (kernel_top_left[0] + i, kernel_top_left[1]),
            (kernel_top_left[0] + i, kernel_bottom_right[1]),
        ]
        for i in range(1, config.kernel.size)
    ]
    horizontal_lines = [
        [
            (kernel_top_left[0], kernel_top_left[1] + i),
            (kernel_bottom_right[0], kernel_top_left[1] + i),
        ]
        for i in range(1, config.kernel.size)
    ]
    return vertical_lines + horizontal_lines


# Search window-related functions
def add_search_window_overlay(
    subplot: plt.Axes, image: np.ndarray, config: VisualizationConfig
) -> None:
    """
    Add search window overlay for the NLM technique to the subplot.

    Args:
        subplot (plt.Axes): The subplot to add the search window overlay to.
        image (np.ndarray): The image being plotted. config
        (VisualizationConfig): Configuration parameters.
    """
    window_left, window_top, window_width, window_height = get_search_window_dims(
        image, config
    )
    subplot.add_patch(
        plt.Rectangle(
            (window_left, window_top),
            window_width,
            window_height,
            edgecolor=config.search_window.outline_color,
            linewidth=config.search_window.outline_width,
            facecolor="none",
        )
    )


def get_search_window_dims(
    image: np.ndarray, config: VisualizationConfig
) -> Tuple[float, float, float, float]:
    """
    Calculate the dimensions of the search window.

    Args:
        image (np.ndarray): The image being plotted.
        config (VisualizationConfig): Configuration parameters.

    Returns:
        Tuple[float, float, float, float]: The left, top, width, and height of
        the search window.
    """
    import streamlit as st

    image_height, image_width = image.shape[:2]

    if st.session_state.get("use_full_image", False):  # Added default value
        return -0.5, -0.5, image_width, image_height

    half_window_size = config.search_window.size // 2
    last_processed_pixel_x, last_processed_pixel_y = config.last_processed_pixel

    window_left = max(0, last_processed_pixel_x - half_window_size) - 0.5
    window_top = max(0, last_processed_pixel_y - half_window_size) - 0.5
    window_right = min(image_width, last_processed_pixel_x + half_window_size + 1)
    window_bottom = min(image_height, last_processed_pixel_y + half_window_size + 1)

    window_width = window_right - window_left
    window_height = window_bottom - window_top

    return window_left, window_top, window_width, window_height


# Pixel value-related functions
def add_pixel_value_overlay(
    subplot: plt.Axes, image: np.ndarray, config: VisualizationConfig
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
            color=config.pixel_value.text_color,
            fontsize=config.pixel_value.font_size,
        )
