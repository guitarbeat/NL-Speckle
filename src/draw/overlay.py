"""
overlay.py: A module for adding visual overlays to image processing visualizations.

This module provides functions for adding visual elements such as kernels, search windows, and pixel values to matplotlib subplots, intended for use in image processing scripts.

Usage:
    from src.draw.overlay import add_overlays, add_kernel_rectangle, ...

Note: This module is not meant to be run directly.
"""

import itertools
from typing import Any, Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

def add_overlays(subplot: plt.Axes, image: np.ndarray, config: Dict[str, Any]) -> None:
    """Add various overlays (kernels, search windows, pixel values) to the subplot based on the config."""
    config = _set_default_config_values(config)

    if config.get("show_per_pixel_processing"):
        add_kernel_rectangle(subplot, config)
        add_kernel_grid_lines(subplot, config)
        highlight_center_pixel(subplot, config)

        if config.get("technique") == "nlm":
            add_search_window_overlay(subplot, image, config)

        if config.get("zoom"):
            add_pixel_value_overlay(subplot, image, config)

    elif config.get("show_kernel") and config.get("title") == "Original Image":
        add_kernel_rectangle(subplot, config)
        add_kernel_grid_lines(subplot, config)
        highlight_center_pixel(subplot, config)


def _set_default_config_values(config: Dict[str, Any]) -> Dict[str, Any]:
    """Set default values for kernel, pixel value, and search window configurations."""
    config.setdefault("kernel", {}).update({
        "outline_color": "red",
        "outline_width": 1,
        "grid_line_color": "red",
        "grid_line_style": ":",
        "grid_line_width": 1,
        "center_pixel_color": "green",
        "center_pixel_opacity": 0.3
    })

    config.setdefault("pixel_value", {}).update({
        "text_color": "white",
        "font_size": 12
    })

    config.setdefault("search_window", {}).update({
        "outline_color": "blue",
        "outline_width": 1.5
    })

    return config


def add_kernel_rectangle(subplot: plt.Axes, config: Dict[str, Any]) -> None:
    """Draw a rectangle representing the kernel on the subplot."""
    kernel_top_left = get_kernel_top_left(config)
    subplot.add_patch(
        plt.Rectangle(
            kernel_top_left,
            config["kernel"]["size"],
            config["kernel"]["size"],
            edgecolor=config["kernel"]["outline_color"],
            linewidth=config["kernel"]["outline_width"],
            facecolor="none"
        )
    )


def add_kernel_grid_lines(subplot: plt.Axes, config: Dict[str, Any]) -> None:
    """Add grid lines within the kernel rectangle."""
    grid_lines = generate_kernel_grid_lines(config)
    subplot.add_collection(
        LineCollection(
            grid_lines,
            colors=config["kernel"]["grid_line_color"],
            linestyles=config["kernel"]["grid_line_style"],
            linewidths=config["kernel"]["grid_line_width"]
        )
    )


def highlight_center_pixel(subplot: plt.Axes, config: Dict[str, Any]) -> None:
    """Highlight the center pixel of the kernel."""
    last_y, last_x = config["last_processed_pixel"]
    center_pixel_coords = (last_x - 0.5, last_y - 0.5)
    subplot.add_patch(
        plt.Rectangle(
            center_pixel_coords,
            1,
            1,
            facecolor=config["kernel"]["center_pixel_color"],
            alpha=config["kernel"]["center_pixel_opacity"],
            edgecolor="none"
        )
    )


def get_kernel_top_left(config: Dict[str, Any]) -> Tuple[float, float]:
    """Calculate the top-left corner coordinates of the kernel."""
    last_y, last_x = config["last_processed_pixel"]
    half_kernel = config["half_kernel"]
    return last_x - half_kernel - 0.5, last_y - half_kernel - 0.5



def generate_kernel_grid_lines(config: Dict[str, Any]) -> List[List[Tuple[float, float]]]:
    """Generate the vertical and horizontal grid lines for the kernel."""
    kernel_top_left = get_kernel_top_left(config)
    kernel_size = config["kernel"]["size"]
    kernel_bottom_right = (
        kernel_top_left[0] + kernel_size,
        kernel_top_left[1] + kernel_size
    )

    vertical_lines = [
        [(kernel_top_left[0] + i, kernel_top_left[1]), (kernel_top_left[0] + i, kernel_bottom_right[1])]
        for i in range(1, kernel_size)
    ]
    horizontal_lines = [
        [(kernel_top_left[0], kernel_top_left[1] + i), (kernel_bottom_right[0], kernel_top_left[1] + i)]
        for i in range(1, kernel_size)
    ]
    
    return vertical_lines + horizontal_lines


def add_search_window_overlay(subplot: plt.Axes, image: np.ndarray, config: Dict[str, Any]) -> None:
    """Draw a rectangle representing the search window on the subplot."""
    if config.get("use_full_image", False):
        window_dims = (-0.5, -0.5, image.shape[1], image.shape[0])
    else:
        window_dims = get_search_window_dims(config)

    subplot.add_patch(
        plt.Rectangle(
            window_dims[:2],
            window_dims[2],
            window_dims[3],
            edgecolor=config["search_window"]["outline_color"],
            linewidth=config["search_window"]["outline_width"],
            facecolor="none"
        )
    )


def get_search_window_dims(config: Dict[str, Any]) -> Tuple[float, float, float, float]:
    """Calculate the dimensions of the search window based on the configuration, using total area."""
    half_window_size = config["search_window_size"] // 2
    last_y, last_x = config["last_processed_pixel"]
    total_area = config["total_area"]

    window_left = max(total_area["left"], last_x - half_window_size) - 0.5
    window_top = max(total_area["top"], last_y - half_window_size) - 0.5
    window_right = min(total_area["right"], last_x + half_window_size) - 0.5
    window_bottom = min(total_area["bottom"], last_y + half_window_size) - 0.5

    return window_left, window_top, window_right - window_left, window_bottom - window_top


def add_pixel_value_overlay(subplot: plt.Axes, image: np.ndarray, config: Dict[str, Any]) -> None:
    """Overlay pixel values as text on the subplot."""
    image_height, image_width = image.shape[:2]

    for i, j in itertools.product(range(image_height), range(image_width)):
        subplot.text(
            j,
            i,
            f"{image[i, j]:.0f}",
            ha="center",
            va="center",
            color=config["pixel_value"]["text_color"],
            fontsize=config["pixel_value"]["font_size"]
        )