"""
The above code defines data classes and functions for visualizing image
processing techniques with overlays such as kernels, search windows, and pixel
values.

:param subplot: The `subplot` parameter in the provided code refers to a subplot
    within a Matplotlib
figure. Subplots are used to arrange multiple plots within a single figure. In
this context, the `subplot` parameter is of type `plt.Axes`, which represents an
individual plot or subplot within a Matplotlib :type subplot: plt.Axes :param
image: The code you provided defines several data classes and functions related
to image processing visualization. Here's a brief overview of the key
components: :type image: np.ndarray :param config: The `config` parameter in the
provided code represents a configuration object that holds various settings for
image visualization and analysis. Here are the key attributes of the
`VisualizationConfig` data class: :type config: VisualizationConfig
"""

import itertools

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, List

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection


@dataclass
class KernelConfig:
    size: int = 3
    outline_color: str = "red"
    outline_width: int = 1
    grid_line_color: str = "red"
    grid_line_width: int = 1
    grid_line_style: str = ":"
    center_pixel_color: str = "green"
    center_pixel_outline_width: int = 1
    kernel_matrix: Optional[np.ndarray] = None


@dataclass
class SearchWindowConfig:
    size: Optional[int] = None
    outline_color: str = "blue"
    outline_width: float = 2.0
    use_full_image: bool = False


@dataclass
class PixelValueConfig:
    text_color: str = "red"
    font_size: int = 15


@dataclass
class VisualizationConfig:
    """Holds configuration for image visualization and analysis settings."""

    vmin: Optional[float] = None
    vmax: Optional[float] = None
    zoom: bool = False
    show_kernel: bool = False
    show_per_pixel_processing: bool = False
    image_array: Optional[np.ndarray] = None
    analysis_params: Dict[str, Any] = field(default_factory=dict)
    results: Optional[Any] = None
    ui_placeholders: Dict[str, Any] = field(default_factory=dict)
    last_processed_pixel: Optional[Tuple[int, int]] = None
    original_pixel_value: float = 0.0
    technique: str = ""
    color_map: str = "gray"
    title: str = ""
    figure_size: Tuple[int, int] = (8, 8)
    kernel: KernelConfig = field(default_factory=KernelConfig)
    search_window: SearchWindowConfig = field(
        default_factory=SearchWindowConfig)
    pixel_value: PixelValueConfig = field(default_factory=PixelValueConfig)

    def __post_init__(self):
        """Post-initialization validation."""
        self._validate_vmin_vmax()

    def _validate_vmin_vmax(self):
        """Ensure vmin is not greater than vmax."""
        if self.vmin is not None and self.vmax is not None and self.vmin > self.vmax:
            raise ValueError("vmin cannot be greater than vmax.")


def add_overlays(subplot: plt.Axes, image: np.ndarray, config: VisualizationConfig) -> None:
    """
    Add overlays to the plot based on the technique and configuration.

    Args:
        subplot (plt.Axes): The subplot to add overlays to. image (np.ndarray):
        The image being plotted. config (VisualizationConfig): Configuration
        parameters.
    """
    if config.show_kernel:
        add_kernel_overlay(subplot, config)

        if config.technique == "nlm" and config.search_window.size is not None:
            add_search_window_overlay(subplot, image, config)

    if config.zoom and config.show_per_pixel_processing:
        add_pixel_value_overlay(subplot, image, config)


def add_kernel_overlay(subplot: plt.Axes, config: VisualizationConfig) -> None:
    """
    Add kernel overlay to the plot.

    Args:
        subplot (plt.Axes): The subplot to add the kernel overlay to. config
        (VisualizationConfig): Configuration parameters.
    """
    add_kernel_rectangle(subplot, config)
    add_kernel_grid_lines(subplot, config)
    highlight_center_pixel(subplot, config)


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


def add_search_window_overlay(subplot: plt.Axes, image: np.ndarray, config: VisualizationConfig) -> None:
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


def add_pixel_value_overlay(subplot: plt.Axes, image: np.ndarray, config: VisualizationConfig) -> None:
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
            f"{image[i, j]:.2f}",
            ha="center",
            va="center",
            color=config.pixel_value.text_color,
            fontsize=config.pixel_value.font_size,
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


def generate_kernel_grid_lines(config: VisualizationConfig) -> List[List[Tuple[float, float]]]:
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


def get_search_window_dims(image: np.ndarray, config: VisualizationConfig) -> Tuple[float, float, float, float]:
    """
    Calculate the dimensions of the search window.

    Args:
        image (np.ndarray): The image being plotted. config
        (VisualizationConfig): Configuration parameters.

    Returns:
        Tuple[float, float, float, float]: The left, top, width, and height of
        the search window.
    """
    image_height, image_width = image.shape[:2]

    if config.search_window.use_full_image:
        return -0.5, -0.5, image_width, image_height

    half_window_size = config.search_window.size // 2
    last_processed_pixel_x, last_processed_pixel_y = config.last_processed_pixel

    window_left = max(0, last_processed_pixel_x - half_window_size) - 0.5
    window_top = max(0, last_processed_pixel_y - half_window_size) - 0.5
    window_right = min(
        image_width, last_processed_pixel_x + half_window_size + 1)
    window_bottom = min(
        image_height, last_processed_pixel_y + half_window_size + 1)

    window_width = window_right - window_left
    window_height = window_bottom - window_top

    return window_left, window_top, window_width, window_height
