import numpy as np
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any, Optional, Type

from src.math.formula import display_analysis_formula, prepare_variables
from src.draw.overlay import add_overlays

import matplotlib.pyplot as plt
import streamlit as st

###############################################################################
#                              Basic Data Classes                             #
###############################################################################

@dataclass
class ImageArray:
    data: np.ndarray

###############################################################################
#                         Visualization Configuration                         #
###############################################################################

@dataclass
class KernelVisualizationConfig:
    size: int
    kernel_matrix: Optional[np.ndarray] = None
    outline_color: str = "red"
    outline_width: float = 1
    grid_line_color: str = "red"
    grid_line_style: str = ":"
    grid_line_width: float = 1
    center_pixel_color: str = "green"
    center_pixel_outline_width: float = 2.0
    origin: Tuple[int, int] = (0, 0)

@dataclass
class SearchWindowConfig:
    size: Optional[int] = None
    outline_color: str = "blue"
    outline_width: float = 2.0
    use_full_image: bool = True

@dataclass
class PixelValueConfig:
    text_color: str = "red"
    font_size: int = 10

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
    original_pixel_value: float = (0.0)
    technique: str = ""
    title: str = ""
    figure_size: Tuple[int, int] = (8, 8)
    kernel: KernelVisualizationConfig = field(default_factory=KernelVisualizationConfig)
    search_window: SearchWindowConfig = field(default_factory=SearchWindowConfig)
    pixel_value: PixelValueConfig = field(default_factory=PixelValueConfig)
    processing_end: Tuple[int, int] = field(default_factory=tuple)
    pixels_to_process: int = 0

    def __post_init__(self):
        """Post-initialization validation."""
        self._validate_vmin_vmax()

    def _validate_vmin_vmax(self):
        """Ensure vmin is not greater than vmax."""
        if self.vmin is not None and self.vmax is not None and self.vmin > self.vmax:
            raise ValueError("vmin cannot be greater than vmax.")

###############################################################################
#                              Base Classes                                   #
###############################################################################

class ResultCombinationError(Exception):
    """Exception raised when there's an error combining results."""
    pass

@dataclass
class BaseResult(ABC):
    processing_end_coord: Tuple[int, int]
    kernel_size: int
    pixels_processed: int
    image_dimensions: Tuple[int, int]

    @classmethod
    def combine(
        class_: Type["BaseResult"], results: List["BaseResult"]
    ) -> "BaseResult":
        if not results:
            raise ResultCombinationError("No results provided for combination")
        return class_(
            processing_end_coord=max(r.processing_end_coord for r in results),
            kernel_size=results[0].kernel_size,
            pixels_processed=sum(r.pixels_processed for r in results),
            image_dimensions=results[0].image_dimensions,
        )

    @classmethod
    def empty_result(class_: Type["BaseResult"]) -> "BaseResult":
        return class_(
            processing_end_coord=(0, 0),
            kernel_size=0,
            pixels_processed=0,
            image_dimensions=(0, 0),
        )

    @staticmethod
    @abstractmethod
    def get_filter_options() -> List[str]:
        pass

    @property
    @abstractmethod
    def filter_data(self) -> Dict[str, np.ndarray]:
        pass

###############################################################################
#                             Utility Functions                               #
###############################################################################


def get_zoomed_image_section(image: np.ndarray, center_x: int, center_y: int, kernel_size: int):
    half_zoom = kernel_size // 2
    top, bottom = max(0, center_y - half_zoom), min(image.shape[0], center_y + half_zoom + 1)
    left, right = max(0, center_x - half_zoom), min(image.shape[1], center_x + half_zoom + 1)
    return image[top:bottom, left:right], center_x - left, center_y - top


def visualize_image(image: np.ndarray, placeholder, *, config: VisualizationConfig) -> None:
    try:
        if config.zoom:
            image, new_center_x, new_center_y = get_zoomed_image_section(
                image, config.last_processed_pixel[0], config.last_processed_pixel[1], config.kernel.size
            )
            config.last_processed_pixel = (new_center_x, new_center_y)

        fig, ax = plt.subplots(1, 1, figsize=config.figure_size)
        ax.imshow(image, vmin=config.vmin, vmax=config.vmax, cmap=st.session_state.color_map)
        ax.set_title(config.title)
        ax.axis("off")
        add_overlays(ax, image, config)
        fig.tight_layout(pad=2)
        placeholder.pyplot(fig)
        plt.close(fig)
    except Exception as e:
        placeholder.error(f"An error occurred while visualizing the image: {e}. Please check the logs for details.")


def visualize_analysis_results(viz_params: VisualizationConfig) -> None:
    filter_options = viz_params.results.filter_data
    filter_options["Original Image"] = viz_params.image_array.data
    selected_filters = st.session_state.get(f"{viz_params.technique}_selected_filters", [])

    for filter_name in selected_filters:
        if filter_name in filter_options:
            filter_data = filter_options[filter_name]
            for plot_type in ["main", "zoomed"]:
                plot_key = f"{'zoomed_' if plot_type == 'zoomed' else ''}{filter_name.lower().replace(' ', '_')}"
                if plot_key in viz_params.ui_placeholders and (plot_type != "zoomed" or viz_params.show_per_pixel_processing):
                    config = VisualizationConfig(
                        **{**viz_params.__dict__,
                           "vmin": None if filter_name == "Original Image" else np.min(filter_data),
                           "vmax": None if filter_name == "Original Image" else np.max(filter_data),
                           "zoom": (plot_type == "zoomed"),
                           "show_kernel": (viz_params.show_per_pixel_processing if plot_type == "main" else True),
                           "show_per_pixel_processing": (plot_type == "zoomed"),
                           "title": f"Zoomed-In {filter_name}" if plot_type == "zoomed" else filter_name}
                    )
                    visualize_image(filter_data, viz_params.ui_placeholders[plot_key], config=config)
        else:
            # Display a placeholder or loading message for unavailable filters
            plot_key = f"{filter_name.lower().replace(' ', '_')}"
            if plot_key in viz_params.ui_placeholders:
                viz_params.ui_placeholders[plot_key].info(f"{filter_name} is not yet available.")

    last_x, last_y = viz_params.last_processed_pixel
    specific_params = {
        "kernel_size": viz_params.results.kernel_size,
        "pixels_processed": viz_params.results.pixels_processed,
        "total_pixels": viz_params.results.kernel_size**2,
        "x": last_x,
        "y": last_y,
        "image_height": viz_params.image_array.data.shape[0],
        "image_width": viz_params.image_array.data.shape[1],
        "half_kernel": viz_params.kernel.size // 2,
        "valid_height": viz_params.image_array.data.shape[0] - viz_params.kernel.size + 1,
        "valid_width": viz_params.image_array.data.shape[1] - viz_params.kernel.size + 1,
        "search_window_size": viz_params.search_window.size,
        "kernel_matrix": viz_params.kernel.kernel_matrix,
        "original_value": viz_params.original_pixel_value,
        "analysis_type": viz_params.technique,
    }

    if viz_params.technique == "nlm":
        specific_params.update({
            "filter_strength": viz_params.results.filter_strength,
            "search_window_size": viz_params.results.search_window_size,
            "nlm_value": viz_params.results.nonlocal_means[last_y, last_x],
        })
    else:  # speckle
        specific_params.update({
            "mean": viz_params.results.mean_filter[last_y, last_x],
            "std": viz_params.results.std_dev_filter[last_y, last_x],
            "sc": viz_params.results.speckle_contrast_filter[last_y, last_x],
        })

    specific_params = prepare_variables(specific_params, viz_params.technique)

    display_analysis_formula(
        specific_params,
        viz_params.ui_placeholders,
        viz_params.technique,
        last_x,
        last_y,
        viz_params.kernel.size,
        viz_params.kernel.kernel_matrix,
        viz_params.original_pixel_value,
    )

