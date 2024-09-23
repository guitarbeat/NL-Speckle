# Import necessary modules and functions from other files


import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import streamlit as st

from src.decor import log_action


@dataclass
class ProcessParams:
    """Holds parameters for image processing."""

    image_array: np.ndarray
    analysis_params: Dict[str, Any]
    show_per_pixel_processing: bool
    technique: str


def process_image(params):
    try:
        technique = params.technique
        analysis_params = params.analysis_params

        kernel_size = st.session_state.get("kernel_size", 3)
        pixels_to_process = analysis_params.get("pixels_to_process", 0)
        search_window_size = analysis_params.get("search_window_size", 21)
        filter_strength = analysis_params.get("filter_strength", 0.10)

        analysis_params.update(
            {
                "kernel_size": kernel_size,
                "pixels_to_process": pixels_to_process,
                "search_window_size": search_window_size,
                "filter_strength": filter_strength,
            }
        )

        from src.nlm import process_nlm
        from src.speckle import process_speckle

        if technique == "nlm":
            results = process_nlm(
                image=params.image_array.data,
                kernel_size=kernel_size,
                pixels_to_process=pixels_to_process,
                search_window_size=search_window_size,
                filter_strength=filter_strength,
            )
        elif technique == "speckle":
            results = process_speckle(
                params.image_array.data, kernel_size, pixels_to_process
            )
        else:
            raise ValueError(f"Unknown technique: {technique}")

        st.session_state.update(
            {"processed_pixels": pixels_to_process, f"{technique}_results": results}
        )
        return params, results

    except Exception as e:
        logging.error(
            json.dumps(
                {"action": "process_image", "technique": technique, "error": str(e)}
            )
        )
        raise


def normalize_image(
    image: np.ndarray, low_percentile: int = 2, high_percentile: int = 98
) -> np.ndarray:
    p_low, p_high = np.percentile(image, [low_percentile, high_percentile])
    logging.info(
        json.dumps({"action": "normalize_image", "p_low": p_low, "p_high": p_high})
    )
    return np.clip(image, p_low, p_high) - p_low / (p_high - p_low)


def extract_kernel_from_image(
    image_array: np.ndarray, end_x: int, end_y: int, kernel_size: int
) -> Tuple[np.ndarray, float, int]:
    half_kernel = kernel_size // 2
    height, width = image_array.shape

    y_start, y_end = max(0, end_y - half_kernel), min(height, end_y + half_kernel + 1)
    x_start, x_end = max(0, end_x - half_kernel), min(width, end_x + half_kernel + 1)
    kernel_values = image_array[y_start:y_end, x_start:x_end]

    if kernel_values.size == 0:
        raise ValueError(
            json.dumps(
                {
                    "action": "extract_kernel_from_image",
                    "error": f"Extracted kernel at ({end_x}, {end_y}) is empty. Image shape: {image_array.shape}, Kernel size: {kernel_size}",
                }
            )
        )

    if kernel_values.shape != (kernel_size, kernel_size):
        kernel_values = np.pad(
            kernel_values,
            (
                (max(0, half_kernel - end_y), max(0, end_y + half_kernel + 1 - height)),
                (max(0, half_kernel - end_x), max(0, end_x + half_kernel + 1 - width)),
            ),
            mode="edge",
        )

    return kernel_values.astype(float), float(image_array[end_y, end_x]), kernel_size


@log_action
def configure_process_params(
    technique: str, process_params: ProcessParams, technique_params: Dict[str, Any]
) -> None:
    """Configure process parameters based on the technique."""
    if technique == "nlm":
        process_params.analysis_params["use_whole_image"] = technique_params.get(
            "use_whole_image", False
        )


# --- Dataclass for Processing Details ---
@dataclass(frozen=True)
class ProcessingDetails:
    """A dataclass to store image processing details."""

    image_dimensions: Tuple[int, int]
    valid_dimensions: Tuple[int, int]
    start_point: Tuple[int, int]
    end_point: Tuple[int, int]
    pixels_to_process: int
    kernel_size: int

    def __post_init__(self):
        """Validate dimensions and coordinates after initialization."""

        def _validate_dimensions():
            if self.image_dimensions[0] <= 0 or self.image_dimensions[1] <= 0:
                raise ValueError("Image dimensions must be positive.")
            if self.valid_dimensions[0] <= 0 or self.valid_dimensions[1] <= 0:
                raise ValueError(
                    "Kernel size is too large for the given image dimensions."
                )
            if self.pixels_to_process < 0:
                raise ValueError("Number of pixels to process must be non-negative.")

        _validate_dimensions()

        def _validate_coordinates():
            if self.start_point[0] < 0 or self.start_point[1] < 0:
                raise ValueError("Start coordinates must be non-negative.")
            if (
                self.end_point[0] >= self.image_dimensions[0]
                or self.end_point[1] >= self.image_dimensions[1]
            ):
                raise ValueError("End coordinates exceed image boundaries.")

        _validate_coordinates()


def calculate_processing_details(
    image: np.ndarray, kernel_size: int, max_pixels: Optional[int]
) -> ProcessingDetails:
    """Calculate processing details for the given image and kernel size."""
    image_height, image_width = image.shape[:2]
    half_kernel = kernel_size // 2
    valid_height, valid_width = (
        image_height - kernel_size + 1,
        image_width - kernel_size + 1,
    )

    if valid_height <= 0 or valid_width <= 0:
        raise ValueError("Kernel size is too large for the given image dimensions.")

    pixels_to_process = min(valid_height * valid_width, max_pixels or float("inf"))
    end_y, end_x = divmod(pixels_to_process - 1, valid_width)
    end_y, end_x = end_y + half_kernel, end_x + half_kernel

    return ProcessingDetails(
        image_dimensions=(image_width, image_height),  # Tuple
        valid_dimensions=(valid_width, valid_height),
        start_point=(half_kernel, half_kernel),
        end_point=(end_x, end_y),
        pixels_to_process=pixels_to_process,
        kernel_size=kernel_size,
    )


@dataclass
class FilterResult(ABC):
    """Abstract base class for various filtering techniques."""

    processing_end_coord: Tuple[int, int]
    kernel_size: int
    pixels_processed: int
    image_dimensions: Tuple[int, int]

    @abstractmethod
    def get_filter_data(self) -> Dict[str, Any]:
        """Get filter-specific data as a dictionary."""

    @staticmethod
    @abstractmethod
    def get_filter_options() -> List[str]:
        """Get available filter options."""

    def get_last_processed_coordinates(self) -> Tuple[int, int]:
        """Get the last processed pixel coordinates."""
        return self.processing_end_coord