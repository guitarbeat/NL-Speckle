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
    """Applies an image processing technique to an input image.
    Parameters:
        - params (object): An object containing 'technique', 'analysis_params', and 'image_array' attributes.
    Returns:
        - tuple: A tuple containing the updated params and the results from the image processing function.
    Processing Logic:
        - Default values are retrieved from `st.session_state` and then updated based on `analysis_params`.
        - Depending on the specified technique, the applicable image processing function (`process_nlm` or `process_speckle`) is called.
        - Processed results are added to `st.session_state`.
        - Detailed logging is performed if an exception is encountered."""
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
            {"processed_pixels": pixels_to_process,
                f"{technique}_results": results}
        )
        return params, results

    except Exception as e:
        logging.error(
            json.dumps(
                {"action": "process_image",
                    "technique": technique, "error": str(e)}
            )
        )
        raise


def normalize_image(
    image: np.ndarray, low_percentile: int = 2, high_percentile: int = 98
) -> np.ndarray:
    """Normalize the intensity range of an image using percentile clipping.
    Parameters:
        - image (np.ndarray): The input image to be normalized.
        - low_percentile (int): Lower percentile bound for intensity clipping.
        - high_percentile (int): Upper percentile bound for intensity clipping.
    Returns:
        - np.ndarray: The normalized image with intensities scaled between 0 and 1.
    Processing Logic:
        - Percentiles for clipping are determined from the input image intensities.
        - Image intensities are clipped to the calculated lower and upper percentiles.
        - The clipped image intensities are scaled so that the range spans from 0 to 1."""
    p_low, p_high = np.percentile(image, [low_percentile, high_percentile])
    logging.info(
        json.dumps({"action": "normalize_image",
                   "p_low": p_low, "p_high": p_high})
    )
    return np.clip(image, p_low, p_high) - p_low / (p_high - p_low)


def extract_kernel_from_image(
    image_array: np.ndarray, end_x: int, end_y: int, kernel_size: int
) -> Tuple[np.ndarray, float, int]:
    """Extract a square kernel from a 2D image array and its central pixel value.
    Parameters:
        - image_array (np.ndarray): The input image as a 2D array.
        - end_x (int): The x-coordinate of the central pixel of the kernel.
        - end_y (int): The y-coordinate of the central pixel of the kernel.
        - kernel_size (int): The size of the square kernel to extract.
    Returns:
        - Tuple[np.ndarray, float, int]: A tuple containing the kernel as a 2D array, the value of the kernel's central pixel as a float, and the kernel size.
    Processing Logic:
        - Validates if the extracted kernel is non-empty.
        - Ensures the returned kernel has the requested size, applying padding if necessary."""
    half_kernel = kernel_size // 2
    height, width = image_array.shape

    y_start, y_end = max(
        0, end_y - half_kernel), min(height, end_y + half_kernel + 1)
    x_start, x_end = max(0, end_x - half_kernel), min(width,
                                                      end_x + half_kernel + 1)
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
                (max(0, half_kernel - end_y),
                 max(0, end_y + half_kernel + 1 - height)),
                (max(0, half_kernel - end_x),
                 max(0, end_x + half_kernel + 1 - width)),
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
            """Validates the dimensions for image processing.
            Parameters:
                None
            Returns:
                - None: This function does not return a value and is used for validation only.
            Processing Logic:
                - Ensures the image dimensions are positive.
                - Ensures the kernel size is not too large for the image dimensions.
                - Ensures the number of pixels to process is non-negative."""
            if self.image_dimensions[0] <= 0 or self.image_dimensions[1] <= 0:
                raise ValueError("Image dimensions must be positive.")
            if self.valid_dimensions[0] <= 0 or self.valid_dimensions[1] <= 0:
                raise ValueError(
                    "Kernel size is too large for the given image dimensions."
                )
            if self.pixels_to_process < 0:
                raise ValueError(
                    "Number of pixels to process must be non-negative.")

        _validate_dimensions()

        def _validate_coordinates():
            """Validates if the start and end coordinates are within valid ranges.
            Parameters:
                - self: The instance of the class that contains start_point, end_point, and image_dimensions.
            Raises:
                - ValueError: If start coordinates are negative or if end coordinates exceed image dimensions.
            Processing Logic:
                - start_point and end_point are assumed to be tuples (x, y).
                - image_dimensions is assumed to be a tuple (width, height).
                - Validation checks that start is non-negative and end is within image dimensions."""
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
        raise ValueError(
            "Kernel size is too large for the given image dimensions.")

    pixels_to_process = min(valid_height * valid_width,
                            max_pixels or float("inf"))
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
