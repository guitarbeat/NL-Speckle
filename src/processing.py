# Import necessary modules and functions from other files


from abc import ABC, abstractmethod
from dataclasses import dataclass
import streamlit as st
import logging
import json
import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from src.decor import log_action


@dataclass
class ProcessParams:
    """Holds parameters for image processing."""

    image_array: np.ndarray
    analysis_params: Dict[str, Any]
    show_per_pixel_processing: bool
    technique: str
    update_state: bool
    handle_visualization: bool


def process_image(params):
    """Applies image processing technique based on provided parameters.
    Parameters:
        - params (object): An object with image processing parameters including technique, image_array, and analysis_params.
    Returns:
        - tuple: A tuple containing the params object and the image processing results.
    Processing Logic:
        - Default values are set if specific analysis parameters are not provided.
        - The technique used is determined by the 'technique' parameter within params.
        - The image processing function corresponding to the chosen technique is called with relevant parameters.
        - Results are stored in session state and returned alongside the params object."""
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
    """Normalizes pixel values in an image to a range based on percentile cutoffs.
    Parameters:
        - image (np.ndarray): The input image to normalize.
        - low_percentile (int): The lower percentile for normalization cutoff.
        - high_percentile (int): The upper percentile for normalization cutoff.
    Returns:
        - np.ndarray: The normalized image.
    Processing Logic:
        - The function uses percentiles to determine the cut-off values for normalization.
        - The pixel values below the low percentile are clipped to the low percentile value.
        - The pixel values above the high percentile are clipped to the high percentile value.
        - The resulting pixel values are scaled to be between 0 and 1."""
    p_low, p_high = np.percentile(image, [low_percentile, high_percentile])
    logging.info(
        json.dumps({"action": "normalize_image", "p_low": p_low, "p_high": p_high})
    )
    return np.clip(image, p_low, p_high) - p_low / (p_high - p_low)


def extract_kernel_from_image(
    image_array: np.ndarray, end_x: int, end_y: int, kernel_size: int
) -> Tuple[np.ndarray, float, int]:
    """Extract a square kernel from a 2D image array centered at a given point.
    Parameters:
        - image_array (np.ndarray): The input 2D numpy array representing the image.
        - end_x (int): The x-coordinate of the center of the kernel to be extracted.
        - end_y (int): The y-coordinate of the center of the kernel to be extracted.
        - kernel_size (int): The size of the square kernel to be extracted.
    Returns:
        - Tuple[np.ndarray, float, int]: A tuple containing the extracted kernel as a 2D numpy array, 
    the pixel value of the kernel's center, and the kernel size.
    Processing Logic:
        - Check if the extracted kernel is non-empty before processing.
        - Use padding with the "edge" mode to handle edge cases where the kernel exceeds image boundaries.
        - The function raises a ValueError if an empty kernel is extracted."""
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
            """Validates image processing dimensions.
            Parameters:
                None
            Returns:
                - None: This function does not return a value; it raises an exception if any validations fail.
            Processing Logic:
                - Checks if image dimensions are greater than 0.
                - Verifies that the kernel size is appropriate for the given image dimensions.
                - Ensures the number of pixels to process is non-negative."""
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
            """Validates the start and end coordinates for an operation on an image.
            Parameters:
                - None
            Returns:
                - None: This function does not return a value but raises ValueError on invalid coordinates.
            Processing Logic:
                - Checks if any of the start coordinates are negative.
                - Validates whether the end coordinates are within the boundaries of the image dimensions."""
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


# --- Abstract Base Class for Filter Results ---
@dataclass
class FilterResult(ABC):
    """Abstract base class for various filtering techniques."""

    processing_coord: Tuple[int, int]
    processing_end_coord: Tuple[int, int]
    kernel_size: int
    pixels_processed: int
    image_dimensions: Tuple[int, int]

    @abstractmethod
    def get_filter_data(self) -> Dict[str, Any]:
        """Get filter-specific data as a dictionary."""

    @classmethod
    @abstractmethod
    def get_filter_options(cls) -> List[str]:
        """Get available filter options."""

    def get_last_processed_coordinates(self) -> Tuple[int, int]:
        """Get the last processed pixel coordinates."""
        return self.processing_end_coord
