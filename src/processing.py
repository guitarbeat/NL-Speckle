# Import necessary modules and functions from other files


from abc import ABC, abstractmethod
from dataclasses import dataclass
import streamlit as st
import logging
import json
import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from src.decor import log_action

@log_action
def extract_or_default(source: dict, key: str, default_value):
    """
    Extract a value from the dictionary or return a default if key is not present.

    Args:
        source (dict): The dictionary to extract from.
        key (str): The key to look for.
        default_value: The value to return if key is not found.

    Returns:
        The extracted value or the default value.
    """
    value = source.get(key, default_value)
    logging.info(
        json.dumps({"action": "extract_or_default", "key": key, "value": value})
    )
    return value


@dataclass
class ProcessParams:
    """Holds parameters for image processing."""

    image_array: np.ndarray
    analysis_params: Dict[str, Any]
    show_per_pixel_processing: bool
    technique: str
    update_state: bool
    handle_visualization: bool

@log_action
def normalize_image(
    image: np.ndarray, low_percentile: int = 2, high_percentile: int = 98
) -> np.ndarray:
    """
    Normalize an image using percentile-based scaling.

    Args:
        image (np.ndarray): The image array to normalize.
        low_percentile (int): The lower percentile for clipping.
        high_percentile (int): The upper percentile for clipping.

    Returns:
        np.ndarray: The normalized image array.
    """
    p_low, p_high = np.percentile(image, [low_percentile, high_percentile])
    logging.info(
        json.dumps({"action": "normalize_image", "p_low": p_low, "p_high": p_high})
    )
    return np.clip(image, p_low, p_high) - p_low / (p_high - p_low)

@log_action
def extract_kernel_from_image(
    image_array: np.ndarray, end_x: int, end_y: int, kernel_size: int
) -> Tuple[np.ndarray, float, int]:
    """
    Extract a kernel from an image centered at the given coordinates.

    Args:
        image_array (np.ndarray): The input image.
        end_x (int): X-coordinate for the kernel center.
        end_y (int): Y-coordinate for the kernel center.
        kernel_size (int): Size of the kernel.

    Returns:
        Tuple[np.ndarray, float, int]: The extracted kernel, the original pixel value, and the kernel size.
    """
    half_kernel = kernel_size // 2
    height, width = image_array.shape

    # Determine kernel bounds
    y_start, y_end = max(0, end_y - half_kernel), min(height, end_y + half_kernel + 1)
    x_start, x_end = max(0, end_x - half_kernel), min(width, end_x + half_kernel + 1)
    kernel_values = image_array[y_start:y_end, x_start:x_end]

    # Handle edge cases by padding the kernel if needed
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
def apply_technique(
    technique: str,
    image: np.ndarray,
    kernel_size: int,
    pixels_to_process: int,
    search_window_size: int,
    filter_strength: float,
):
    """
    Apply the specified technique to the image.

    Args:
        technique (str): The processing technique to apply.
        image (np.ndarray): The image array to process.
        kernel_size (int): The size of the kernel.
        pixels_to_process (int): Number of pixels to process.
        search_window_size (int): The size of the search window (for NLM).
        filter_strength (float): The strength of the filter.

    Returns:
        The results of the image processing.
    """
    from src.nlm import process_nlm
    from src.speckle import process_speckle
    if technique == "nlm":
        return process_nlm(
            image=image,
            kernel_size=kernel_size,
            pixels_to_process=pixels_to_process,
            search_window_size=search_window_size,
            filter_strength=filter_strength,
        )
    elif technique == "speckle":
        return process_speckle(image, kernel_size, pixels_to_process)
    else:
        logging.error(
            json.dumps(
                {
                    "action": "apply_technique",
                    "error": f"Unknown technique: {technique}",
                }
            )
        )
        raise ValueError(f"Unknown technique: {technique}")

def process_image(params: ProcessParams):
    """
    Process an image based on the provided parameters.

    Args:
        params (ProcessParams): The processing parameters including image array and techniques.

    Returns:
        tuple: The modified parameters and results.
    """
    try:
        technique = params.technique
        analysis_params = params.analysis_params

        # Extract or assign default parameters
        kernel_size = extract_or_default(st.session_state, "kernel_size", 3)
        pixels_to_process = extract_or_default(analysis_params, "pixels_to_process", 0)
        search_window_size = extract_or_default(
            analysis_params, "search_window_size", 21
        )
        filter_strength = extract_or_default(
            analysis_params, "filter_strength", 0.10
        )

        # Update analysis_params with the resolved values
        analysis_params.update(
            {
                "kernel_size": kernel_size,
                "pixels_to_process": pixels_to_process,
                "search_window_size": search_window_size,
                "filter_strength": filter_strength,
            }
        )

        # Add your processing logic here
        # For example:
        results = apply_technique(
            technique,
            params.image_array.data,
            kernel_size,
            pixels_to_process,
            search_window_size,
            filter_strength,
        )

        return params, results

    except Exception as e:
        logging.error(
            json.dumps(
                {"action": "process_image", "technique": technique, "error": str(e)}
            )
        )
        raise

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
