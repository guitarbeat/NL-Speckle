"""
Utility functions and classes for image processing and comparison.

Includes:
- ImageComparison: Handles image comparison in Streamlit.
- FilterResult: Abstract base class for filtering techniques.
- Point and Dimensions: Named tuples for coordinates and dimensions.
- ProcessingDetails: Dataclass for storing image processing details.
- calculate_processing_details: Calculates processing details for an image.
"""

# --- Imports ---
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, NamedTuple
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from streamlit_image_comparison import image_comparison


# --- Type Aliases ---
ImageArray = np.ndarray
PixelCoordinates = Tuple[int, int]


# --- Common Error Handling ---
def handle_error(message: str):
    """Handle errors and display an error message in Streamlit."""
    st.error(f"{message}. Please check the logs.")


# --- Image Comparison Class ---
@dataclass
class ImageComparison:
    """Class for handling image comparison functionality."""

    @staticmethod
    def handle(tab, cmap_name, images):
        """Handle image comparison in a Streamlit tab."""
        try:
            with tab:
                st.header("Image Comparison")
                if not images:
                    st.warning("No images available for comparison.")
                    return
                available_images = list(images.keys())

                def get_image_choices():
                    col1, col2 = st.columns(2)
                    image_choice_1 = col1.selectbox(
                        "Select first image to compare:",
                        [""] + available_images,
                        index=0,
                    )
                    image_choice_2 = col2.selectbox(
                        "Select second image to compare:",
                        [""] + available_images,
                        index=0,
                    )
                    return image_choice_1, image_choice_2

                image_choice_1, image_choice_2 = get_image_choices()

                if image_choice_1 and image_choice_2:
                    img1, img2 = images[image_choice_1], images[image_choice_2]

                    def display():
                        if image_choice_1 == image_choice_2:

                            def normalize_and_colorize(images, cmap_names):
                                colored_images = []
                                for img, cmap_name in zip(images, cmap_names):
                                    normalized = (
                                        (img - np.min(img))
                                        / (np.max(img) - np.min(img))
                                        if np.max(img) - np.min(img) != 0
                                        else img
                                    )
                                    colored = plt.get_cmap(cmap_name)(normalized)[
                                        :, :, :3
                                    ]
                                    colored_images.append(
                                        (colored * 255).astype(np.uint8)
                                    )
                                return colored_images

                            def process_normalized_images(normalized_images):
                                if not isinstance(normalized_images, list):
                                    st.error("Expected a list of normalized images.")
                                    return None, None

                                if len(normalized_images) != 2:
                                    st.error(
                                        f"Got {len(normalized_images)}."
                                    )
                                    return None, None

                                return normalized_images[0], normalized_images[1]

                            normalized_images = normalize_and_colorize(
                                [img1, img2], [cmap_name] * 2
                            )
                            img1_uint8, img2_uint8 = process_normalized_images(
                                normalized_images
                            )

                            image_comparison(
                                img1=img1_uint8,
                                img2=img2_uint8,
                                label1=image_choice_1,
                                label2=image_choice_2,
                            )
                            st.subheader("Selected Images")
                            st.image(
                                [img1_uint8, img2_uint8],
                                caption=[image_choice_1, image_choice_2],
                            )
                        else:
                            st.error(
                                "Please select two different images for comparison."
                            )

                            def display_difference_map():
                                diff_map = np.abs(img1 - img2)
                                display_diff = normalize_and_colorize(
                                    [diff_map], [cmap_name]
                                )[0]
                                st.image(
                                    display_diff,
                                    caption="Difference Map",
                                    use_column_width=True,
                                )

                            display_difference_map()

                    display()
                else:
                    st.info("Select two images to compare.")
        except (KeyError, ValueError, TypeError) as e:
            handle_error(f"Error while handling image comparison: {e}")


# --- Abstract Base Class for Filter Results ---
@dataclass
class FilterResult(ABC):
    """Abstract base class for various filtering techniques."""

    processing_coord: PixelCoordinates
    processing_end_coord: PixelCoordinates
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

    def get_last_processed_coordinates(self) -> PixelCoordinates:
        """Get the last processed pixel coordinates."""
        return self.processing_end_coord


# --- Named Tuples for Coordinates and Dimensions ---
class Point(NamedTuple):
    """A named tuple representing a point with x and y coordinates."""

    x: int
    y: int


class Dimensions(NamedTuple):
    """A named tuple representing dimensions with width and height."""

    width: int
    height: int


# --- Dataclass for Processing Details ---
@dataclass(frozen=True)
class ProcessingDetails:
    """A dataclass to store image processing details."""

    image_dimensions: Dimensions
    valid_dimensions: Dimensions
    start_point: Point
    end_point: Point
    pixels_to_process: int
    kernel_size: int

    def __post_init__(self):
        """Validate dimensions and coordinates after initialization."""

        def _validate_dimensions():
            if self.image_dimensions.width <= 0 or self.image_dimensions.height <= 0:
                raise ValueError("Image dimensions must be positive.")
            if self.valid_dimensions.width <= 0 or self.valid_dimensions.height <= 0:
                raise ValueError(
                    "Kernel size is too large for the given image dimensions."
                )
            if self.pixels_to_process < 0:
                raise ValueError("Number of pixels to process must be non-negative.")

        _validate_dimensions()

        def _validate_coordinates():
            if self.start_point.x < 0 or self.start_point.y < 0:
                raise ValueError("Start coordinates must be non-negative.")
            if (
                self.end_point.x >= self.image_dimensions.width
                or self.end_point.y >= self.image_dimensions.height
            ):
                raise ValueError("End coordinates exceed image boundaries.")

        _validate_coordinates()


# --- Function to Calculate Processing Details ---
def calculate_processing_details(
    image: ImageArray, kernel_size: int, max_pixels: Optional[int]
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
        image_dimensions=Dimensions(width=image_width, height=image_height),
        valid_dimensions=Dimensions(width=valid_width, height=valid_height),
        start_point=Point(x=half_kernel, y=half_kernel),
        end_point=Point(x=end_x, y=end_y),
        pixels_to_process=pixels_to_process,
        kernel_size=kernel_size,
    )
