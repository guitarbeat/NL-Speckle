"""
This module handles image processing and visualization for the Streamlit application.
"""

import numpy as np
import streamlit as st
import src.session_state as session_state
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count
import time
from typing import Dict, Any, List, Tuple
import logging
from functools import partial

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


## Utility Functions ##


def extract_window(
    y_coord: int, x_coord: int, kernel_size: int, image: np.ndarray
) -> np.ndarray:
    """
    Extract a window from the image centered at (y_coord, x_coord) with the given kernel size.
    Pads with edge values if necessary.
    """
    half_kernel = kernel_size // 2
    height, width = image.shape

    top = max(0, y_coord - half_kernel)
    bottom = min(height, y_coord + half_kernel + 1)
    left = max(0, x_coord - half_kernel)
    right = min(width, x_coord + half_kernel + 1)

    window = image[top:bottom, left:right]

    # Pad the window if it's smaller than the kernel size
    if window.shape != (kernel_size, kernel_size):
        padded_window = np.pad(
            window,
            (
                (
                    max(0, half_kernel - y_coord),
                    max(0, y_coord + half_kernel + 1 - height),
                ),
                (
                    max(0, half_kernel - x_coord),
                    max(0, x_coord + half_kernel + 1 - width),
                ),
            ),
            mode="edge",
        )
        window = padded_window[:kernel_size, :kernel_size]

    return window


def compute_statistics(window: np.ndarray) -> Tuple[float, float]:
    """Compute mean and standard deviation of the given window."""
    mean = np.nanmean(window)
    std = np.nanstd(window)
    return mean, std


def calculate_weights(
    patch_xy: np.ndarray, patch_ij: np.ndarray, filter_strength: float
) -> float:
    """Calculate the weight between two patches."""
    squared_diff = np.sum((patch_xy - patch_ij) ** 2)
    weight = np.exp(-squared_diff / (filter_strength**2))
    return weight


def process_pixel_wrapper(args: Tuple[Any, ...]) -> Any:
    """Wrapper to process a pixel with error handling."""
    try:
        return args[0](*args[1:])
    except Exception as e:
        st.error(f"Error processing pixel: {e}")
        return None


## Shared Configuration ##


def create_shared_config(technique: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """Create a shared configuration for both processing and overlays."""
    nlm_options = session_state.get_nlm_options()
    image_array = session_state.get_image_array()
    height, width = image_array.shape
    kernel_size = params["kernel_size"]
    half_kernel = kernel_size // 2

    config = {
        "technique": technique,
        "kernel_size": kernel_size,
        "half_kernel": half_kernel,
        "search_window_size": params.get(
            "search_window_size", nlm_options["search_window_size"]
        ),
        "use_full_image": params.get("use_full_image", nlm_options["use_whole_image"]),
        "show_per_pixel_processing": session_state.get_session_state(
            "show_per_pixel", False
        ),
        "image_shape": (height, width),
        "total_pixels": image_array.size,
        "pixels_to_process": session_state.get_session_state(
            "pixels_to_process", image_array.size
        ),
        "pixel_percentage": (
            session_state.get_session_state("pixels_to_process", image_array.size)
            / image_array.size
        )
        * 100,
        "processable_area": {
            "top": half_kernel,
            "bottom": height - half_kernel,
            "left": half_kernel,
            "right": width - half_kernel,
        },
        # Define the total area
        "total_area": {
            "top": 0,
            "bottom": height,
            "left": 0,
            "right": width,
        },
    }

    return config


## Image Processing ##


class ImageProcessor:
    """A class to handle different image processing techniques."""

    def __init__(self, image: np.ndarray, technique: str, params: Dict[str, Any]):
        self.image = image
        self.technique = technique
        self.params = params
        self.height, self.width = image.shape
        self.shared_config = create_shared_config(technique, params)
        self.result_images = self.initialize_result_images()
        self.pixels = self.get_pixels_for_processing()
        self.current_pixel = None
        self.last_processed_pixel = None

    def initialize_result_images(self) -> List[np.ndarray]:
        """Initialize result images based on the technique."""
        num_images = 5 if self.technique == "nlm" else 3
        return [np.full_like(self.image, np.nan, dtype=np.float32) for _ in range(num_images)]

    def get_pixels_for_processing(self) -> List[Tuple[int, int]]:
        """Retrieve the list of pixels to process."""
        processable_area = self.shared_config["processable_area"]
        all_pixels = [
            (y_coord, x_coord)
            for y_coord in range(processable_area["top"], processable_area["bottom"])
            for x_coord in range(processable_area["left"], processable_area["right"])
        ]
        return all_pixels[: self.shared_config["pixels_to_process"]]

    def create_args_list(self) -> List[Tuple[Any, ...]]:
        """Create a list of arguments for processing pixels."""
        return [
            (
                self.process_pixel,
                y_coord,
                x_coord,
            )
            for y_coord, x_coord in self.pixels
        ]

    def process_pixel(self, y_coord: int, x_coord: int):
        """Process a single pixel based on the technique."""
        self.current_pixel = (y_coord, x_coord)
        processor = getattr(self, f"process_{self.technique}_pixel")
        return processor(y_coord, x_coord)

    def process_lsci_pixel(
        self, y_coord: int, x_coord: int
    ) -> Tuple[int, int, float, float, float]:
        """Process a single pixel using LSCI filtering."""
        window = extract_window(
            y_coord, x_coord, self.shared_config["kernel_size"], self.image
        )
        mean, std = compute_statistics(window)
        std_ratio = std / mean if mean != 0 else 0
        return y_coord, x_coord, float(mean), float(std), float(std_ratio)

    def process_nlm_pixel(
        self, y_coord: int, x_coord: int
    ) -> Tuple[int, int, float, float, float, float]:
        """Process a single pixel using NL Means (NLM) filtering."""
        kernel_size = self.shared_config["kernel_size"]
        search_window_size = self.shared_config["search_window_size"]
        filter_strength = self.params["filter_strength"]
        use_full_image = self.shared_config["use_full_image"]

        search_radius = max(self.height, self.width) if use_full_image else search_window_size // 2

        y_start, y_end = max(0, y_coord - search_radius), min(self.height, y_coord + search_radius + 1)
        x_start, x_end = max(0, x_coord - search_radius), min(self.width, x_coord + search_radius + 1)

        patch_xy = extract_window(y_coord, x_coord, kernel_size, self.image)

        weights_and_neighbors = [
            (calculate_weights(patch_xy, extract_window(y, x, kernel_size, self.image), filter_strength), (y, x))
            for y in range(y_start, y_end)
            for x in range(x_start, x_end)
        ]

        if not weights_and_neighbors:
            st.error(f"No weights calculated for pixel ({y_coord}, {x_coord})")
            return y_coord, x_coord, self.image[y_coord, x_coord], 0, 0, 0

        weights, neighbors = zip(*weights_and_neighbors)
        weights_sum = sum(weights)
        pixel_sum = sum(w * self.image[y, x] for w, (y, x) in weights_and_neighbors)

        nlm_value = pixel_sum / weights_sum if weights_sum > 0 else self.image[y_coord, x_coord]
        weight_avg = weights_sum / len(weights)

        return y_coord, x_coord, nlm_value, weight_avg, 0, max(weights)

    def run_parallel_processing(self) -> Dict[str, Any]:
        """Run the image processing function in parallel."""
        args_list = self.create_args_list()
        progress_bar, status = st.progress(0), st.empty()
        num_processes = min(cpu_count(), st.session_state.get("max_workers", 4))
        start_time = time.time()

        try:
            with st.spinner(f"Processing {self.technique.upper()}..."):
                with ProcessPoolExecutor(max_workers=num_processes) as executor:
                    process_func = partial(self._process_pixels, executor, args_list, progress_bar, status, start_time)
                    executor.submit(process_func).result()
        except Exception as e:
            logger.error(f"Error during parallel processing: {e}")
            st.error(f"An error occurred during processing: {e}")

        processing_end = self.current_pixel or (0, 0)
        return self.format_result(processing_end)

    def _process_pixels(
        self,
        executor: ProcessPoolExecutor,
        args_list: List[Tuple[Any, ...]],
        progress_bar: Any,
        status: st.delta_generator.DeltaGenerator,
        start_time: float,
    ):
        total_pixels = len(self.pixels)
        for i, result in enumerate(executor.map(process_pixel_wrapper, args_list)):
            if result is not None:
                self._handle_pixel_result(result)
            self._update_progress(i + 1, total_pixels, start_time, progress_bar, status)

    def _handle_pixel_result(self, result: Tuple[int, int, float, float, float]):
        y_coord, x_coord, *values = result
        for j, value in enumerate(values):
            if self._is_valid_pixel(y_coord, x_coord):
                self.result_images[j][y_coord, x_coord] = float(value)
                self.last_processed_pixel = (y_coord, x_coord)

    def _is_valid_pixel(self, y_coord: int, x_coord: int) -> bool:
        return 0 <= y_coord < self.height and 0 <= x_coord < self.width

    def _update_progress(
        self,
        current: int,
        total: int,
        start_time: float,
        progress_bar: Any,
        status: st.delta_generator.DeltaGenerator,
    ):
        progress = current / total
        progress_bar.progress(progress)
        if progress > 0:
            elapsed_time = time.time() - start_time
            estimated_total_time = elapsed_time / progress
            remaining_time = estimated_total_time - elapsed_time
            status.text(
                f"Processed {current}/{total} pixels. Estimated time remaining: {remaining_time:.2f} seconds"
            )
        else:
            status.text("Initializing processing...")

    def format_result(self, processing_end: Tuple[int, int]) -> Dict[str, Any]:
        """Format the processing results."""
        base_result = {
            "processing_end_coord": processing_end,
            "kernel_size": self.shared_config["kernel_size"],
            "pixels_processed": len(self.pixels),
            "image_dimensions": (self.height, self.width),
            "processable_area": self.shared_config["processable_area"],
            "last_processed_pixel": self.last_processed_pixel,
            "last_pixel_intensity": self.image[self.last_processed_pixel],
        }

        technique_specific_result = getattr(self, f"format_{self.technique}_result")()
        return {**base_result, **technique_specific_result}

    def format_nlm_result(self) -> Dict[str, Any]:
        """Format NLM-specific results."""
        if len(self.result_images) < 3:
            st.error(
                f"Expected at least 3 result images for NLM, but got {len(self.result_images)}"
            )
            return {}

        nlm_image, normalization_factors, last_similarity_map = self.result_images[:3]
        return self._create_filter_data(
            "NLM", nlm_image, normalization_factors, last_similarity_map
        )

    def format_lsci_result(self) -> Dict[str, Any]:
        """Format LSCI-specific results."""
        mean_filter, std_dev_filter, lsci_filter = self.result_images
        return self._create_filter_data(
            "LSCI", mean_filter, std_dev_filter, lsci_filter
        )

    @staticmethod
    def _create_filter_data(
        technique: str, *filter_images: np.ndarray
    ) -> Dict[str, Any]:
        """Create a dictionary of filter data for the given technique."""
        filter_names = {
            "NLM": ["NL Means", "Normalization Factors", "Last Similarity Map"],
            "LSCI": ["Mean Filter", "Std Dev Filter", "LSCI"],
        }

        filter_data = dict(zip(filter_names[technique], filter_images))

        result = {key.lower().replace(" ", "_"): value for key, value in filter_data.items()}
        result["filter_data"] = filter_data

        return result


@st.cache_data(show_spinner=False)
def apply_processing(
    image: np.ndarray,
    technique: str,
    params: Dict[str, Any],
    pixels_to_process: int,
    kernel_size: int,
) -> Dict[str, Any]:
    """Apply the specified processing technique to the image."""
    params["pixels_to_process"] = pixels_to_process
    params["kernel_size"] = kernel_size

    try:
        processor = ImageProcessor(image, technique, params)
        result = processor.run_parallel_processing()
        return result
    except Exception as e:
        logger.error(f"Error during image processing: {e}")
        st.error(f"An error occurred during image processing: {e}")
        return {}
