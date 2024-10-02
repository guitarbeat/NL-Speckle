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
        return [np.zeros_like(self.image, dtype=np.float32) for _ in range(num_images)]

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

    ########################
    #    LSCI Processing   #
    ########################

    def process_lsci_pixel(
        self, y_center: int, x_center: int
    ) -> Tuple[int, int, float, float, float]:
        """Process a single pixel using LSCI filtering."""
        # Extract the window (patch) centered at (y_center, x_center)
        # This corresponds to the Border Handling formula:
        # P_{x,y}(i,j) = I_{x+i,y+j} if (x+i,y+j) ∈ valid region, 0 otherwise
        patch_center = extract_window(
            y_center, x_center, self.shared_config["kernel_size"], self.image
        )

        # Compute mean and standard deviation
        # This implements the Mean Filter and Standard Deviation Calculation formulas:
        # μ_{x,y} = (1/N) ∑_{i,j ∈ K_{x,y}} I_{i,j}
        # σ_{x,y} = sqrt((1/N) ∑_{i,j ∈ K_{x,y}} (I_{i,j} - μ_{x,y})^2)
        mean_intensity, std_intensity = compute_statistics(patch_center)

        # Calculate the Speckle Contrast
        # This implements the Speckle Contrast Calculation formula:
        # SC_{x,y} = σ_{x,y} / μ_{x,y}
        speckle_contrast = std_intensity / mean_intensity if mean_intensity != 0 else 0

        # Return the results: coordinates, mean, standard deviation, and speckle contrast
        return y_center, x_center, float(mean_intensity), float(std_intensity), float(speckle_contrast)

    def format_lsci_result(self) -> Dict[str, Any]:
        """Format LSCI-specific results."""
        mean_filter, std_dev_filter, lsci_filter = self.result_images
        return self._create_filter_data(
            "LSCI", mean_filter, std_dev_filter, lsci_filter
        )

    ########################
    #    NLM Processing    #
    ########################

    def process_nlm_pixel(
        self, y_center: int, x_center: int
    ) -> Tuple[int, int, float, float, float]:
        """Process a single pixel using NL Means (NLM) filtering."""
        kernel_size = self.shared_config["kernel_size"]
        search_window_size = self.shared_config["search_window_size"]
        h = self.params["filter_strength"]
        use_full_image = self.shared_config["use_full_image"]

        # Define the search window Ω(x,y)
        # This corresponds to the Search Window formula:
        # Ω(x,y) = I if search_size = 'full', [(x-s,y-s), (x+s,y+s)] ∩ valid region otherwise
        search_radius = (
            max(self.height, self.width) if use_full_image else search_window_size // 2
        )

        y_start = max(0, y_center - search_radius)
        y_end = min(self.height, y_center + search_radius + 1)
        x_start = max(0, x_center - search_radius)
        x_end = min(self.width, x_center + search_radius + 1)

        # Extract the patch centered at (y_center, x_center)
        # This corresponds to the Patch Analysis step:
        # P_{x,y} centered at (x,y)
        patch_center = extract_window(y_center, x_center, kernel_size, self.image)

        weights = []
        neighbor_coords = []
        for y_neighbor in range(y_start, y_end):
            for x_neighbor in range(x_start, x_end):
                # Extract neighboring patch
                patch_neighbor = extract_window(
                    y_neighbor, x_neighbor, kernel_size, self.image
                )
                # Calculate weight
                # This implements the Weight Calculation formula:
                # w_{x,y}(i,j) = exp(-|P_{x,y} - P_{i,j}|^2 / h^2)
                weight = calculate_weights(patch_center, patch_neighbor, h)
                weights.append(weight)
                neighbor_coords.append((y_neighbor, x_neighbor))

        if not weights:
            st.error(f"No weights calculated for pixel ({y_center}, {x_center})")
            return y_center, x_center, self.image[y_center, x_center], 0, 0

        # Calculate the normalization factor C_{x,y}
        # This corresponds to the Normalization Factor formula:
        # C_{x,y} = ∑_{i,j ∈ Ω(x,y)} w_{x,y}(i,j)
        normalization_factor = sum(weights)

        # Calculate the NLM value
        # This implements the main NLM Calculation formula:
        # NLM_{x,y} = (1 / C_{x,y}) * ∑_{i,j ∈ Ω_{x,y}} I_{i,j} * w_{x,y}(i,j)
        weighted_sum = sum(w * self.image[y, x] for w, (y, x) in zip(weights, neighbor_coords))

        nlm_value = (
            weighted_sum / normalization_factor if normalization_factor > 0 else self.image[y_center, x_center]
        )
        average_weight = normalization_factor / len(weights) if weights else 0
        max_similarity = max(weights) if weights else 0

        return y_center, x_center, nlm_value, average_weight, max_similarity

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

    ########################
    #    Shared Methods    #
    ########################

    def _create_filter_data(
        self, technique: str, *filter_images: np.ndarray
    ) -> Dict[str, Any]:
        """Create a dictionary of filter data for the given technique."""
        filter_names = {
            "NLM": ["NL Means", "Normalization Factors", "Last Similarity Map"],
            "LSCI": ["Mean Filter", "Std Dev Filter", "LSCI"],
        }

        filter_data = {
            name: image for name, image in zip(filter_names[technique], filter_images)
        }

        result = {
            key.lower().replace(" ", "_"): value for key, value in filter_data.items()
        }
        result["filter_data"] = filter_data

        if technique == "NLM":
            result.update(
                {
                    "search_window_size": self.shared_config["search_window_size"],
                    "filter_strength": self.params["filter_strength"],
                }
            )

        return result

    def run_parallel_processing(self) -> Dict[str, Any]:
        """Run the image processing function in parallel."""
        args_list = self.create_args_list()
        progress_bar, status = st.progress(0), st.empty()
        num_processes = min(cpu_count(), st.session_state.get("max_workers", 4))
        start_time = time.time()

        try:
            with st.spinner(
                f"Processing {self.technique.upper()}..."
            ), ProcessPoolExecutor(max_workers=num_processes) as executor:
                self._process_pixels(
                    executor, args_list, progress_bar, status, start_time
                )
        except Exception as e:
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
        st.error(f"An error occurred during image processing: {e}")
        return {}