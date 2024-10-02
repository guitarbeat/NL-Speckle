"""
This module handles image visualization for the Streamlit application.
"""

import numpy as np
import streamlit as st
import src.session_state as session_state
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count
import time


## Utility Functions ##


def extract_window(y_coord, x_coord, kernel_size, image):
    """Extract a window from the image centered at (y_coord, x_coord) with the given kernel size, padding with edge values if necessary."""
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
                (max(0, half_kernel - y_coord), max(0, y_coord + half_kernel + 1 - height)),
                (max(0, half_kernel - x_coord), max(0, x_coord + half_kernel + 1 - width)),
            ),
            mode="edge",
        )
        window = padded_window[:kernel_size, :kernel_size]

    return window


def compute_statistics(window):
    """Compute mean and standard deviation of the given window."""
    mean = np.nanmean(window)
    std = np.nanstd(window)
    return mean, std


def calculate_weights(patch_xy, patch_ij, filter_strength):
    """Calculate the weight between two patches."""
    squared_diff = (patch_xy - patch_ij) ** 2
    weight = np.exp(-np.sum(squared_diff) / (filter_strength**2))
    return weight


def process_pixel_wrapper(args):
    """Wrapper to process a pixel with error handling."""
    try:
        return args[0](*args[1:])
    except Exception as e:
        print(f"Error processing pixel: {str(e)}")
        return None


## Shared Configuration ##


def create_shared_config(technique, params):
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
        "pixel_percentage": (session_state.get_session_state(
            "pixels_to_process", image_array.size
        ) / image_array.size) * 100,
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
        }
    }

    return config


## Image Processing ##


class ImageProcessor:
    """A class to handle different image processing techniques."""

    def __init__(self, image, technique, params):
        self.image = image
        self.technique = technique
        self.params = params
        self.height, self.width = image.shape
        self.shared_config = create_shared_config(technique, params)
        self.result_images = self.initialize_result_images()
        self.pixels = self.get_pixels_for_processing()
        self.current_pixel = None
        self.last_processed_pixel = None

    def initialize_result_images(self):
        """Initialize result images based on the technique."""
        return [
            np.zeros_like(self.image, dtype=np.float32)
            for _ in range(5 if self.technique == "nlm" else 3)
        ]

    def get_pixels_for_processing(self):
        """Retrieve the list of pixels to process."""
        processable_area = self.shared_config["processable_area"]
        all_pixels = [
            (y_coord, x_coord)
            for y_coord in range(processable_area["top"], processable_area["bottom"])
            for x_coord in range(processable_area["left"], processable_area["right"])
        ]
        return all_pixels[: self.shared_config["pixels_to_process"]]

    def create_args_list(self):
        """Create a list of arguments for processing pixels."""
        return [
            (
                self.process_pixel,
                y_coord,
                x_coord,
            )
            for y_coord, x_coord in self.pixels
        ]

    def process_pixel(self, y_coord, x_coord):
        """Process a single pixel based on the technique."""
        self.current_pixel = (y_coord, x_coord)
        if self.technique == "nlm":
            return self.process_nlm_pixel(y_coord, x_coord)
        elif self.technique == "lsci":
            return self.process_lsci_pixel(y_coord, x_coord)
        else:
            raise ValueError(f"Unknown technique: {self.technique}")

    def process_lsci_pixel(self, y_coord, x_coord):
        """Process a single pixel using LSCI filtering."""
        kernel_size = self.shared_config["kernel_size"]
        window = extract_window(y_coord, x_coord, kernel_size, self.image)
        mean, std = compute_statistics(window)
        std_ratio = std / mean if mean != 0 else 0
        return y_coord, x_coord, mean, std, std_ratio

    def process_nlm_pixel(self, y_coord, x_coord):
        """Process a single pixel using NL Means (NLM) filtering."""
        kernel_size = self.shared_config["kernel_size"]
        search_window_size = self.shared_config["search_window_size"]
        filter_strength = self.params["filter_strength"]
        use_full_image = self.shared_config["use_full_image"]

        search_radius = max(self.height, self.width) if use_full_image else search_window_size // 2

        y_start = max(0, y_coord - search_radius)
        y_end = min(self.height, y_coord + search_radius + 1)
        x_start = max(0, x_coord - search_radius)
        x_end = min(self.width, x_coord + search_radius + 1)

        patch_xy = extract_window(y_coord, x_coord, kernel_size, self.image)

        weights = []
        neighbors = []
        for y_neighbor in range(y_start, y_end):
            for x_neighbor in range(x_start, x_end):
                if (y_neighbor, x_neighbor) == (y_coord, x_coord):
                    continue
                patch_neighbor = extract_window(y_neighbor, x_neighbor, kernel_size, self.image)
                weight = calculate_weights(patch_xy, patch_neighbor, filter_strength)
                weights.append(weight)
                neighbors.append((y_neighbor, x_neighbor))

        if not weights:
            return y_coord, x_coord, self.image[y_coord, x_coord], 0, 0, 0, 0

        weights_sum = sum(weights)
        pixel_sum = sum(
            w * self.image[y, x]
            for w, (y, x) in zip(weights, neighbors)
        )

        nlm_value = pixel_sum / weights_sum
        weight_avg = weights_sum / len(weights)
        weight_std = np.sqrt(
            sum(w**2 for w in weights) / len(weights) - weight_avg**2
        )

        return y_coord, x_coord, nlm_value, weight_avg, weight_std, max(weights), min(weights)

    def run_parallel_processing(self):
        """Run the image processing function in parallel."""
        args_list = self.create_args_list()
        progress_bar, status = st.progress(0), st.empty()
        num_processes = min(cpu_count(), st.session_state.get("max_workers", 4))
        start_time = time.time()

        with st.spinner(f"Processing {self.technique.upper()}..."), ProcessPoolExecutor(
            max_workers=num_processes
        ) as executor:
            self._process_pixels(executor, args_list, progress_bar, status, start_time)

        processing_end = self.current_pixel or (0, 0)
        return self.format_result(processing_end)

    def _process_pixels(self, executor, args_list, progress_bar, status, start_time):
        total_pixels = len(self.pixels)
        for i, result in enumerate(executor.map(process_pixel_wrapper, args_list)):
            if result is not None:
                self._handle_pixel_result(result)
            self._update_progress(i + 1, total_pixels, start_time, progress_bar, status)

    def _handle_pixel_result(self, result):
        y_coord, x_coord, *values = result
        for j, value in enumerate(values):
            if self._is_valid_pixel(y_coord, x_coord):
                self.result_images[j][y_coord, x_coord] = value
                self.last_processed_pixel = (y_coord, x_coord)

    def _is_valid_pixel(self, y_coord, x_coord):
        return 0 <= y_coord < self.height and 0 <= x_coord < self.width

    def _update_progress(self, current, total, start_time, progress_bar, status):
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

    def format_result(self, processing_end):
        """Format the processing results."""
        base_result = {
            "processing_end_coord": processing_end,
            "kernel_size": self.shared_config["kernel_size"],
            "pixels_processed": len(self.pixels),
            "image_dimensions": (self.height, self.width),
            "processable_area": self.shared_config["processable_area"],
            "last_processed_pixel": self.last_processed_pixel,
        }

        if self.technique == "nlm":
            (
                nlm_image,
                normalization_factors,
                nl_std_image,
                nl_lsci_image,
                last_similarity_map,
            ) = self.result_images
            return {
                **base_result,
                "NL_means": nlm_image,
                "normalization_factors": normalization_factors,
                "NL_std": nl_std_image,
                "NL_lsci": nl_lsci_image,
                "search_window_size": self.shared_config["search_window_size"],
                "filter_strength": self.params["filter_strength"],
                "last_similarity_map": last_similarity_map,
                "filter_data": {
                    "NL Means": nlm_image,
                    "Normalization Factors": normalization_factors,
                    "Last Similarity Map": last_similarity_map,
                    "NL Standard Deviation": nl_std_image,
                    "NL LSCI": nl_lsci_image,
                },
            }
        else:  # LSCI
            mean_filter, std_dev_filter, lsci_filter = self.result_images
            return {
                **base_result,
                "mean_filter": mean_filter,
                "std_dev_filter": std_dev_filter,
                "lsci_filter": lsci_filter,
                "filter_data": {
                    "Mean Filter": mean_filter,
                    "Std Dev Filter": std_dev_filter,
                    "LSCI": lsci_filter,
                },
            }


## Image Rendering ##


def create_technique_config(technique, tab):
    """Create a configuration dictionary for the specified denoising technique."""
    result_image = session_state.get_technique_result(technique)
    if result_image is None:
        st.error(f"No results available for {technique}.")
        return None

    params = session_state.get_technique_params(technique)
    shared_config = create_shared_config(technique, params)

    selected_filters = session_state.get_session_state(
        f"{technique}_filters", session_state.get_filter_selection(technique)
    )

    config = {
        **shared_config,
        "results": result_image,
        "ui_placeholders": create_ui_placeholders(tab, selected_filters),
        "selected_filters": selected_filters,
        "kernel": {
            "size": shared_config["kernel_size"],
            "outline_color": "red",
            "outline_width": 1,
            "grid_line_color": "red",
            "grid_line_style": ":",
            "grid_line_width": 1,
            "center_pixel_color": "green",
            "center_pixel_opacity": 0.5,
        },
        "search_window": {
            "outline_color": "blue",
            "outline_width": 2,
            "size": shared_config["search_window_size"],
        },
        "pixel_value": {
            "text_color": "white",
            "font_size": 8,
        },
        "zoom": False,
        "processable_area": shared_config["processable_area"],
        "last_processed_pixel": result_image.get("last_processed_pixel", (0, 0)),
    }
    return config


def display_filters(config):
    """Prepare filters data based on the provided configuration."""
    filter_options = {
        **config["results"].get("filter_data", {}),
        "Original Image": session_state.get_image_array(),
    }

    display_data = []
    for filter_name in config["selected_filters"]:
        if filter_name in filter_options:
            filter_data = prepare_filter_data(filter_options[filter_name])
            plot_config = create_plot_config(config, filter_name, filter_data)
            display_data.append(
                (
                    plot_config,
                    config["ui_placeholders"][filter_name.lower().replace(" ", "_")],
                    False,
                )
            )

            if config["show_per_pixel_processing"]:
                display_data.append(
                    (
                        plot_config,
                        config["ui_placeholders"],
                        True,
                    )
                )
        else:
            st.warning(f"Data for {filter_name} is not available.")

    return display_data


def prepare_filter_data(filter_data):
    """Ensure filter data is 2D."""
    return (
        filter_data.reshape(session_state.get_image_array().shape)
        if filter_data.ndim == 1
        else filter_data
    )


def create_plot_config(config, filter_name, filter_data):
    """Create a plot configuration dictionary."""
    return {
        **config,
        "filter_data": filter_data,
        "vmin": np.min(filter_data),
        "vmax": np.max(filter_data),
        "title": filter_name,
    }


def create_ui_placeholders(tab, selected_filters):
    """Create UI placeholders for filters and formula."""
    with tab:
        placeholders = {"formula": st.empty()}
        columns = st.columns(max(1, len(selected_filters)))

        for i, filter_name in enumerate(selected_filters):
            filter_key = filter_name.lower().replace(" ", "_")
            placeholders[filter_key] = columns[i].empty()
            if session_state.get_session_state("show_per_pixel", False):
                placeholders[f"zoomed_{filter_key}"] = (
                    columns[i]
                    .expander(f"Zoomed-in {filter_name}", expanded=False)
                    .empty()
                )

    return placeholders


def get_zoomed_image_section(image, center_x_coord, center_y_coord, zoom_size):
    """Get a zoomed-in section of the image centered at the specified pixel."""
    half_zoom = zoom_size // 2
    top, bottom = (
        max(0, center_y_coord - half_zoom),
        min(image.shape[0], center_y_coord + half_zoom + 1),
    )
    left, right = (
        max(0, center_x_coord - half_zoom),
        min(image.shape[1], center_x_coord + half_zoom + 1),
    )

    return image[top:bottom, left:right], (center_x_coord - left, center_y_coord - top)


def apply_processing(image, technique, params):
    """Apply the specified processing technique to the image."""
    pixels_to_process = session_state.get_session_state("pixels_to_process", image.size)
    params["pixels_to_process"] = pixels_to_process

    processor = ImageProcessor(image, technique, params)
    result = processor.run_parallel_processing()

    return result