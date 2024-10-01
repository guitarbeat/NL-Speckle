"""
This module handles image visualization for the Streamlit application.
"""

import numpy as np
import streamlit as st
import src.session_state as session_state
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count
import time

## Image Processing ##

def extract_window(y, x, kernel_size, image, height, width):
    """Extract a window from the image centered at (y, x) with the given kernel size."""
    half_kernel = kernel_size // 2
    top = max(0, y - half_kernel)
    bottom = min(height, y + half_kernel + 1)
    left = max(0, x - half_kernel)
    right = min(width, x + half_kernel + 1)
    window = image[top:bottom, left:right]
    return window


def compute_statistics(window):
    """Compute mean and standard deviation of the given window."""
    mean = np.nanmean(window)
    std = np.nanstd(window)
    return mean, std


def calculate_weights(patch_xy, patch_ij, filter_strength):
    """Calculate the weight between two patches."""
    squared_diff = (
        patch_xy[
            : min(patch_xy.shape[0], patch_ij.shape[0]),
            : min(patch_xy.shape[1], patch_ij.shape[1]),
        ]
        - patch_ij[
            : min(patch_xy.shape[0], patch_ij.shape[0]),
            : min(patch_xy.shape[1], patch_ij.shape[1]),
        ]
    ) ** 2
    weight = np.exp(-np.sum(squared_diff) / (filter_strength**2))
    return weight


def create_shared_config(technique, params):
    """Create a shared configuration for both processing and overlays."""
    nlm_options = session_state.get_nlm_options()
    return {
        "technique": technique,
        "kernel_size": params["kernel_size"],
        "search_window_size": params.get("search_window_size", nlm_options["search_window_size"]),
        "use_full_image": params.get("use_full_image", nlm_options["use_whole_image"]),
        "last_processed_pixel": session_state.get_last_processed_pixel(),
        "show_per_pixel_processing": session_state.get_show_per_pixel_processing(),
    }


class ImageProcessor:
    """A class to handle different image processing techniques."""

    def __init__(self, image, technique, params, pixel_percentage):
        self.image = image
        self.technique = technique
        self.params = params
        self.pixel_percentage = pixel_percentage
        self.height, self.width = image.shape
        self.result_images = self.initialize_result_images()
        self.pixels = self.get_pixels_for_processing()
        self.shared_config = create_shared_config(technique, params)

    def initialize_result_images(self):
        """Initialize result images based on the technique."""
        return [np.zeros_like(self.image) for _ in range(5 if self.technique == "nlm" else 3)]

    def get_pixels_for_processing(self):
        """Generate a list of pixels to be processed based on the desired percentage."""
        pixels = [(y, x) for y in range(self.height) for x in range(self.width)]
        return pixels[:int(len(pixels) * self.pixel_percentage / 100)]

    def create_args_list(self):
        """Create a list of arguments for processing pixels."""
        return [
            (
                self.process_pixel,
                y,
                x,
            )
            for y, x in self.pixels
        ]

    def process_pixel(self, y, x):
        """Process a single pixel based on the technique."""
        if self.technique == "nlm":
            return self.process_nlm_pixel(y, x)
        elif self.technique == "speckle":
            return self.process_speckle_pixel(y, x)
        else:
            raise ValueError(f"Unknown technique: {self.technique}")

    def process_speckle_pixel(self, y, x):
        """Process a single pixel using speckle filtering."""
        window = extract_window(y, x, self.shared_config["kernel_size"], self.image, self.height, self.width)
        mean, std = compute_statistics(window)
        std_ratio = std / mean if mean != 0 else 0
        return y, x, mean, std, std_ratio

    def process_nlm_pixel(self, y, x):
        """Process a single pixel using Non-Local Means (NLM) filtering."""
        kernel_size = self.shared_config["kernel_size"]
        search_window_size = self.shared_config["search_window_size"]
        filter_strength = self.params["filter_strength"]
        use_full_image = self.shared_config["use_full_image"]

        y, x = (
            max(kernel_size // 2, min(y, self.height - kernel_size // 2 - 1)),
            max(kernel_size // 2, min(x, self.width - kernel_size // 2 - 1)),
        )

        if use_full_image:
            search_radius = max(self.height, self.width)
        else:
            search_radius = search_window_size // 2

        y_start, y_end = max(0, y - search_radius), min(self.height, y + search_radius + 1)
        x_start, x_end = max(0, x - search_radius), min(self.width, x + search_radius + 1)

        patch_xy = self.image[
            y - kernel_size // 2 : y + kernel_size // 2 + 1,
            x - kernel_size // 2 : x + kernel_size // 2 + 1,
        ]

        weights = []
        for i in range(y_start, y_end):
            for j in range(x_start, x_end):
                if (i, j) == (y, x):
                    continue
                patch_ij = extract_window(i, j, kernel_size, self.image, self.height, self.width)
                if patch_ij.size == 0:
                    continue
                weight = calculate_weights(patch_xy, patch_ij, filter_strength)
                weights.append(weight)

        if not weights:
            return y, x, self.image[y, x], 0, 0, 0, 0

        weights_sum = sum(weights)
        pixel_sum = sum(
            w * self.image[i, j]
            for w, (i, j) in zip(
                weights,
                (
                    (i, j)
                    for i in range(y_start, y_end)
                    for j in range(x_start, x_end)
                    if (i, j) != (y, x)
                ),
            )
        )

        nlm_value = pixel_sum / weights_sum
        weight_avg = weights_sum / ((y_end - y_start) * (x_end - x_start) - 1)
        weight_std = (
            sum(w**2 for w in weights) / ((y_end - y_start) * (x_end - x_start) - 1)
            - weight_avg**2
        ) ** 0.5

        return y, x, nlm_value, weight_avg, weight_std, max(weights), min(weights)

    def run_parallel_processing(self):
        """Run the image processing function in parallel."""
        args_list = self.create_args_list()
        progress_bar, status = st.progress(0), st.empty()
        num_processes = min(cpu_count(), st.session_state.get("max_workers", 4))
        start_time = time.time()

        with st.spinner(f"Processing {self.technique.upper()}..."), ProcessPoolExecutor(max_workers=num_processes) as executor:
            for i, result in enumerate(executor.map(process_pixel_wrapper, args_list)):
                if result is not None:
                    y, x, *values = result
                    for j, value in enumerate(values):
                        self.result_images[j][y, x] = value
                # Call the module-level update_progress function directly
                update_progress(i + 1, len(self.pixels), start_time, progress_bar, status)

        # Set the last processed pixel
        if self.pixels:
            last_x, last_y = self.pixels[-1]
            session_state.set_last_processed_pixel(last_x, last_y)
        
        progress_bar.progress(1.0)
        status.text("Processing complete!")

        processing_end = (self.pixels[-1][0], self.pixels[-1][1]) if self.pixels else (0, 0)
        return self.format_result(processing_end)

    def format_result(self, processing_end):
        """Format the processing results."""
        base_result = {
            "processing_end_coord": processing_end,
            "kernel_size": self.shared_config["kernel_size"],
            "pixels_processed": len(self.pixels),
            "image_dimensions": (self.height, self.width),
        }

        if self.technique == "nlm":
            (
                nlm_image,
                normalization_factors,
                nl_std_image,
                nl_speckle_image,
                last_similarity_map,
            ) = self.result_images
            return {
                **base_result,
                "nonlocal_means": nlm_image,
                "normalization_factors": normalization_factors,
                "nonlocal_std": nl_std_image,
                "nonlocal_speckle": nl_speckle_image,
                "search_window_size": self.shared_config["search_window_size"],
                "filter_strength": self.params["filter_strength"],
                "last_similarity_map": last_similarity_map,
                "filter_data": {
                    "Non-Local Means": nlm_image,
                    "Normalization Factors": normalization_factors,
                    "Last Similarity Map": last_similarity_map,
                    "Non-Local Standard Deviation": nl_std_image,
                    "Non-Local Speckle": nl_speckle_image,
                },
            }
        else:  # speckle
            mean_filter, std_dev_filter, speckle_contrast_filter = self.result_images
            return {
                **base_result,
                "mean_filter": mean_filter,
                "std_dev_filter": std_dev_filter,
                "speckle_contrast_filter": speckle_contrast_filter,
                "filter_data": {
                    "Mean Filter": mean_filter,
                    "Std Dev Filter": std_dev_filter,
                    "Speckle Contrast": speckle_contrast_filter,
                },
            }

def process_pixel_wrapper(args):
    """Wrapper to process a pixel with error handling."""
    try:
        return args[0](*args[1:])
    except Exception as e:
        print(f"Error processing pixel: {str(e)}")
        return None


def update_progress(current, total, start_time, progress_bar, status):
    """Update the progress bar and status text."""
    progress = current / total
    progress_bar.progress(progress)
    elapsed_time = time.time() - start_time
    estimated_total_time = elapsed_time / progress if progress > 0 else 0
    remaining_time = estimated_total_time - elapsed_time
    status.text(
        f"Processed {current}/{total} pixels. Estimated time remaining: {remaining_time:.2f} seconds"
    )

## Image Rendering ##


def create_technique_config(technique, tab):
    """Create a configuration dictionary for the specified denoising technique."""
    result_image = session_state.get_technique_result(technique)
    if result_image is None:
        st.error(f"No results available for {technique}.")
        return None

    params = session_state.get_technique_params(technique)
    shared_config = create_shared_config(technique, params)
    
    selected_filters = st.session_state.get(
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
            plot_config = create_plot_config(
                config, filter_name, prepare_filter_data(filter_options[filter_name])
            )
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
            if session_state.get_show_per_pixel_processing():
                placeholders[f"zoomed_{filter_key}"] = (
                    columns[i]
                    .expander(f"Zoomed-in {filter_name}", expanded=False)
                    .empty()
                )

    return placeholders


def get_zoomed_image_section(image, center_x, center_y, zoom_size):
    """Get a zoomed-in section of the image centered at the specified pixel."""
    half_zoom = zoom_size // 2
    top, bottom = (
        max(0, center_y - half_zoom),
        min(image.shape[0], center_y + half_zoom + 1),
    )
    left, right = (
        max(0, center_x - half_zoom),
        min(image.shape[1], center_x + half_zoom + 1),
    )

    return image[top:bottom, left:right], (center_x - left, center_y - top)


def apply_processing(image, technique, params, pixel_percentage):
    """Apply the specified processing technique to the image."""
    processor = ImageProcessor(image, technique, params, pixel_percentage)
    return processor.run_parallel_processing()