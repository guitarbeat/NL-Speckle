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


def apply_processing_to_image(image, technique, params, pixel_percentage):
    height, width = image.shape
    result_images = initialize_result_images(technique, image)
    pixels = get_pixels_for_processing(height, width, pixel_percentage)

    args_list = create_args_list(technique, pixels, image, params, height, width)

    progress_bar, status = st.progress(0), st.empty()

    run_parallel_processing(technique, args_list, progress_bar, status, result_images, pixels)

    session_state.set_last_processed_pixel(pixels[-1][1], pixels[-1][0]) if pixels else None
    progress_bar.progress(1.0)
    status.text("Processing complete!")

    processing_end = (pixels[-1][0], pixels[-1][1]) if pixels else (0, 0)
    return format_result(technique, tuple(result_images), (processing_end, params["kernel_size"], len(pixels), (height, width)), params)


def initialize_result_images(technique, image):
    """Initialize result images based on the technique."""
    return [np.zeros_like(image) for _ in range(5 if technique == "nlm" else 3)]


def get_pixels_for_processing(height, width, pixel_percentage):
    """Generate a list of pixels to be processed based on the desired percentage."""
    pixels = [(y, x) for y in range(height) for x in range(width)]
    return pixels[:int(len(pixels) * pixel_percentage / 100)]


def run_parallel_processing(technique, args_list, progress_bar, status, result_images, pixels):
    """Run the image processing function in parallel."""
    num_processes = min(cpu_count(), st.session_state.get("max_workers", 4))
    start_time = time.time()
    with st.spinner(f"Processing {technique.upper()}..."), ProcessPoolExecutor(max_workers=num_processes) as executor:
        for i, result in enumerate(executor.map(process_pixel_wrapper, args_list)):
            if result is not None:
                y, x, *values = result
                for j, value in enumerate(values):
                    result_images[j][y, x] = value
            update_progress(i + 1, len(pixels), start_time, progress_bar, status)

def create_args_list(technique, pixels, image, params, height, width):
    if technique == "nlm":
        return [
            (
                process_nlm_pixel,
                y,
                x,
                image,
                params["kernel_size"],
                params["search_window_size"],
                params["filter_strength"],
                params.get("use_full_image", False),
                height,
                width,
            )
            for y, x in pixels
        ]
    else:  # speckle
        return [
            (
                process_speckle_pixel,
                y,
                x,
                image,
                params["kernel_size"],
                (params["kernel_size"] // 2, params["kernel_size"] // 2),
                height,
                width,
                width,
            )
            for y, x in pixels
        ]


def format_result(technique, result_images, common_results, params):
    processing_end, kernel_size, pixel_count, image_dimensions = common_results
    base_result = {
        "processing_end_coord": processing_end,
        "kernel_size": kernel_size,
        "pixels_processed": pixel_count,
        "image_dimensions": image_dimensions,
    }

    if technique == "nlm":
        (
            nlm_image,
            normalization_factors,
            nl_std_image,
            nl_speckle_image,
            last_similarity_map,
        ) = result_images
        return {
            **base_result,
            "nonlocal_means": nlm_image,
            "normalization_factors": normalization_factors,
            "nonlocal_std": nl_std_image,
            "nonlocal_speckle": nl_speckle_image,
            "search_window_size": params["search_window_size"],
            "filter_strength": params["filter_strength"],
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
        mean_filter, std_dev_filter, speckle_contrast_filter = result_images
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


def process_speckle_pixel(y, x, image, kernel_size, center, height, width, max_width):
    """Process a single pixel using speckle filtering."""
    window = extract_window(y, x, kernel_size, image, height, width)
    mean, std = compute_statistics(window)
    std_ratio = std / mean if mean != 0 else 0
    return y, x, mean, std, std_ratio


def process_nlm_pixel(
    y,
    x,
    image,
    kernel_size,
    search_window_size,
    filter_strength,
    use_full_image,
    height,
    width,
):
    """Process a single pixel using Non-Local Means (NLM) filtering."""
    y, x = (
        max(kernel_size // 2, min(y, height - kernel_size // 2 - 1)),
        max(kernel_size // 2, min(x, width - kernel_size // 2 - 1)),
    )

    search_radius = max(height, width) if use_full_image else search_window_size // 2
    y_start, y_end = max(0, y - search_radius), min(height, y + search_radius + 1)
    x_start, x_end = max(0, x - search_radius), min(width, x + search_radius + 1)

    patch_xy = image[
        y - kernel_size // 2 : y + kernel_size // 2 + 1,
        x - kernel_size // 2 : x + kernel_size // 2 + 1,
    ]

    weights = []
    for i in range(y_start, y_end):
        for j in range(x_start, x_end):
            if (i, j) == (y, x):
                continue
            patch_ij = extract_window(i, j, kernel_size, image, height, width)
            if patch_ij.size == 0:
                continue
            weight = calculate_weights(patch_xy, patch_ij, filter_strength)
            weights.append(weight)

    if not weights:
        return y, x, image[y, x], 0, 0, 0, 0

    weights_sum = sum(weights)
    pixel_sum = sum(
        w * image[i, j]
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


def process_pixel_wrapper(args):
    try:
        return args[0](*args[1:])
    except Exception as e:
        print(f"Error processing pixel: {str(e)}")
        return None


def update_progress(current, total, start_time, progress_bar, status):
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

    selected_filters = st.session_state.get(
        f"{technique}_filters", session_state.get_filter_selection(technique)
    )
    return {
        "technique": technique,
        "results": result_image,
        "ui_placeholders": create_ui_placeholders(tab, selected_filters),
        "last_processed_pixel": session_state.get_last_processed_pixel(),
        "selected_filters": selected_filters,
        "kernel": {"size": session_state.kernel_size()},
        "show_kernel": session_state.get_show_per_pixel_processing(),
        "zoom": False,
        "show_per_pixel_processing": session_state.get_show_per_pixel_processing(),
    }


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
            display_data.append((plot_config, config["ui_placeholders"][filter_name.lower().replace(" ", "_")], False))
            
            if config["show_per_pixel_processing"]:
                display_data.append((plot_config, config["ui_placeholders"], True))
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
