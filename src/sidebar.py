"""
This module defines the SidebarUI class and related constants for the Streamlit
application. It includes available color maps and preloaded image paths.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import streamlit as st

# import streamlit_image_coordinates
from PIL import Image

# Create a Generator object with a seed
rng = np.random.default_rng(seed=42)

AVAILABLE_COLOR_MAPS = [
    "gray",
    "plasma",
    "inferno",
    "magma",
    "pink",
    "hot",
    "cool",
    "YlOrRd",
]
PRELOADED_IMAGE_PATHS = {
    "image50.png": "media/image50.png",
    "spatial.tif": "media/spatial.tif",
    "logo.jpg": "media/logo.jpg",
}


@dataclass
class SidebarUI:
    @staticmethod
    def setup() -> Optional[Dict[str, Any]]:
        st.sidebar.title("Image Processing Settings")

        with st.sidebar.expander("Image Selector", expanded=True):
            image, color_map = SidebarUI.select_image_source()
            if image is None:
                return None

        display_options = SidebarUI.setup_display_options(image)

        with st.sidebar.expander("NLM Parameters", expanded=True):
            nlm_params = SidebarUI._setup_nlm_options(image)

        # Store the use_full_image flag in session state for future use
        st.session_state["use_full_image"] = nlm_params.get("use_whole_image")
        with st.sidebar.expander("Advanced Options", expanded=True):
            advanced_options = SidebarUI.setup_advanced_options(image)

        return {
            "image": image,
            "image_array": np.array(image),
            "color_map": color_map,
            **display_options,
            **nlm_params,
            **advanced_options,
            "use_full_image": nlm_params.get("use_whole_image", False),
        }

    @staticmethod
    def select_image_source() -> Optional[Image.Image]:
        """
        Handle image source selection and return loaded image.
        """
        image_source_type = st.sidebar.radio(
            "Select Image Source", ("Preloaded Images", "Upload Image")
        )
        try:
            if image_source_type == "Preloaded Images":
                selected_image_name = st.sidebar.selectbox(
                    "Select Image", list(PRELOADED_IMAGE_PATHS.keys())
                )
                loaded_image = Image.open(
                    PRELOADED_IMAGE_PATHS[selected_image_name]
                ).convert("L")
            else:
                uploaded_file = st.sidebar.file_uploader(
                    "Upload Image", type=["png", "jpg", "jpeg", "tif", "tiff"]
                )
                if uploaded_file is None:
                    st.sidebar.warning("Please upload an image.")
                    return None
                loaded_image = Image.open(uploaded_file).convert("L")

            st.sidebar.image(loaded_image, caption="Input Image", use_column_width=True)
            color_map = SidebarUI.select_color_map()

            return loaded_image, color_map
        except (FileNotFoundError, IOError) as e:
            st.sidebar.error(f"Error loading image: {e}. Please try again.")
            return None

    @staticmethod
    def setup_display_options(image: Image.Image) -> Dict[str, Any]:
        kernel_size = st.session_state.get("kernel_size", 3)
        kernel_size = st.sidebar.slider(
            "Kernel Size",
            min_value=3,
            max_value=21,
            value=kernel_size,
            step=2,
            key="kernel_size_slider",
        )
        st.session_state.kernel_size = kernel_size

        show_per_pixel = st.sidebar.checkbox(
            "Show Per-Pixel Processing Steps", value=True, key="show_per_pixel"
        )
        total_pixels = (image.width - kernel_size + 1) * (
            image.height - kernel_size + 1
        )
        pixels_to_process = (
            SidebarUI.setup_pixel_processing(total_pixels)
            if show_per_pixel
            else total_pixels
        )

        return {
            "show_per_pixel_processing": show_per_pixel,
            "total_pixels": total_pixels,
            "pixels_to_process": pixels_to_process,
            "kernel_size": kernel_size,
        }

    @staticmethod
    def setup_pixel_processing(total_pixels: int) -> int:
        if "exact_pixel_count" not in st.session_state:
            st.session_state.exact_pixel_count = total_pixels
        if "percentage_slider" not in st.session_state:
            st.session_state.percentage_slider = 100

        def update_exact_count():
            st.session_state.exact_pixel_count = int(
                total_pixels * st.session_state.percentage_slider / 100
            )

        def update_percentage():
            st.session_state.percentage_slider = int(
                (st.session_state.exact_pixel_count / total_pixels) * 100
            )

        col1, col2 = st.sidebar.columns(2)
        with col1:
            st.slider(
                "Percentage",
                min_value=1,
                max_value=100,
                value=st.session_state.percentage_slider,
                step=1,
                key="percentage_slider",
                on_change=update_exact_count,
            )
        with col2:
            st.number_input(
                "Exact Pixels",
                min_value=1,
                max_value=total_pixels,
                value=min(st.session_state.exact_pixel_count, total_pixels),
                step=1,
                key="exact_pixel_count",
                on_change=update_percentage,
            )
        return st.session_state.exact_pixel_count

    @staticmethod
    def select_color_map() -> str:
        """
        Select color map for image display and update session state.
        """
        # Initialize color_map in session state if it doesn't exist
        if "color_map" not in st.session_state:
            st.session_state.color_map = "gray"  # Default color map

        # Get the current color map from session state or use default
        current_color_map = st.session_state.get("color_map")

        # Create the selectbox for color map selection
        selected_color_map = st.sidebar.selectbox(
            "Select Color Map",
            AVAILABLE_COLOR_MAPS,
            index=AVAILABLE_COLOR_MAPS.index(current_color_map),
            key="color_map_select",
        )

        # Update session state if the color map has changed
        if selected_color_map != current_color_map:
            st.session_state.color_map = selected_color_map
            # st.rerun()

        return selected_color_map

    @staticmethod
    def setup_advanced_options(image: Image.Image) -> Dict[str, Any]:
        """
        Set up advanced options and return selected values.
        """
        normalization_option = st.sidebar.selectbox(
            "Normalization",
            options=["None", "Percentile"],
            index=0,
            help="Choose the normalization method for the image",
        )

        apply_gaussian_noise = st.sidebar.checkbox(
            "Add Gaussian Noise", value=False, help="Add Gaussian noise to the image"
        )
        noise_params = (
            SidebarUI._setup_gaussian_noise_params() if apply_gaussian_noise else {}
        )

        image_np = np.array(image) / 255.0
        if apply_gaussian_noise:
            image_np = SidebarUI._apply_gaussian_noise(image_np, **noise_params)
        if normalization_option == "Percentile":
            image_np = SidebarUI._normalize_percentile(image_np)
        return {
            "image_np": image_np,
            "add_noise": apply_gaussian_noise,
            "normalization_option": normalization_option,
        }

    @staticmethod
    def _setup_gaussian_noise_params() -> Dict[str, float]:
        """
        Set up Gaussian noise parameter inputs.
        """
        mean = st.sidebar.number_input(
            "Noise Mean",
            min_value=0.0,
            max_value=1.0,
            value=0.1,
            step=0.01,
            format="%.2f",
        )
        std_dev = st.sidebar.number_input(
            "Noise Standard Deviation",
            min_value=0.0,
            max_value=1.0,
            value=0.1,
            step=0.01,
            format="%.2f",
        )
        return {"mean": mean, "std_dev": std_dev}

    @staticmethod
    def _apply_gaussian_noise(
        image_np: np.ndarray, mean: float, std_dev: float
    ) -> np.ndarray:
        """
        Apply Gaussian noise to the image.

        Args:
            image_np (np.ndarray): Input image as a numpy array. mean (float):
            Mean of the Gaussian noise. std_dev (float): Standard deviation of
            the Gaussian noise.

        Returns:
            np.ndarray: Image with applied Gaussian noise.
        """
        noise = rng.random.normal(mean, std_dev, image_np.shape)
        return np.clip(image_np + noise, 0, 1)

    @staticmethod
    def _normalize_percentile(image_np: np.ndarray) -> np.ndarray:
        """
        Normalize image using percentile scaling
        """
        p_low, p_high = np.percentile(image_np, [2, 98])
        image_np = np.clip(image_np, p_low, p_high)
        image_np = (image_np - p_low) / (p_high - p_low)
        return image_np

    @staticmethod
    def _setup_nlm_options(image: Image.Image) -> Dict[str, Any]:
        """
        Set up non-local means denoising options, including search window size
        selection.

        Args:
            image (Image.Image): The input image.

        Returns:
            Dict[str, Any]: Dictionary of NLM options.
        """
        try:
            image_shape = image.size
            max_search_window = min(101, *image_shape)
            default_search_window = min(21, max_search_window)
            use_whole_image = st.checkbox(
                "Use whole image as search window", value=False
            )

            st.session_state["use_full_image"] = use_whole_image

            # Search window size selection
            if not use_whole_image:
                search_window_size = st.slider(
                    "Search Window Size",
                    min_value=3,
                    max_value=max_search_window,
                    value=default_search_window,
                    step=2,
                    help="Size of the search window for NLM (must be odd)",
                )
                search_window_size = (
                    search_window_size
                    if search_window_size % 2 == 1
                    else search_window_size + 1
                )
            else:
                search_window_size = max(image_shape)

            filter_strength = st.slider(
                "Filter Strength (h)",
                min_value=0.1,
                max_value=20.0,
                value=10.0,
                step=0.1,
                format="%.1f",
                help="Filter strength for NLM (higher values mean more smoothing)",
            )
            return {
                "search_window_size": search_window_size,
                "filter_strength": filter_strength,
                "use_whole_image": use_whole_image,
            }
        except (ValueError, TypeError, KeyError) as e:
            st.error(f"Error setting up NLM options: {e}")
            return {
                "search_window_size": 21,
                "filter_strength": 10.0,
                "use_whole_image": False,
            }
