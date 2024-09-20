"""
This module defines the SidebarUI class and related constants for the Streamlit application.
It includes available color maps and preloaded image paths.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
from PIL import Image
import numpy as np
import streamlit as st
from src.plotting import VisualizationConfig  # Import the VisualizationConfig class

AVAILABLE_COLOR_MAPS = [
    "gray",
    "viridis",
    "plasma",
    "inferno",
    "magma",
    "cividis",
    "pink",
]
PRELOADED_IMAGE_PATHS = {
    "image50.png": "media/image50.png",
    "spatial.tif": "media/spatial.tif",
    "logo.jpg": "media/logo.jpg",
}



@dataclass
class SidebarUI:
    """
    A class to handle the setup and management of the sidebar UI for image processing.

    This class provides static methods to create and manage various UI components
    in the Streamlit sidebar, including image selection, color map selection,
    display options, non-local means (NLM) denoising parameters, and advanced options.

    The main entry point is the `setup` method, which orchestrates the creation
    of all UI components and returns a dictionary of selected options and parameters.
    """

    @staticmethod
    def setup() -> Optional[Dict[str, Any]]:
        """
        Set up the sidebar UI and return selected options.

        Returns:
            Optional[Dict[str, Any]]: A dictionary containing all selected options and parameters,
            or None if there was an error in image loading.
        """
        st.sidebar.title("Image Processing Settings")

        with st.sidebar.expander("Image Selector", expanded=True):
            image = SidebarUI._select_image_source()
            if image is None:
                return None

            st.sidebar.markdown("### 🎨 Color Map")
            color_map = SidebarUI._select_color_map()

        display_options = SidebarUI._setup_display_options(image)
        with st.sidebar.expander("NLM Parameters", expanded=True):
            nlm_params = SidebarUI._setup_nlm_options(image)
        with st.sidebar.expander("Advanced Options", expanded=True):
            advanced_options = SidebarUI._setup_advanced_options(image)

        return {
            "image": image,
            "image_array": np.array(image),
            "cmap": color_map,
            **display_options,
            **nlm_params,
            **advanced_options,
            "use_full_image": nlm_params.get("use_whole_image", False),
        }

    @staticmethod
    def _select_image_source() -> Optional[Image.Image]:
        """
        Handle image source selection and return loaded image.

        Returns:
            Optional[Image.Image]: Loaded image or None if loading failed.
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
            return loaded_image
        except (FileNotFoundError, IOError) as e:
            st.sidebar.error(f"Error loading image: {e}. Please try again.")
            return None

    @staticmethod
    def _select_color_map() -> str:
        """
        Select color map for image display.

        Returns:
            str: Selected color map name.
        """
        config = VisualizationConfig()
        return st.sidebar.selectbox(
            "Select Color Map",
            AVAILABLE_COLOR_MAPS,
            index=AVAILABLE_COLOR_MAPS.index(
                st.session_state.get("color_map", config.color_map)
            ),
        )

    @staticmethod
    def _setup_display_options(image: Image.Image) -> Dict[str, Any]:
        """
        Set up display options and return selected values.

        Args:
            image (Image.Image): The input image.

        Returns:
            Dict[str, Any]: Dictionary of display options.
        """
        st.sidebar.markdown("### 🖥️ Display Options")
        show_per_pixel = st.sidebar.checkbox(
            "Show Per-Pixel Processing Steps", value=False
        )
        kernel_size = SidebarUI._select_kernel_size()
        total_pixels = (image.width - kernel_size + 1) * (image.height - kernel_size + 1)
        pixels_to_process = (
            SidebarUI._setup_pixel_count_sliders(total_pixels)
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
    def _select_kernel_size() -> int:
        """
        Select kernel size for processing.

        Returns:
            int: Selected kernel size.
        """
        if "kernel_size" not in st.session_state:
            st.session_state.kernel_size = 3
        return st.sidebar.slider(
            "Kernel Size",
            min_value=3,
            max_value=21,
            value=st.session_state.kernel_size,
            step=2,
        )

    @staticmethod
    def _setup_pixel_count_sliders(total_pixels: int) -> int:
        """
        Set up percentage and exact pixel count sliders.

        Args:
            total_pixels (int): Total number of pixels in the image.

        Returns:
            int: Number of pixels to process.
        """
        SidebarUI._init_pixel_count_state(total_pixels)
        col1, col2 = st.sidebar.columns(2)
        with col1:
            percentage = st.slider(
                "Percentage",
                min_value=1,
                max_value=100,
                value=st.session_state.percentage_slider,
                step=1,
            )
            st.session_state.exact_pixel_count = int(total_pixels * percentage / 100)
        with col2:
            exact_count = st.number_input(
                "Exact Pixels",
                min_value=0,
                max_value=total_pixels,
                value=st.session_state.exact_pixel_count,
                step=1,
            )
            st.session_state.percentage_slider = int((exact_count / total_pixels) * 100)
        return st.session_state.exact_pixel_count

    @staticmethod
    def _init_pixel_count_state(total_pixels: int) -> None:
        """
        Initialize pixel count session state if needed.

        Args:
            total_pixels (int): Total number of pixels in the image.
        """
        if "exact_pixel_count" not in st.session_state:
            st.session_state.exact_pixel_count = total_pixels
        if "percentage_slider" not in st.session_state:
            st.session_state.percentage_slider = 100

    @staticmethod
    def _setup_advanced_options(image: Image.Image) -> Dict[str, Any]:
        """
        Set up advanced options and return selected values.

        Args:
            image (Image.Image): The input image.

        Returns:
            Dict[str, Any]: Dictionary of advanced options.
        """
        normalization_option = SidebarUI._select_normalization()
        apply_gaussian_noise, noise_params = SidebarUI._setup_gaussian_noise()
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
    def _select_normalization() -> str:
        """
        Select normalization option.

        Returns:
            str: Selected normalization option.
        """
        return st.sidebar.selectbox(
            "Normalization",
            options=["None", "Percentile"],
            index=0,
            help="Choose the normalization method for the image",
        )

    @staticmethod
    def _setup_gaussian_noise() -> Tuple[bool, Dict[str, float]]:
        """
        Set up Gaussian noise parameters.

        Returns:
            Tuple[bool, Dict[str, float]]: Tuple containing a boolean
            and a dictionary of noise parameters.
        """
        add_noise = st.sidebar.checkbox(
            "Add Gaussian Noise", value=False, help="Add Gaussian noise to the image"
        )
        gaussian_noise_params = (
            SidebarUI._setup_gaussian_noise_params() if add_noise else {}
        )
        return add_noise, gaussian_noise_params

    @staticmethod
    def _setup_gaussian_noise_params() -> Dict[str, float]:
        """
        Set up Gaussian noise parameter inputs.

        Returns:
            Dict[str, float]: Dictionary of Gaussian noise parameters.
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
            image_np (np.ndarray): Input image as a numpy array.
            mean (float): Mean of the Gaussian noise.
            std_dev (float): Standard deviation of the Gaussian noise.

        Returns:
            np.ndarray: Image with applied Gaussian noise.
        """
        noise = np.random.normal(mean, std_dev, image_np.shape)
        return np.clip(image_np + noise, 0, 1)

    @staticmethod
    def _normalize_percentile(image_np: np.ndarray) -> np.ndarray:
        """
        Normalize image using percentile scaling.

        Args:
            image_np (np.ndarray): Input image as a numpy array.

        Returns:
            np.ndarray: Normalized image.
        """
        p_low, p_high = np.percentile(image_np, [2, 98])
        image_np = np.clip(image_np, p_low, p_high)
        image_np = (image_np - p_low) / (p_high - p_low)
        return image_np

    @staticmethod
    def _setup_nlm_options(image: Image.Image) -> Dict[str, Any]:
        """
        Set up non-local means denoising options.

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
            search_window_size = SidebarUI._select_search_window_size(
                max_search_window, default_search_window, use_whole_image, image_shape
            )
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

    @staticmethod
    def _select_search_window_size(
        max_size: int, default_size: int, use_whole: bool, image_shape: Tuple[int, int]
    ) -> int:
        """
        Select search window size for NLM.

        Args:
            max_size (int): Maximum allowed size for the search window.
            default_size (int): Default size for the search window.
            use_whole (bool): Whether to use the whole image as the search window.
            image_shape (Tuple[int, int]): Shape of the input image.

        Returns:
            int: Selected search window size.
        """
        if not use_whole:
            search_window_size = st.slider(
                "Search Window Size",
                min_value=3,
                max_value=max_size,
                value=default_size,
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
        return search_window_size
