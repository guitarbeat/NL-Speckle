"""
This module defines the SidebarUI class and related constants for the Streamlit
application. It includes available color maps and preloaded image paths.
"""

import numpy as np
import streamlit as st
from PIL import Image
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from src.config import AVAILABLE_COLOR_MAPS, PRELOADED_IMAGE_PATHS

# Create a Generator object with a seed
rng = np.random.default_rng(seed=42)

@dataclass
class SidebarUI:
    @staticmethod
    def setup() -> Optional[Dict[str, Any]]:
        # Apply custom CSS for better spacing and wider sidebar
        SidebarUI._apply_custom_css()

        # Main setup logic
        image, color_map = SidebarUI._image_selection()
        if image is None:
            return None

        display_options = SidebarUI._display_options(image)
        nlm_params = SidebarUI._nlm_options(image)
        advanced_options = SidebarUI._advanced_options(image)

        # Update session state
        SidebarUI._update_session_state(nlm_params, advanced_options, display_options)

        return SidebarUI._create_return_dict(image, color_map, display_options, nlm_params, advanced_options)

    @staticmethod
    def _apply_custom_css():
        st.sidebar.markdown("""
            <style>
                [data-testid="stSidebar"] {
                    min-width: 300px;
                }
            </style>
        """, unsafe_allow_html=True)

    @staticmethod
    def _update_session_state(nlm_params, advanced_options, display_options):
        st.session_state.update({
            "use_full_image": nlm_params["use_whole_image"],
            "use_sat": advanced_options["use_sat"],
            "pixels_to_process": display_options["pixels_to_process"]
        })

    @staticmethod
    def _create_return_dict(image, color_map, display_options, nlm_params, advanced_options):
        return {
            "image": image,
            "image_array": np.array(image),
            "color_map": color_map,
            **display_options,
            **nlm_params,
            **advanced_options,
        }

    # Image selection methods
    @staticmethod
    def _image_selection() -> Tuple[Optional[Image.Image], str]:
        with st.sidebar.expander("🖼️ Image Selector", expanded=True):
            image_source = st.radio("Select Image Source", ("Preloaded Images", "Upload Image"))
            
            if image_source == "Preloaded Images":
                return SidebarUI._load_preloaded_image()
            else:
                return SidebarUI._load_uploaded_image()

    @staticmethod
    def _load_preloaded_image() -> Tuple[Optional[Image.Image], str]:
        selected_image = st.selectbox("Select Image", list(PRELOADED_IMAGE_PATHS.keys()))
        try:
            image = Image.open(PRELOADED_IMAGE_PATHS[selected_image]).convert("L")
            st.session_state.image_file = selected_image
            return SidebarUI._process_loaded_image(image)
        except (FileNotFoundError, IOError) as e:
            st.error(f"Error loading image: {e}. Please try again.")
            return None, ""

    @staticmethod
    def _load_uploaded_image() -> Tuple[Optional[Image.Image], str]:
        uploaded_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg", "tif", "tiff"])
        if uploaded_file is None:
            st.warning("Please upload an image.")
            return None, ""
        try:
            image = Image.open(uploaded_file).convert("L")
            st.session_state.image_file = uploaded_file.name
            return SidebarUI._process_loaded_image(image)
        except IOError as e:
            st.error(f"Error loading image: {e}. Please try again.")
            return None, ""

    @staticmethod
    def _process_loaded_image(image: Image.Image) -> Tuple[Image.Image, str]:
        st.image(image, caption="Input Image", use_column_width=True)
        color_map = SidebarUI._select_color_map()
        return image, color_map

    # Display options methods
    @staticmethod
    def _display_options(image: Image.Image) -> Dict[str, Any]:
        with st.sidebar.expander("🔧 Display Options", expanded=True):
            kernel_size = st.session_state.get("kernel_size", 3)
            kernel_size = st.slider(
                "Kernel Size",
                min_value=3,
                max_value=21,
                value=kernel_size,
                step=2,
                key="kernel_size_slider",
            )
            st.session_state.kernel_size = kernel_size

            show_per_pixel = st.toggle(
                "Show Per-Pixel Processing Steps", value=False, key="show_per_pixel"
            )
            total_pixels = (image.width - kernel_size + 1) * (
                image.height - kernel_size + 1
            )
            pixels_to_process = (
                SidebarUI._setup_pixel_processing(total_pixels)
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
    def _setup_pixel_processing(total_pixels: int) -> int:
        if "percentage_slider" not in st.session_state:
            st.session_state.percentage_slider = 100
        if "exact_pixel_count" not in st.session_state:
            st.session_state.exact_pixel_count = total_pixels

        def update_exact_count():
            st.session_state.exact_pixel_count = int(
                total_pixels * st.session_state.percentage_slider / 100
            )

        def update_percentage():
            st.session_state.percentage_slider = int(
                (st.session_state.exact_pixel_count / total_pixels) * 100
            )
        col1, col2 = st.columns(2)
        with col1:
            st.slider(
                "Percentage",
                min_value=1,
                max_value=100,
                value=st.session_state.percentage_slider,
                key="percentage_slider",
                on_change=update_exact_count,
            )
        with col2:
            st.number_input(
                "Exact Pixels",
                min_value=1,
                max_value=total_pixels,
                value=st.session_state.exact_pixel_count,
                key="exact_pixel_count",
                on_change=update_percentage,
            )
        
        # Use the value from session state, which will be updated by the callbacks
        return st.session_state.exact_pixel_count

    @staticmethod
    def _select_color_map() -> str:
        if "color_map" not in st.session_state:
            st.session_state.color_map = "gray"

        current_color_map = st.session_state.get("color_map")
        selected_color_map = st.selectbox(
            "Select Color Map",
            AVAILABLE_COLOR_MAPS,
            index=AVAILABLE_COLOR_MAPS.index(current_color_map),
            key="color_map_select",
        )

        if selected_color_map != current_color_map:
            st.session_state.color_map = selected_color_map

        return selected_color_map

    # Advanced options methods
    @staticmethod
    def _advanced_options(image: Image.Image) -> Dict[str, Any]:
        with st.sidebar.expander("🔬 Advanced Options", expanded=False):
            normalization_option = st.selectbox(
                "Normalization",
                options=["None", "Percentile"],
                index=0,
                help="Choose the normalization method for the image",
            )

            apply_gaussian_noise = st.checkbox(
                "Add Gaussian Noise", value=False, help="Add Gaussian noise to the image"
            )
            noise_params = (
                SidebarUI._setup_gaussian_noise_params() if apply_gaussian_noise else {}
            )

            use_sat = st.toggle(
                "Use Summed-Area Tables (SAT)",
                value=False,
                help="Enable Summed-Area Tables for faster Speckle Contrast calculation"
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
            "use_sat": use_sat,
        }

    @staticmethod
    def _setup_gaussian_noise_params() -> Dict[str, float]:
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
        noise = rng.random.normal(mean, std_dev, image_np.shape)
        return np.clip(image_np + noise, 0, 1)

    @staticmethod
    def _normalize_percentile(image_np: np.ndarray) -> np.ndarray:
        p_low, p_high = np.percentile(image_np, [2, 98])
        image_np = np.clip(image_np, p_low, p_high)
        image_np = (image_np - p_low) / (p_high - p_low)
        return image_np

    # NLM options method
    @staticmethod
    def _nlm_options(image: Image.Image) -> Dict[str, Any]:
        with st.sidebar.expander("🔍 NLM Options", expanded=False):
            try:
                image_shape = image.size
                max_search_window = min(101, *image_shape)
                default_search_window = min(21, max_search_window)
                use_whole_image = st.checkbox(
                    "Use whole image as search window", value=False
                )

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
                    max_value=100.0,
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
