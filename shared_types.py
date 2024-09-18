# Imports
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from PIL import Image
from streamlit_image_comparison import image_comparison

# Constants and Global Variables
AVAILABLE_COLOR_MAPS = ["gray", "viridis", "plasma", "inferno", "magma", "cividis", "pink"]
PRELOADED_IMAGE_PATHS = {
    "image50.png": "media/image50.png", 
    "spatial.tif": "media/spatial.tif",
    "logo.jpg": "media/logo.jpg"
}

DEFAULT_COLOR_MAP = 'gray'
DEFAULT_KERNEL_SIZE = 3


DEFAULT_SEARCH_WINDOW_SIZE = 51
DEFAULT_FILTER_STRENGTH = 0.1



# Type aliases
ImageArray = np.ndarray
PixelCoordinates = Tuple[int, int]


# Common error handling
def handle_error(e: Exception, message: str):
    st.error(f"{message}. Please check the logs.")

#---------- Main ---------    #
@dataclass
class SidebarUI:
    @staticmethod
    def setup():
        st.sidebar.title("Image Processing Settings")
        image = SidebarUI.create_image_source_ui()

        if image is None:
            return None

        st.sidebar.markdown("### 🎨 Color Map")
        color_map = st.sidebar.selectbox("Select Color Map", AVAILABLE_COLOR_MAPS, index=AVAILABLE_COLOR_MAPS.index(st.session_state.get('color_map', 'gray')))
        st.session_state.color_map = color_map

        display_options = SidebarUI.create_display_options_ui(image)
        advanced_options = SidebarUI.create_advanced_options_ui(image)

        return {
            "image": image,
            "image_array": np.array(image), 
            "cmap": color_map,
            "kernel_size": display_options['kernel_size'],
            "normalization_option": advanced_options['normalization_option'],
            **display_options,
            **advanced_options
        }

    @staticmethod
    def create_image_source_ui():
        image_source_type = st.sidebar.radio("Select Image Source", ("Preloaded Images", "Upload Image"))

        try:
            if image_source_type == "Preloaded Images":
                selected_image_name = st.sidebar.selectbox("Select Image", list(PRELOADED_IMAGE_PATHS.keys()))
                loaded_image = Image.open(PRELOADED_IMAGE_PATHS[selected_image_name]).convert('L')
            else:
                uploaded_file = st.sidebar.file_uploader("Upload Image", type=['png', 'jpg', 'jpeg', 'tif', 'tiff'])
                loaded_image = Image.open(uploaded_file).convert('L') if uploaded_file else None

            if loaded_image is None:
                st.sidebar.warning('Please select or upload an image.')
                return None

            st.sidebar.image(loaded_image, caption="Input Image", use_column_width=True)
            return loaded_image
        except Exception as e:
            handle_error(e, "Error while creating image source UI")
            return None

    @staticmethod
    def create_display_options_ui(image):
        st.sidebar.markdown("### 🖥️ Display Options")
        show_per_pixel_processing = st.sidebar.checkbox("Show Per-Pixel Processing Steps", value=False)

        if 'kernel_size' not in st.session_state:
            st.session_state.kernel_size = 3  # Default value

        kernel_size = st.sidebar.slider("Kernel Size", min_value=3, max_value=21, value=st.session_state.kernel_size, step=2, key='kernel_size_slider')
        st.session_state.kernel_size = kernel_size  # Update session state

        total_pixels = (image.width - kernel_size + 1) * (image.height - kernel_size + 1)
        pixels_to_process = total_pixels

        if show_per_pixel_processing:
            try:
                pixels_to_process = SidebarUI.handle_pixel_processing(total_pixels)
            except Exception as e:
                handle_error(e, "Error while handling pixel processing")

            st.sidebar.write(f"Processing {pixels_to_process:,} out of {total_pixels:,} pixels")

        return {
            "show_per_pixel_processing": show_per_pixel_processing,
            "total_pixels": total_pixels,
            "pixels_to_process": pixels_to_process,
            "kernel_size": kernel_size
        }

    @staticmethod
    def handle_pixel_processing(total_pixels):
        col1, col2 = st.sidebar.columns(2)

        def update_exact_pixel_count():
            st.session_state['exact_pixel_count'] = int(total_pixels * st.session_state['percentage_slider'] / 100)

        def update_percentage():
            if 'exact_pixel_count' in st.session_state:
                st.session_state['percentage_slider'] = int((st.session_state['exact_pixel_count'] / total_pixels) * 100)

        with col1:
            st.slider("Percentage", min_value=0, max_value=100,
                      value=st.session_state.get('percentage_slider', 100),
                      step=1, key="percentage_slider", on_change=update_exact_pixel_count)

        with col2:
            st.number_input("Exact Pixels", min_value=0, max_value=total_pixels,
                            value=st.session_state.get('exact_pixel_count', total_pixels),
                            step=1, key="exact_pixel_count", on_change=update_percentage)

        return st.session_state['exact_pixel_count']

    @staticmethod
    def create_advanced_options_ui(image):
        st.sidebar.markdown("### 🔬 Advanced Options")

        # Add normalization option
        normalization_option = st.sidebar.selectbox(
            "Normalization",
            options=['None', 'Percentile'],
            index=0,
            key="normalization_option",
            help="Choose the normalization method for the image"
        )

        add_noise = st.sidebar.checkbox("Add Gaussian Noise", value=False,
                                        help="Add Gaussian noise to the image")

        try:
            image_np = np.array(image) / 255.0

            # If noise addition is selected, delegate the logic to the extracted method
            if add_noise:
                noise_mean = st.sidebar.number_input("Noise Mean", min_value=0.0, max_value=1.0, value=0.1, step=0.01, format="%.2f", key="noise_mean")
                noise_std = st.sidebar.number_input("Noise Standard Deviation", min_value=0.0, max_value=1.0, value=0.1, step=0.01, format="%.2f", key="noise_std")
                image_np = SidebarUI.add_gaussian_noise(image_np, noise_mean, noise_std)

            # Apply normalization if selected
            if normalization_option == 'Percentile':
                p_low, p_high = np.percentile(image_np, [2, 98])
                image_np = np.clip(image_np, p_low, p_high)
                image_np = (image_np - p_low) / (p_high - p_low)

            return {
                "image_np": image_np, 
                "add_noise": add_noise,
                "normalization_option": normalization_option
            }
        except Exception as e:
            handle_error(e, "Error while creating advanced options UI")
            return {
                "image_np": np.array([]), 
                "add_noise": False,
                "normalization_option": 'None'
            }

    @staticmethod
    def add_gaussian_noise(image_np, mean, std):
        """Adds Gaussian noise to the image."""
        noise = np.random.normal(mean, std, image_np.shape)
        return np.clip(image_np + noise, 0, 1)

@dataclass
class ImageComparison:
    @staticmethod
    def handle(tab, cmap_name, images):
        try:
            with tab:
                st.header("Image Comparison")
            if not images:
                st.warning("No images available for comparison.")
                return
            available_images = list(images.keys())
            image_choice_1, image_choice_2 = ImageComparison.get_image_choices(available_images)
            if image_choice_1 and image_choice_2:
                img1, img2 = images[image_choice_1], images[image_choice_2]
                ImageComparison.display(img1, img2, image_choice_1, image_choice_2, cmap_name)
            else:
                st.info("Select two images to compare.")
        except Exception as e:
            handle_error(e, "Error while handling image comparison")

    @staticmethod
    def get_image_choices(available_images):
        col1, col2 = st.columns(2)
        image_choice_1 = col1.selectbox('Select first image to compare:', [''] + available_images, index=0)
        image_choice_2 = col2.selectbox('Select second image to compare:', [''] + available_images, index=0)
        return image_choice_1, image_choice_2

    @staticmethod
    def display(img1, img2, label1, label2, cmap_name):
        if label1 != label2:
            img1_uint8, img2_uint8 = ImageComparison.normalize_and_colorize([img1, img2], [cmap_name]*2)
            image_comparison(img1=img1_uint8, img2=img2_uint8, label1=label1, label2=label2, make_responsive=True)
            st.subheader("Selected Images")
            st.image([img1_uint8, img2_uint8], caption=[label1, label2])
        else:
            st.error("Please select two different images for comparison.")
            ImageComparison.display_difference_map(img1, img2, cmap_name)

    @staticmethod
    def display_difference_map(img1, img2, cmap_name):
        diff_map = np.abs(img1 - img2)
        display_diff = ImageComparison.normalize_and_colorize([diff_map], [cmap_name])[0]
        st.image(display_diff, caption="Difference Map", use_column_width=True)

    @staticmethod
    def normalize_and_colorize(images, cmap_names):
        colored_images = []
        for img, cmap_name in zip(images, cmap_names):
            normalized = (img - np.min(img)) / (np.max(img) - np.min(img)) if np.max(img) - np.min(img) != 0 else img
            colored = plt.get_cmap(cmap_name)(normalized)[:, :, :3]
            colored_images.append((colored * 255).astype(np.uint8))
        return colored_images

# --------- Called in Analysis Module ---------    #
@dataclass
class FilterResult(ABC):
    """
    Abstract base class for various filtering techniques.
    Defines common attributes and methods across different filters.
    """
    processing_coord: PixelCoordinates
    processing_end_coord: PixelCoordinates
    kernel_size: int
    pixels_processed: int
    image_dimensions: Tuple[int, int]

    @abstractmethod
    def get_filter_data(self) -> Dict[str, Any]:
        pass

    @classmethod
    @abstractmethod
    def get_filter_options(cls) -> List[str]:
        pass

#---------- Function ---------    #
@dataclass(frozen=True)
class ProcessingDetails:
    image_height: int
    image_width: int
    start_x: int
    start_y: int
    end_x: int
    end_y: int
    pixels_to_process: int
    valid_height: int
    valid_width: int
    kernel_size: int

    def __post_init__(self):
        self._validate_dimensions()
        self._validate_coordinates()

    def _validate_dimensions(self):
        if self.image_height <= 0 or self.image_width <= 0:
            raise ValueError("Image dimensions must be positive.")
        if self.valid_height <= 0 or self.valid_width <= 0:
            raise ValueError("Kernel size is too large for the given image dimensions.")
        if self.pixels_to_process < 0:
            raise ValueError("Number of pixels to process must be non-negative.")

    def _validate_coordinates(self):
        if (self.start_x < 0 or self.start_y < 0 or 
            self.end_x >= self.image_width or self.end_y >= self.image_height):
            raise ValueError("Invalid processing coordinates.")

def calculate_processing_details(image: ImageArray, kernel_size: int, max_pixels: Optional[int]) -> ProcessingDetails:
    image_height, image_width = image.shape[:2]
    half_kernel = kernel_size // 2
    valid_height, valid_width = image_height - kernel_size + 1, image_width - kernel_size + 1

    if valid_height <= 0 or valid_width <= 0:
        raise ValueError("Kernel size is too large for the given image dimensions.")
    
    pixels_to_process = min(valid_height * valid_width, max_pixels or float('inf'))
    end_y, end_x = divmod(pixels_to_process - 1, valid_width)
    end_y, end_x = end_y + half_kernel, end_x + half_kernel

    return ProcessingDetails(
        image_height=image_height,
        image_width=image_width,
        start_x=half_kernel,
        start_y=half_kernel,
        end_x=end_x,
        end_y=end_y,
        pixels_to_process=pixels_to_process,
        valid_height=valid_height,
        valid_width=valid_width,
        kernel_size=kernel_size
    )
