# Imports
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple,NamedTuple
from src.plotting import AVAILABLE_COLOR_MAPS, PRELOADED_IMAGE_PATHS
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from PIL import Image
from streamlit_image_comparison import image_comparison



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
    def setup() -> Optional[Dict[str, Any]]:
        st.sidebar.title("Image Processing Settings")

        with st.sidebar.expander("Image Selector",expanded=True):
            image = SidebarUI._create_image_source_ui()
            st.sidebar.markdown("### 🎨 Color Map")
            color_map = SidebarUI._select_color_map()
        
        display_options = SidebarUI._create_display_options_ui(image)

        with st.sidebar.expander("NLM Parameters",expanded=True):
            nlm_params = SidebarUI._create_nlm_options_ui(image)

        with st.sidebar.expander("Advanced Options",expanded=True):
            advanced_options = SidebarUI._create_advanced_options_ui(image)

        return {
            "image": image,
            "image_array": np.array(image),
            "cmap": color_map,
            **display_options,
            **nlm_params,  # Remove .__dict__ as nlm_params is already a dictionary
            **advanced_options
        }
    
    @staticmethod
    def _create_image_source_ui() -> Optional[Image.Image]:
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
            st.error(f"Error while creating image source UI: {e}")
            return None

    @staticmethod        
    def _select_color_map():
        return st.sidebar.selectbox(
            "Select Color Map",
            AVAILABLE_COLOR_MAPS, 
            index=AVAILABLE_COLOR_MAPS.index(st.session_state.get('color_map', 'gray'))
        )

    @staticmethod
    def _create_display_options_ui(image: Image.Image) -> Dict[str, Any]:
        st.sidebar.markdown("### 🖥️ Display Options")
        show_per_pixel_processing = st.sidebar.checkbox("Show Per-Pixel Processing Steps", value=False)
        kernel_size = SidebarUI._select_kernel_size()

        total_pixels = (image.width - kernel_size + 1) * (image.height - kernel_size + 1)
        pixels_to_process = SidebarUI._handle_pixel_processing(total_pixels) if show_per_pixel_processing else total_pixels

        return {
            "show_per_pixel_processing": show_per_pixel_processing,
            "total_pixels": total_pixels,
            "pixels_to_process": pixels_to_process,
            "kernel_size": kernel_size
        }

    @staticmethod
    def _select_kernel_size():
        if 'kernel_size' not in st.session_state:
            st.session_state.kernel_size = 3
        return st.sidebar.slider("Kernel Size", min_value=3, max_value=21, value=st.session_state.kernel_size, step=2)

    @staticmethod
    def _handle_pixel_processing(total_pixels: int) -> int:
        col1, col2 = st.sidebar.columns(2)

        if 'exact_pixel_count' not in st.session_state:
            st.session_state.exact_pixel_count = total_pixels
        if 'percentage_slider' not in st.session_state:
            st.session_state.percentage_slider = 100

        with col1:
            percentage = st.slider("Percentage", min_value=1, max_value=100, value=st.session_state.percentage_slider, step=1)
            st.session_state.exact_pixel_count = int(total_pixels * percentage / 100)

        with col2:
            exact_count = st.number_input("Exact Pixels", min_value=0, max_value=total_pixels, value=st.session_state.exact_pixel_count, step=1)
            st.session_state.percentage_slider = int((exact_count / total_pixels) * 100)

        return st.session_state.exact_pixel_count

    @staticmethod
    def _create_nlm_options_ui(image: Image.Image) -> Dict[str, Any]:
        try:
            image_shape = image.size
            max_search_window = min(101, min(image_shape))
            default_search_window = min(21, max_search_window)
            search_window_size = st.slider(
                "Search Window Size",
                min_value=3,
                max_value=max_search_window,
                value=default_search_window,
                step=2,
                help="Size of the search window for NLM (must be odd)"
            )
            search_window_size = search_window_size if search_window_size % 2 == 1 else search_window_size + 1
        
            filter_strength = st.slider(
                "Filter Strength (h)",
                min_value=0.1,
                max_value=20.0,
                value=10.0,
                step=0.1,
                format="%.1f",
                help="Filter strength for NLM (higher values result in more smoothing)"
                )
                
            return {
                "search_window_size": search_window_size,
                "filter_strength": filter_strength
            }
        except Exception as e:
            st.sidebar.error(f"Error creating NLM options: {e}")
            return {"search_window_size": 21, "filter_strength": 10.0}  # Default values
        

    @staticmethod
    def _create_advanced_options_ui(image: Image.Image) -> Dict[str, Any]:

        normalization_option = SidebarUI._select_normalization_option()
        add_noise, noise_params = SidebarUI._add_gaussian_noise_option()

       
        image_np = np.array(image) / 255.0
        
        if add_noise:
            image_np = SidebarUI._apply_gaussian_noise(image_np, **noise_params)
            
        if normalization_option == 'Percentile':
            image_np = SidebarUI._normalize_percentile(image_np)

        return {
            "image_np": image_np,
            "add_noise": add_noise,
            "normalization_option": normalization_option
        }
    

    @staticmethod
    def _select_normalization_option():
        return st.sidebar.selectbox(
            "Normalization",
            options=['None', 'Percentile'],
            index=0,
            help="Choose the normalization method for the image"
        )

    @staticmethod
    def _add_gaussian_noise_option():
        add_noise = st.sidebar.checkbox("Add Gaussian Noise", value=False, help="Add Gaussian noise to the image")
        noise_params = {}

        if add_noise:
            noise_params['mean'] = st.sidebar.number_input("Noise Mean", min_value=0.0, max_value=1.0, value=0.1, step=0.01, format="%.2f")
            noise_params['std'] = st.sidebar.number_input("Noise Standard Deviation", min_value=0.0, max_value=1.0, value=0.1, step=0.01, format="%.2f")

        return add_noise, noise_params    

    @staticmethod
    def _apply_gaussian_noise(image_np: np.ndarray, mean: float, std: float) -> np.ndarray:
        noise = np.random.normal(mean, std, image_np.shape)
        return np.clip(image_np + noise, 0, 1)

    @staticmethod
    def _normalize_percentile(image_np: np.ndarray) -> np.ndarray:
        p_low, p_high = np.percentile(image_np, [2, 98])
        image_np = np.clip(image_np, p_low, p_high)
        image_np = (image_np - p_low) / (p_high - p_low)
        return image_np

#--------- Called in SidebarUI ---------    #
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
class Point(NamedTuple):
    x: int
    y: int

class Dimensions(NamedTuple):
    width: int
    height: int

@dataclass(frozen=True)
class ProcessingDetails:
    """A dataclass to store image processing details and enforce immutability."""
    image_dimensions: Dimensions
    valid_dimensions: Dimensions
    start_point: Point
    end_point: Point
    pixels_to_process: int
    kernel_size: int

    def __post_init__(self):
        self._validate_dimensions()
        self._validate_coordinates()

    def _validate_dimensions(self):
        if self.image_dimensions.width <= 0 or self.image_dimensions.height <= 0:
            raise ValueError("Image dimensions must be positive.")
        if self.valid_dimensions.width <= 0 or self.valid_dimensions.height <= 0:
            raise ValueError("Kernel size is too large for the given image dimensions.")
        if self.pixels_to_process < 0:
            raise ValueError("Number of pixels to process must be non-negative.")

    def _validate_coordinates(self):
        """Ensure the processing coordinates are valid and within image bounds."""
        if self.start_point.x < 0 or self.start_point.y < 0:
            raise ValueError("Start coordinates must be non-negative.")
        if self.end_point.x >= self.image_dimensions.width or self.end_point.y >= self.image_dimensions.height:
            raise ValueError("End coordinates exceed image boundaries.")

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
        image_dimensions=Dimensions(width=image_width, height=image_height),
        valid_dimensions=Dimensions(width=valid_width, height=valid_height),
        start_point=Point(x=half_kernel, y=half_kernel),
        end_point=Point(x=end_x, y=end_y),
        pixels_to_process=pixels_to_process,
        kernel_size=kernel_size
    )