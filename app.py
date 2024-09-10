import streamlit as st
import streamlit_nested_layout  # type: ignore  # noqa: F401
import numpy as np
import logging
from PIL import Image
from typing import Any, Dict, List, Optional

from visualization import prepare_comparison_images
from image_processing import handle_image_comparison, process_techniques
from utils import calculate_processing_details

PRELOADED_IMAGES = {
    "image50.png": "media/image50.png",
    "spatial.tif": "media/spatial.tif",
    "logo.jpg": "media/logo.jpg"
}

PAGE_CONFIG = {
    "page_title": "Speckle Contrast Visualization",
    "layout": "wide",
    "page_icon": "favicon.png",
    "initial_sidebar_state": "expanded"
}

COLOR_MAPS = ["viridis", "plasma", "inferno", "magma", "cividis", "gray", "pink"]

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_image() -> Optional[Image.Image]:
    st.sidebar.markdown("### 📷 Image Source")
    image_source = st.sidebar.radio("", ("Preloaded Images", "Upload Image"))
    
    try:
        if image_source == "Preloaded Images":
            selected_image = st.sidebar.selectbox("Select Image", list(PRELOADED_IMAGES))
            image = Image.open(PRELOADED_IMAGES[selected_image]).convert('L')
        else:
            uploaded_file = st.sidebar.file_uploader("Upload Image")
            image = Image.open(uploaded_file).convert('L') if uploaded_file else None

        if image is None:
            st.sidebar.warning('Please select or upload an image.')
            return None

        st.sidebar.image(image, "Input Image", use_column_width=True)
        return image
    except Exception as e:
        logger.error(f"Error loading image: {str(e)}")
        st.sidebar.error("Failed to load the image. Please try again or choose a different image.")
        return None

def get_processing_params(image: Image.Image) -> Dict[str, Any]:
    try:
        st.markdown("### ⚙️ Processing Parameters")
        col1, col2 = st.columns(2)

        with col1:
            kernel_size = st.number_input('Kernel Size (pixels)', min_value=3, max_value=21, value=7, step=2)
            use_full_image = st.checkbox("Use Full Image for Search", value=False, 
                                         help="Toggle to use the entire image as the search window")
            search_window_size = (
                st.number_input("Search Window Size (pixels)", 
                                min_value=kernel_size + 2, 
                                max_value=min(max(image.width, image.height) // 2, 35),
                                value=kernel_size + 2,
                                step=2,
                                help="Size of the search window for NL-Means denoising")
                if not use_full_image else None
            )

        with col2:
            filter_strength = st.slider("Filter Strength (h)", min_value=0.01, max_value=30.0, value=0.10, step=0.01)
            cmap = st.selectbox("🎨 Color Map", COLOR_MAPS, index=0)

        if st.button("Reset to Defaults"):
            kernel_size = 7
            use_full_image = False
            search_window_size = 9
            filter_strength = 0.10
            cmap = "viridis"

        return locals()
    except Exception as e:
        logger.error(f"Error getting processing parameters: {str(e)}")
        st.error("Failed to set processing parameters. Using default values.")
        return {"kernel_size": 7, "search_window_size": None, "filter_strength": 0.1, "cmap": "viridis"}

def get_display_options(image: Image.Image, kernel_size: int) -> Dict[str, Any]:
    try:
        with st.sidebar.expander("🖼️ Display Options", expanded=True):
            show_full_processed = st.checkbox("Show Fully Processed Image", value=True)
            max_pixels = (image.width - kernel_size + 1) * (image.height - kernel_size + 1)

            pixels_to_process = (
                max_pixels if show_full_processed else
                st.number_input("Pixels to process", min_value=1, max_value=max_pixels, 
                                value=1, step=1, key='pixels_to_process')
            )

        return locals()
    except Exception as e:
        logger.error(f"Error getting display options: {str(e)}")
        st.sidebar.error("Failed to set display options. Using default values.")
        return {"show_full_processed": True, "max_pixels": max_pixels, "pixels_to_process": max_pixels}

def get_advanced_options(image: Image.Image) -> Dict[str, Any]:
    try:
        with st.sidebar.expander("🔬 Advanced Options"):
            add_noise = st.checkbox("Add Gaussian Noise", value=False,
                                    help="Add Gaussian noise to the image")
            if add_noise:
                noise_mean = st.number_input("Noise Mean", min_value=0.0, max_value=1.0, value=0.0, step=0.01, format="%.2f")
                noise_std = st.number_input("Noise Std", min_value=0.0, max_value=1.0, value=0.1, step=0.01, format="%.2f")
                image_np = np.clip(np.array(image) / 255.0 + np.random.normal(noise_mean, noise_std, np.array(image).shape), 0, 1)
            else:
                image_np = np.array(image) / 255.0
        return locals()
    except Exception as e:
        logger.error(f"Error getting advanced options: {str(e)}")
        st.sidebar.error("Failed to set advanced options. Using default values.")
        return {"image_np": np.array(image) / 255.0}

def setup_sidebar() -> Optional[Dict[str, Any]]:
    try:
        st.sidebar.title("Image Processing Settings")
        image = load_image()
        if image is None:
            return None

        display_options = get_display_options(image, 7)  # Use default kernel_size of 7
        advanced_options = get_advanced_options(image)

        return {
            "image": image,
            "image_np": advanced_options['image_np'],
            **display_options,
            **advanced_options
        }
    except Exception as e:
        logger.error(f"Error setting up sidebar: {str(e)}")
        st.sidebar.error("An error occurred while setting up the sidebar. Please try again.")
        return None

def setup_page_config() -> None:
    st.set_page_config(**PAGE_CONFIG)
    st.logo("media/logo.png")

def setup_tabs() -> List[st.delta_generator.DeltaGenerator]:
    return st.tabs(["Speckle", "NL-Means", "Image Comparison"])

def prepare_analysis_params(sidebar_params: Dict[str, Any], details: Dict[str, Any]) -> Dict[str, Any]:
    try:
        return {
            "image_np": sidebar_params['image_np'],
            "kernel_size": sidebar_params['kernel_size'],
            "search_window_size": sidebar_params['search_window_size'],
            "filter_strength": sidebar_params['filter_strength'],
            "cmap": sidebar_params['cmap'],
            "max_pixels": details['pixels_to_process'],
            "height": details['height'],
            "width": details['width'],
            "show_full_processed": sidebar_params['show_full_processed']
        }
    except KeyError as e:
        logger.error(f"Missing key in parameters: {str(e)}")
        st.error("Some required parameters are missing. Please check your inputs.")
        return {}
    except Exception as e:
        logger.error(f"Error preparing analysis parameters: {str(e)}")
        st.error("Failed to prepare analysis parameters. Please try again.")
        return {}

def main() -> None:
    setup_page_config()
    
    try:
        sidebar_params = setup_sidebar()
        if sidebar_params is None:
            st.stop()

        # Move processing parameters to the top of the page
        processing_params = get_processing_params(sidebar_params['image'])
        
        tabs = setup_tabs()
        
        details = calculate_processing_details(
            sidebar_params['image_np'], 
            processing_params['kernel_size'], 
            None if sidebar_params['show_full_processed'] else sidebar_params['pixels_to_process']
        )
        
        analysis_params = prepare_analysis_params({**sidebar_params, **processing_params}, details)
        if not analysis_params:
            st.stop()
        
        st.session_state.tabs = tabs 
        st.session_state.sidebar_params = sidebar_params
        st.session_state.analysis_params = analysis_params
        
        process_techniques(analysis_params)
        
        comparison_images = prepare_comparison_images()
        handle_image_comparison(tabs[2], analysis_params['cmap'], comparison_images)
    
    except Exception as e:
        logger.error(f"Unhandled exception in main: {str(e)}")
        st.error("An unexpected error occurred. Please try reloading the application.")
        st.exception(e)

if __name__ == "__main__":
    main()