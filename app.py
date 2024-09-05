import streamlit as st
from typing import Dict, List
from PIL import Image
import numpy as np
import time

# Ensure streamlit_nested_layout is imported
import streamlit_nested_layout  # noqa: F401

# Import custom modules
from image_processing import handle_image_analysis, handle_image_comparison

# Constants
TABS = ["Speckle Contrast Calculation", "Non-Local Means Denoising", "Speckle Contrast Comparison"]

PAGE_CONFIG = {
    "page_title": "Speckle Contrast Visualization",
    "layout": "wide",
    "page_icon": "favicon.png",
    "initial_sidebar_state": "expanded",
}

# Image Configuration
LOGO_PATH = "media/logo.png"
PRELOADED_IMAGES: Dict[str, str] = {
    "image50.png": "media/image50.png",
    "spatial.tif": "media/spatial.tif",
    "logo.jpg": "media/logo.jpg",
}

# Color Maps
COLOR_MAPS: List[str] = [
    "viridis", "plasma", "inferno", "magma", 
    "cividis", "gray", "pink"
]

# Default values
DEFAULT_STRIDE = 1
DEFAULT_SEARCH_WINDOW_SIZE = "full"
DEFAULT_FILTER_STRENGTH = .10
DEFAULT_KERNEL_SIZE = 7

# Slider Ranges
KERNEL_SIZE_RANGE = (3, 21, DEFAULT_KERNEL_SIZE, 2)  # min, max, default, step
STRIDE_RANGE = (1, 5, DEFAULT_STRIDE)
FILTER_STRENGTH_RANGE = (0.01, 30.0, DEFAULT_FILTER_STRENGTH)


#--------------------------------------------------------------#

# Caching loaded image data
@st.cache_data
def load_image(image_source: str, selected_image: str = None, uploaded_file=None) -> Image.Image:
    if image_source == "Preloaded Image":
        return Image.open(PRELOADED_IMAGES[selected_image]).convert('L')
    elif uploaded_file:
        return Image.open(uploaded_file).convert('L')
    st.warning('Please upload or select an image.')
    st.stop()

# NLM Parameters and Image Processing
def configure_image_processing():
    with st.sidebar:
        # Image source
        image_source = st.radio("Choose Image Source", ["Preloaded Image", "Upload Image"])
        selected_image = st.selectbox("Select Image", list(PRELOADED_IMAGES)) if image_source == "Preloaded Image" else None
        uploaded_file = st.file_uploader("Upload Image") if image_source == "Upload Image" else None
        image = load_image(image_source, selected_image, uploaded_file)
        st.image(image, "Input Image", use_column_width=True)

        # Processing parameters
        kernel_size = st.slider('Kernel Size', *KERNEL_SIZE_RANGE)

        stride = st.slider('Stride', *STRIDE_RANGE, DEFAULT_STRIDE)

        # NLM Parameters
        with st.expander("🔍 NLM Parameters", expanded=True):
            use_full_image = st.checkbox("Use Full Image for Search", value=True)
            if use_full_image:
                search_window_size = "full"
            else:
                min_size = kernel_size + 2
                max_size = min(max(image.width, image.height) // 2, 35)
                search_window_size = st.slider("Search Window Size", min_size, max_size, min_size, step=2)
        
            filter_strength = st.slider("Filter Strength (h)", *FILTER_STRENGTH_RANGE, DEFAULT_FILTER_STRENGTH)

        # Color map and animation settings
        cmap = st.selectbox("🎨 Color Map", COLOR_MAPS, index=0)

        # Convert image to numpy array and normalize
        st.session_state.original_image_np = np.array(image) / 255.0

        # Gaussian noise handling
        if st.checkbox("Toggle Gaussian Noise"):
            noise_mean = st.slider("Noise Mean", 0.0, 1.0, 0.0, 0.01)
            noise_std = st.slider("Noise Std", 0.0, 1.0, 0.1, 0.01)
            noise = np.random.normal(noise_mean, noise_std, st.session_state.original_image_np.shape)
            st.session_state.image_np = np.clip(st.session_state.original_image_np + noise, 0, 1)
            st.success("Noise added!")
        else:
            st.session_state.image_np = st.session_state.original_image_np.copy()

        st.image(st.session_state.image_np, caption="Manipulated Image", use_column_width=True)

    return image, kernel_size, stride, search_window_size, filter_strength, cmap, st.session_state.image_np

# Ensure session state is initialized
if 'max_pixels' not in st.session_state:
    st.session_state.max_pixels = 1

def main():
    """Set the Streamlit page configuration."""
    st.set_page_config(**PAGE_CONFIG)
    st.logo(LOGO_PATH)

    # Configure sidebar and get parameters
    image, kernel_size, stride, search_window_size, filter_strength, cmap, image_np = configure_image_processing()

    # Calculate max processable pixels
    # Calculate max processable pixels
    max_processable_pixels = (image.width - kernel_size + 1) * (image.height - kernel_size + 1)
    
    # Add a checkbox to toggle the animation
    animate = st.checkbox("Animate pixel processing", value=False)
    
    # Placeholder for the slider
    slider_placeholder = st.empty()
    
    if animate:
        current_value = st.session_state.max_pixels
        # Loop to animate the slider
        while current_value <= max_processable_pixels:
            # Update the slider and session state value
            st.session_state.max_pixels = current_value
            slider_placeholder.slider("Pixels to process", 1, max_processable_pixels, st.session_state.max_pixels)
            current_value += 1
            time.sleep(0.001)
    else:
        # Regular slider when animation is off
        st.session_state.max_pixels = slider_placeholder.slider("Pixels to process", 1, max_processable_pixels, st.session_state.max_pixels)

    # Create tabs
    tabs = st.tabs(TABS)

    # Process images
    std_dev_image, speckle_contrast_image, mean_image = handle_image_analysis(
        tabs[0], image_np, kernel_size, stride, st.session_state.max_pixels, cmap, "speckle", 
        search_window_size, filter_strength)[:3]

    denoised_image = handle_image_analysis(
        tabs[1], image_np, kernel_size, stride, st.session_state.max_pixels, cmap, "nlm", 
        search_window_size, filter_strength)[0]

    # Handle comparison tab
    handle_image_comparison(
        tab=tabs[2],
        cmap_name=cmap,
        images={
            'Unprocessed Image': image_np,
            'Standard Deviation': std_dev_image,
            'Speckle Contrast': speckle_contrast_image,
            'Mean Filter': mean_image,
            'Denoised Image': denoised_image
        }
    )

if __name__ == "__main__":
    main()

