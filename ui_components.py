import streamlit as st
from streamlit_image_comparison import image_comparison
import matplotlib.pyplot as plt
from matplotlib.colors import Colormap
from PIL import Image
import numpy as np
from config import (PRELOADED_IMAGES, COLOR_MAPS, 
                    KERNEL_SIZE_RANGE, STRIDE_RANGE,
                    FILTER_STRENGTH_RANGE, ANIMATION_SPEED_RANGE,
                    DEFAULT_KERNEL_SIZE, DEFAULT_STRIDE,
                    DEFAULT_SEARCH_WINDOW_SIZE, DEFAULT_FILTER_STRENGTH,
                    DEFAULT_ANIMATION_SPEED,
                    SESSION_STATE_KEYS)
from typing import Tuple, Dict


# Function to get dynamic search window size range
def get_search_window_size_range(kernel_size: int, image_size: Tuple[int, int]) -> Tuple[int, int, int]:
    """
    Calculate the range for the search window size slider based on kernel size and image dimensions.
    
    :param kernel_size: Size of the kernel
    :param image_size: Tuple containing (width, height) of the image
    :return: Tuple of (min, max, step) for the search window size slider
    """
    min_size = kernel_size + 2
    max_size = min(max(image_size) // 2, 35)
    step = 2  # Using a step of 2 to ensure odd numbers
    return int(min_size), int(max_size), int(step)

# Main configuration function
def configure_sidebar() -> Tuple[Image.Image, int, int, float, int, str, float, np.ndarray]:
    """Configure sidebar options and return user inputs."""
    with st.sidebar:
        image, image_np = configure_image_settings()
        cmap = select_color_map()
        kernel_size, stride, search_window_size, filter_strength = configure_processing_parameters(image)
        animation_speed = configure_animation_settings()
        configure_image_manipulation(image_np)

    return image, kernel_size, search_window_size, filter_strength, stride, cmap, animation_speed, st.session_state.get('image_np', image_np)

# Image configuration functions
def configure_image_settings():
    st.markdown("### 🖼️ Image Configuration")
    with st.expander("Image Settings", expanded=True):
        image_source = st.radio("Choose Image Source", ["Preloaded Image", "Upload Image"])
        selected_image = st.selectbox("Select Preloaded Image", list(PRELOADED_IMAGES.keys())) if image_source == "Preloaded Image" else None
        uploaded_file = st.file_uploader("Upload an Image") if image_source == "Upload Image" else None

        image = load_image(image_source, selected_image, uploaded_file)
        st.image(image, caption="Input Image", use_column_width=True)
        
        image_np = process_image(image)
        initialize_session_state()

    return image, image_np

def select_color_map():
    return st.selectbox("🎨 Select Color Map", COLOR_MAPS, index=0, help="Choose a color map for all visualizations.")

def load_image(image_source: str, selected_image: str, uploaded_file) -> Image.Image:
    """Load the image based on the user's selection."""
    if image_source == "Preloaded Image":
        return Image.open(PRELOADED_IMAGES[selected_image]).convert('L')
    elif uploaded_file:
        return Image.open(uploaded_file).convert('L')
    else:
        st.warning('Please upload an image or select a preloaded image.')
        st.stop()

@st.cache_data
def process_image(image: Image.Image) -> np.ndarray:
    """Convert the image to a numpy array and normalize pixel values."""
    return np.array(image) / 255.0

# Processing parameter configuration
def configure_processing_parameters(image):
    st.markdown("### 🛠️ Processing Parameters")
    with st.expander("Image Processing", expanded=True):
        kernel_size = st.slider('Kernel Size', *KERNEL_SIZE_RANGE, DEFAULT_KERNEL_SIZE, help="Size of the sliding window (patch size).")
        stride = st.slider('Stride', *STRIDE_RANGE, DEFAULT_STRIDE, help="Stride of the sliding window.")
        
        with st.expander("🔍 Non-Local Means Parameters", expanded=True):
            use_full_image_window = st.checkbox("Use Full Image as Search Window", value=False, help="If checked, the search window will cover the entire image.")
            
            if not use_full_image_window:
                image_size = (image.width, image.height)
                try:
                    min_size, max_size, step = get_search_window_size_range(kernel_size, image_size)
                    search_window_size = st.slider("Search Window Size", min_value=min_size, max_value=max_size, step=step,
                                                   value=min_size, help="Size of the window used to search for similar patches in the image.")
                except Exception as e:
                    st.warning(f"Error setting search window size: {str(e)}. Using default value.")
                    search_window_size = DEFAULT_SEARCH_WINDOW_SIZE
            else:
                search_window_size = "full"
            
            filter_strength = st.slider("Filter Strength (h)", *FILTER_STRENGTH_RANGE, DEFAULT_FILTER_STRENGTH,
                                        help="Controls the decay of the similarity function, affecting how strongly similar patches are weighted.")

    return kernel_size, stride, search_window_size, filter_strength

# Animation configuration
def configure_animation_settings():
    st.markdown("### 🎞️ Animation Controls")
    with st.expander("Animation Settings", expanded=True):
        animation_speed = st.slider("Animation Speed (seconds per frame)", *ANIMATION_SPEED_RANGE, DEFAULT_ANIMATION_SPEED)
        handle_animation_controls()
    return animation_speed

def handle_animation_controls():
    """Handle the start and stop animation buttons."""
    col1, col2 = st.columns(2)
    
    if col1.button('Start'):
        st.session_state.is_animating = True
        st.session_state.is_paused = False
    
    if col2.button('Stop'):
        st.session_state.is_animating = False
        st.session_state.is_paused = False

    # Display current animation state
    animation_state = "Running" if st.session_state.is_animating else "Stopped"
    st.write(f"Animation State: {animation_state}")

# Image manipulation functions
def configure_image_manipulation(image_np):
    st.markdown("### 🎭 Image Manipulation")
    with st.expander("Noise Addition", expanded=True):
        if 'original_image_np' not in st.session_state:
            st.session_state.original_image_np = image_np.copy()

        col1, col2 = st.columns(2)
        if col1.button("Add Gaussian Noise"):
            add_gaussian_noise_to_image()
        if col2.button("Remove Noise"):
            remove_noise_from_image()

def add_gaussian_noise_to_image():
    noise_mean = st.slider("Noise Mean", 0.0, 1.0, 0.0, 0.01)
    noise_std = st.slider("Noise Standard Deviation", 0.0, 1.0, 0.1, 0.01)
    image_np = add_gaussian_noise(st.session_state.original_image_np, noise_mean, noise_std)
    st.session_state.image_np = image_np
    st.success("Gaussian noise added to the image!")

def remove_noise_from_image():
    if 'original_image_np' in st.session_state:
        st.session_state.image_np = st.session_state.original_image_np.copy()
        st.success("Noise removed. Original image restored!")
    else:
        st.warning("No original image found to restore.")

# Session state management
def initialize_session_state():
    for key in SESSION_STATE_KEYS:
        if key not in st.session_state:
            if key.startswith("is_"):
                st.session_state[key] = False
            elif key == "cache":
                st.session_state[key] = {}
            else:
                st.session_state[key] = "Standard"

# Image comparison functions
def handle_comparison_tab(tab, cmap_name: str, images: Dict[str, np.ndarray]):
    with tab:
        st.header("Image Comparison")
        
        available_images = list(images.keys())
        col1, col2 = st.columns(2)
        image_choice_1 = col1.selectbox('Select first image to compare:', [''] + available_images, index=0)
        image_choice_2 = col2.selectbox('Select second image to compare:', [''] + available_images, index=0)
        
        cmap = plt.get_cmap(cmap_name)

        if image_choice_1 and image_choice_2:
            if image_choice_1 != image_choice_2:
                img1, img2 = images[image_choice_1], images[image_choice_2]
                display_image_comparison(img1, img2, image_choice_1, image_choice_2, cmap)
            else:
                st.error("Please select two different images for comparison.")
        else:
            st.info("Select two images to compare.")

def display_image_comparison(img1: np.ndarray, img2: np.ndarray, label1: str, label2: str, cmap: Colormap):
    def normalize_and_apply_cmap(img):
        normalized = (img - np.min(img)) / (np.max(img) - np.min(img))
        return (cmap(normalized)[:, :, :3] * 255).astype(np.uint8)

    img1_uint8, img2_uint8 = map(normalize_and_apply_cmap, [img1, img2])

    image_comparison(img1=img1_uint8, img2=img2_uint8, label1=label1, label2=label2, make_responsive=True)
    st.subheader("Selected Images")
    st.image([img1_uint8, img2_uint8], caption=[label1, label2])

def get_images_to_compare(image_choice_1, image_choice_2, images):
    return images[image_choice_1], images[image_choice_2]

def add_gaussian_noise(image: np.ndarray, mean: float = 0, std: float = 0.1) -> np.ndarray:
    noise = np.random.normal(mean, std, image.shape)
    noisy_image = image + noise
    return np.clip(noisy_image, 0, 1)  # Ensure values are between 0 and 1
