import streamlit as st
from streamlit_image_comparison import image_comparison
import matplotlib.pyplot as plt
from matplotlib.colors import Colormap
from PIL import Image
import numpy as np
from config import (PRELOADED_IMAGES, COLOR_MAPS, 
                    DEFAULT_KERNEL_SIZE, 
                    DEFAULT_STRIDE, 
                    DEFAULT_SEARCH_WINDOW_SIZE, 
                    DEFAULT_FILTER_STRENGTH, 
                    DEFAULT_ANIMATION_SPEED, 
                    KERNEL_SIZE_RANGE, 
                    STRIDE_RANGE, 
                    SEARCH_WINDOW_SIZE_RANGE, 
                    FILTER_STRENGTH_RANGE, 
                    ANIMATION_SPEED_RANGE,
                    SESSION_STATE_KEYS)
from typing import Tuple

#-----------------------------Stuff ------------------------------ #


def configure_sidebar() -> Tuple[Image.Image, int, int, float, int, str, float, np.ndarray]:
    """Configure sidebar options and return user inputs."""
    with st.sidebar:
        st.markdown("### 🖼️ Image Configuration")
        with st.expander("Image Settings", expanded=True):
            image_source = st.radio("Choose Image Source", ["Preloaded Image", "Upload Image"])
            selected_image = st.selectbox(
                "Select Preloaded Image", 
                list(PRELOADED_IMAGES.keys())
            ) if image_source == "Preloaded Image" else None
            uploaded_file = st.file_uploader("Upload an Image") if image_source == "Upload Image" else None

            # Load and display the image
            image = load_image(image_source, selected_image, uploaded_file)
            st.image(image, caption="Input Image", use_column_width=True)
            
            # Process the image
            image_np = process_image(image)
            
            # Initialize session state
            initialize_session_state()

            # Select color map
            cmap = st.selectbox(
                "🎨 Select Color Map",
                COLOR_MAPS,
                index=0,
                help="Choose a color map for all visualizations."
            )

        st.markdown("### 🛠️ Processing Parameters")
        with st.expander("Image Processing", expanded=True):
            kernel_size = st.slider('Kernel Size', *KERNEL_SIZE_RANGE, DEFAULT_KERNEL_SIZE, help="Size of the sliding window (patch size).")
            stride = st.slider('Stride', *STRIDE_RANGE, DEFAULT_STRIDE, help="Stride of the sliding window.")
            with st.expander("🔍 Non-Local Means Parameters", expanded=False):
                use_full_image_window = st.checkbox("Use Full Image as Search Window", value=False, help="If checked, the search window will cover the entire image.")
                
                if not use_full_image_window:
                    search_window_size = st.slider(
                        "Search Window Size", 
                        *SEARCH_WINDOW_SIZE_RANGE,
                        DEFAULT_SEARCH_WINDOW_SIZE,
                        help="Size of the window used to search for similar patches in the image."
                    )
                else:
                    search_window_size = None  # Use None to indicate full image search window
                
                filter_strength = st.slider(
                    "Filter Strength (h)", 
                    *FILTER_STRENGTH_RANGE,
                    DEFAULT_FILTER_STRENGTH,
                    help="Controls the decay of the similarity function, affecting how strongly similar patches are weighted."
                )

        st.markdown("### 🎞️ Animation Controls")
        with st.expander("Animation Settings", expanded=True):
            animation_speed = st.slider("Animation Speed (seconds per frame)", *ANIMATION_SPEED_RANGE, DEFAULT_ANIMATION_SPEED)
            
            # Handle animation controls
            handle_animation_controls()

    return image, kernel_size, search_window_size, filter_strength, stride, cmap, animation_speed, image_np

# Function to initialize session state
def initialize_session_state():
    for key in SESSION_STATE_KEYS:
        if key not in st.session_state:
            if key.startswith("is_"):
                st.session_state[key] = False
            elif key == "cache":
                st.session_state[key] = {}
            else:
                st.session_state[key] = "Standard"

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


#------------------- Image Processing -------------------#

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

#----------------------------- Comparison Plotting Stuff ------------------------------ #


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

def handle_comparison_tab(tab, cmap_name, std_dev_image, speckle_contrast_image, mean_image, original_image):
    with tab:
        st.header("Speckle Contrast Comparison")
        
        images = {
            'Unprocessed Image': original_image,
            'Standard Deviation': std_dev_image,
            'Speckle Contrast': speckle_contrast_image,
            'Mean Filter': mean_image
        }
        
        available_images = list(images.keys())
        col1, col2 = st.columns(2)
        image_choice_1 = col1.selectbox('Select first image to compare:', [''] + available_images, index=0)
        image_choice_2 = col2.selectbox('Select second image to compare:', [''] + available_images, index=0)
        
        cmap = plt.get_cmap(cmap_name)

        if image_choice_1 and image_choice_2:
            if image_choice_1 != image_choice_2:
                img1, img2 = get_images_to_compare(image_choice_1, image_choice_2, images)
                display_image_comparison(img1, img2, image_choice_1, image_choice_2, cmap)
            else:
                st.error("Please select two different images for comparison.")
        else:
            st.info("Select two images to compare.")
