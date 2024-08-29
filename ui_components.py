import streamlit as st
from image_processing import load_image, process_image
from config import PRELOADED_IMAGES
from streamlit_image_comparison import image_comparison
import numpy as np

def configure_sidebar() -> tuple:
    """Configure sidebar options and return user inputs."""
    with st.sidebar:
        with st.expander("Image Configuration", expanded=True):
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
                "Select Color Map",
                ["viridis", "plasma", "inferno", "magma", "cividis", "gray"],
                index=0,
                help="Choose a color map for all visualizations."
            )

        with st.expander("Processing Parameters"):
            kernel_size = st.slider('Kernel Size', 1, 10, 3, help="Size of the sliding window (patch size).")
            stride = st.slider('Stride', 1, 5, 1, help="Stride of the sliding window.")

        with st.expander("Animation Controls"):
            animation_speed = st.slider("Animation Speed (seconds per frame)", 0.001, 0.5, 0.01, 0.01)
            
            # Handle animation controls
            handle_animation_controls()

    return image, kernel_size, stride, cmap, animation_speed, image_np

def initialize_session_state():
    """Initialize session state variables."""
    if 'is_animating' not in st.session_state:
        st.session_state.is_animating = False
    if 'is_paused' not in st.session_state:
        st.session_state.is_paused = False
    if 'cache' not in st.session_state:
        st.session_state.cache = {}
    if 'animation_mode' not in st.session_state:
        st.session_state.animation_mode = 'Standard'

def handle_animation_controls():
    """Handle the start/pause/stop animation buttons."""
    col1, col2, col3 = st.columns(3)
    with col1:
        start_button = st.button('Start')
    with col2:
        pause_button = st.button('Pause')
    with col3:
        stop_button = st.button('Stop')



    if start_button:
        st.session_state.is_animating = True
        st.session_state.is_paused = False
    if pause_button:
        st.session_state.is_paused = not st.session_state.is_paused
    if stop_button:
        st.session_state.is_animating = False
        st.session_state.is_paused = False


def display_original_image_section():
    with st.expander("Original Image with Current Kernel", expanded=True):
        original_image_placeholder = st.empty()
        with st.popover("Zoomed-in Kernel"):
            zoomed_kernel_placeholder = st.empty()

    with st.expander("Mean Filter", expanded=False):
        mean_filter_placeholder = st.empty()
        with st.popover("Zoomed-in Mean"):
            zoomed_mean_placeholder = st.empty()

    return original_image_placeholder, mean_filter_placeholder, zoomed_kernel_placeholder, zoomed_mean_placeholder

def display_speckle_contrast_section():
    with st.expander("Speckle Contrast", expanded=True):
        speckle_contrast_placeholder = st.empty()
        with st.popover("Zoomed-in Speckle Contrast"):
            zoomed_sc_placeholder = st.empty()

    with st.expander("Standard Deviation Filter", expanded=False):
        std_dev_filter_placeholder = st.empty()
        with st.popover("Zoomed-in Std Dev"):
            zoomed_std_placeholder = st.empty()

    return speckle_contrast_placeholder, std_dev_filter_placeholder, zoomed_sc_placeholder, zoomed_std_placeholder

def display_speckle_contrast_formula(formula_placeholder):
    with formula_placeholder.container():
        st.latex(r'SC_{{({}, {})}} = \frac{{\sigma}}{{\mu}} = \frac{{{:.3f}}}{{{:.3f}}} = {:.3f}'.format(0, 0, 0.0, 0.0, 0.0))

def display_image_comparison_error(image_choice_1, image_choice_2):
    if image_choice_1 == image_choice_2 and image_choice_1:
        st.error("Please select two different images for comparison.")
    else:
        st.info("Select two images to compare.")

def apply_colormap_to_images(img1, img2, cmap):
    img1_normalized = (img1 - np.min(img1)) / (np.max(img1) - np.min(img1))
    img2_normalized = (img2 - np.min(img2)) / (np.max(img2) - np.min(img2))

    img1_colored = cmap(img1_normalized)
    img2_colored = cmap(img2_normalized)

    img1_uint8 = (img1_colored[:, :, :3] * 255).astype(np.uint8)
    img2_uint8 = (img2_colored[:, :, :3] * 255).astype(np.uint8)
    
    return img1_uint8, img2_uint8


def display_image_comparison(img1, img2, label1, label2, cmap):
    img1_uint8, img2_uint8 = apply_colormap_to_images(img1, img2, cmap)
    image_comparison(img1=img1_uint8, img2=img2_uint8, label1=label1, label2=label2, make_responsive=True, width=700)
    st.subheader("Selected Images")
    st.image([img1_uint8, img2_uint8], caption=[label1, label2], width=350)
