import streamlit as st
from streamlit_image_comparison import image_comparison
from helpers import apply_colormap_to_images
import matplotlib.pyplot as plt
import io
from PIL import Image
import numpy as np
from config import (initialize_session_state, 
                    PRELOADED_IMAGES, COLOR_MAPS, 
                    DEFAULT_KERNEL_SIZE, 
                    DEFAULT_STRIDE, 
                    DEFAULT_SEARCH_WINDOW_SIZE, 
                    DEFAULT_FILTER_STRENGTH, 
                    DEFAULT_ANIMATION_SPEED, 
                    KERNEL_SIZE_RANGE, 
                    STRIDE_RANGE, 
                    SEARCH_WINDOW_SIZE_RANGE, 
                    FILTER_STRENGTH_RANGE, 
                    ANIMATION_SPEED_RANGE)

#-----------------------------Stuff ------------------------------ #

def create_section(title: str, expanded_main: bool = False, expanded_zoomed: bool = False):
    """Create a Streamlit section with main and zoomed views."""
    with st.expander(title, expanded=expanded_main):
        main_placeholder = st.empty()
        with st.expander(f"Zoomed-in {title.split()[0]}", expanded=expanded_zoomed):
            zoomed_placeholder = st.empty()
    return main_placeholder, zoomed_placeholder

def configure_sidebar() -> tuple:
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

def save_results_section(std_dev_image, speckle_contrast_image, mean_image):
    def create_image_download_button(image, filename, button_text):
        """Convert a numpy array to an image and create a download button."""
        img_buffer = io.BytesIO()
        Image.fromarray((255 * image).astype(np.uint8)).save(img_buffer, format='PNG')
        img_buffer.seek(0)
        st.download_button(label=button_text, data=img_buffer, file_name=filename, mime="image/png")

    with st.expander("Save Results"):
        if std_dev_image is not None and speckle_contrast_image is not None:
            create_image_download_button(std_dev_image, "std_dev_filter.png", "Download Std Dev Filter")
            create_image_download_button(speckle_contrast_image, "speckle_contrast.png", "Download Speckle Contrast Image")
            create_image_download_button(mean_image, "mean_filter.png", "Download Mean Filter")
        else:
            st.error("No results to save. Please generate images by running the analysis.")


#----------------------------- Comparison Plotting Stuff ------------------------------ #
def display_image_comparison(img1, img2, label1, label2, cmap):
    img1_uint8, img2_uint8 = apply_colormap_to_images(img1, img2, cmap)
    image_comparison(img1=img1_uint8, img2=img2_uint8, label1=label1, label2=label2, make_responsive=True)
    st.subheader("Selected Images")
    st.image([img1_uint8, img2_uint8], caption=[label1, label2])

def get_images_to_compare(image_choice_1, image_choice_2, std_dev_image, speckle_contrast_image, mean_image, original_image):
    images_to_compare = {
        'Unprocessed Image': original_image,  # Include unprocessed image
        'Standard Deviation': std_dev_image,
        'Speckle Contrast': speckle_contrast_image,
        'Mean Filter': mean_image
    }
    return images_to_compare[image_choice_1], images_to_compare[image_choice_2]

def handle_comparison_tab(tab, cmap_name, std_dev_image, speckle_contrast_image, mean_image, original_image):
    with tab:
        st.header("Speckle Contrast Comparison")
        available_images = ['Unprocessed Image', 'Standard Deviation', 'Speckle Contrast', 'Mean Filter']  # Include unprocessed image
        col1, col2 = st.columns(2)
        with col1:
            image_choice_1 = st.selectbox('Select first image to compare:', [''] + available_images, index=0)
        with col2:
            image_choice_2 = st.selectbox('Select second image to compare:', [''] + available_images, index=0)
        # Convert cmap_name to the actual colormap function
        cmap = plt.get_cmap(cmap_name)

        if image_choice_1 and image_choice_2 and image_choice_1 != image_choice_2:
            img1, img2 = get_images_to_compare(image_choice_1, image_choice_2, std_dev_image, speckle_contrast_image, mean_image, original_image)
            display_image_comparison(img1, img2, image_choice_1, image_choice_2, cmap)
        else:
            if image_choice_1 == image_choice_2 and image_choice_1:
                st.error("Please select two different images for comparison.")
            else:
                st.info("Select two images to compare.")
