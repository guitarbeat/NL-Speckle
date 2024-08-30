import streamlit as st
from ui_components import handle_animation_controls
from PIL import Image
import numpy as np
import io

# media/image50.png
# Define preloaded images
PRELOADED_IMAGES = {
    "image50.png": "media/image50.png",
    "spatial.tif": "media/spatial.tif",
    "logo.jpg": "media/logo.jpg",
}

# Load the Streamlit logo
image = "media/logo.png"
def set_page_config():
    """Set page configuration for the Streamlit app."""
    st.set_page_config(
        page_title="Speckle Contrast Visualization",
        layout="wide",
        page_icon="favicon.png",
        initial_sidebar_state="expanded",
    )
    st.logo(image)
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
                ["viridis", "plasma", "inferno", "magma", "cividis", "gray"],
                index=0,
                help="Choose a color map for all visualizations."
            )

        st.markdown("### 🛠️ Processing Parameters")
        with st.expander("Image Processing", expanded=True):
            kernel_size = st.slider('Kernel Size', 1, 10, 3, help="Size of the sliding window (patch size).")
            stride = st.slider('Stride', 1, 5, 1, help="Stride of the sliding window.")
            with st.expander("🔍 Non-Local Means Parameters", expanded=False):
                # Option to use entire image as the search window
                use_full_image_window = st.checkbox("Use Full Image as Search Window", value=False, help="If checked, the search window will cover the entire image.")
                
                # Conditionally display search window size slider based on checkbox state
                if not use_full_image_window:
                    search_window_size = st.slider(
                        "Search Window Size", 
                        min_value=1, 
                        max_value=21, 
                        value=5, 
                        step=2,
                        help="Size of the window used to search for similar patches in the image."
                    )
                else:
                    search_window_size = None  # Use None to indicate full image search window
                
                filter_strength = st.slider(
                    "Filter Strength (h)", 
                    min_value=0.1, 
                    max_value=50.0, 
                    value=10.0, 
                    step=0.1,
                    help="Controls the decay of the similarity function, affecting how strongly similar patches are weighted."
                )

        st.markdown("### 🎞️ Animation Controls")
        with st.expander("Animation Settings", expanded=True):
            animation_speed = st.slider("Animation Speed (seconds per frame)", 0.001, 0.5, 0.01, 0.01)
            
            # Handle animation controls
            handle_animation_controls()

    return image, kernel_size, search_window_size, filter_strength, stride, cmap, animation_speed, image_np



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


def get_image_download_link(image, filename, button_text="Download Image"):
    """Convert a numpy array to an image and create a download button."""
    img_buffer = io.BytesIO()
    Image.fromarray((255 * image).astype(np.uint8)).save(img_buffer, format='PNG')
    img_buffer.seek(0)
    return st.download_button(label=button_text, data=img_buffer, file_name=filename, mime="image/png")
