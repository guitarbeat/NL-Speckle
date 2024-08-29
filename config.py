import streamlit as st
from ui_components import handle_animation_controls
from image_processing import load_image, process_image

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

# Add any other configuration variables or constants here