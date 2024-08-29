import streamlit as st
import numpy as np
from PIL import Image
from helpers import toggle_animation, handle_speckle_contrast_calculation
from streamlit_image_comparison import image_comparison
import cv2
import io

# Set page configuration
st.set_page_config(
    page_title="Speckle Contrast Visualization",
    layout="wide",
    page_icon="favicon.png"
)

st.title("Interactive Speckle Contrast Analysis")

# Introduction and instructions
st.markdown("""
Welcome to the Interactive Speckle Contrast Analysis tool! This application allows you to visualize and analyze speckle contrast in images.

**How to use this tool:**
1. Choose an image source (preloaded or upload your own) in the sidebar.
2. Adjust the processing parameters and visualization settings.
3. Explore the speckle contrast analysis results in the tabs below.
4. Use the animation controls to see the analysis process in action.

Let's get started!
""")

# Define preloaded images
PRELOADED_IMAGES = {
    "image50.png": "image50.png",
    "spatial.tif": "spatial.tif",
    "logo.jpg": "logo.jpg",
}

def load_image(image_source: str, selected_image: str, uploaded_file) -> Image.Image:
    """Load the image based on the user's selection."""
    if image_source == "Preloaded Image":
        return Image.open(PRELOADED_IMAGES[selected_image]).convert('L')
    elif uploaded_file:
        return Image.open(uploaded_file).convert('L')
    else:
        st.warning('Please upload an image or select a preloaded image.')
        st.stop()

def configure_sidebar() -> tuple:
    """Configure sidebar options and return user inputs."""
    with st.sidebar:
        with st.expander("Image Configuration", expanded=True):
            image_source = st.radio("Choose Image Source", ["Preloaded Image", "Upload Image"])
            selected_image = st.selectbox("Select Preloaded Image", list(PRELOADED_IMAGES.keys())) if image_source == "Preloaded Image" else None
            uploaded_file = st.file_uploader("Upload an Image") if image_source == "Upload Image" else None

            image = load_image(image_source, selected_image, uploaded_file)
            st.image(image, caption="Input Image", use_column_width=True)

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
            start_button = st.button('Start Animation')
            stop_button = st.button('Stop Animation')


    return image, kernel_size, stride, cmap, animation_speed, start_button, stop_button

@st.cache_data
def process_image(image: Image.Image) -> np.ndarray:
    """Convert the image to a numpy array and normalize pixel values."""
    return np.array(image) / 255.0

def initialize_session_state():
    """Initialize session state variables."""
    st.session_state.setdefault('is_animating', False)
    st.session_state.setdefault('cache', {})
    st.session_state.setdefault('animation_mode', 'Standard')

def handle_animation_controls(start_button: bool, stop_button: bool):
    """Handle the start/stop animation buttons."""
    if start_button or stop_button:
        toggle_animation()

def get_image_download_link(image, filename, button_text="Download Image"):
    """Convert a numpy array to an image and create a download button."""
    img_buffer = io.BytesIO()
    Image.fromarray((255 * image).astype(np.uint8)).save(img_buffer, format='PNG')
    img_buffer.seek(0)
    return st.download_button(label=button_text, data=img_buffer, file_name=filename, mime="image/png")


def main():
    # Configure sidebar and get user inputs
    image, kernel_size, stride, cmap, animation_speed, start_button, stop_button = configure_sidebar()

    # Process image
    image_np = process_image(image)

    # Initialize session state
    initialize_session_state()

    # Handle animation controls
    handle_animation_controls(start_button, stop_button)

    # Pixels to process slider (moved out of sidebar)
    max_pixels = st.slider("Pixels to process", 1, image.width * image.height, image.width * image.height)

    # Create tabs for different calculation methods
    tab1, tab2, tab3 = st.tabs(["Speckle Contrast Method", "Non-Local Means Method", "Speckle Contrast Comparison"])

    with tab1:
        st.header("Speckle Contrast Calculation")
        st.session_state.animation_mode = 'Standard'

        # Display the speckle contrast formula at the top of tab 1
        formula_placeholder = st.empty()
        with formula_placeholder.container():
            st.latex(
                r'SC_{{({}, {})}} = \frac{{\sigma}}{{\mu}} = \frac{{{:.3f}}}{{{:.3f}}} = {:.3f}'.format(0, 0, 0.0, 0.0, 0.0)
            )

        # Create two columns for side-by-side layout
        col1, col2 = st.columns(2)

        # Place expanders inside each column
        with col1:
            with st.expander("Original Image with Current Kernel", expanded=True):
                original_image_placeholder = st.empty()
                with st.popover("Zoomed-in Kernel"):
                    zoomed_kernel_placeholder = st.empty()

            with st.expander("Mean Filter", expanded=False):
                mean_filter_placeholder = st.empty()
                with st.popover("Zoomed-in Mean"):
                    zoomed_mean_placeholder = st.empty()

        with col2:
            with st.expander("Speckle Contrast", expanded=True):
                speckle_contrast_placeholder = st.empty()
                with st.popover("Zoomed-in Speckle Contrast"):
                    zoomed_sc_placeholder = st.empty()

            with st.expander("Standard Deviation Filter", expanded=False):
                std_dev_filter_placeholder = st.empty()
                with st.popover("Zoomed-in Std Dev"):
                    zoomed_std_placeholder = st.empty()

        # Container for the plots and visualizations
        with st.container():
            # Perform the calculation and get final images
            std_dev_image, speckle_contrast_image, mean_image = handle_speckle_contrast_calculation(
                max_pixels, image_np, kernel_size, stride, 
                original_image_placeholder, mean_filter_placeholder, 
                std_dev_filter_placeholder, speckle_contrast_placeholder, 
                zoomed_kernel_placeholder, zoomed_mean_placeholder, 
                zoomed_std_placeholder, zoomed_sc_placeholder, 
                formula_placeholder, animation_speed, cmap
            )

    
        # Save Results Section with Download Button
        with st.expander("Save Results"):
            if std_dev_image is not None and speckle_contrast_image is not None:
                get_image_download_link(std_dev_image, "std_dev_filter.png", "Download Std Dev Filter")
                get_image_download_link(speckle_contrast_image, "speckle_contrast.png", "Download Speckle Contrast Image")
                get_image_download_link(mean_image, "mean_filter.png", "Download Mean Filter")
            else:
                st.error("No results to save. Please generate images by running the analysis.")

    with tab2:
        st.header("Non-Local Means Method")
        st.write("Coming soon...")

    with tab3:
        st.header("Speckle Contrast Comparison")

        # Ensure images are loaded before conversion
        if std_dev_image is not None and speckle_contrast_image is not None:
            # Convert images to uint8 and then to BGR for proper color handling in image comparison
            std_dev_image_uint8 = (255 * std_dev_image).astype(np.uint8)
            speckle_contrast_image_uint8 = (255 * speckle_contrast_image).astype(np.uint8)
            std_dev_image_bgr = cv2.cvtColor(std_dev_image_uint8, cv2.COLOR_GRAY2BGR)
            speckle_contrast_image_bgr = cv2.cvtColor(speckle_contrast_image_uint8, cv2.COLOR_GRAY2BGR)

            image_comparison(
                img1=std_dev_image_bgr,
                img2=speckle_contrast_image_bgr,
                label1='Standard Deviation',
                label2='Speckle Contrast',
                make_responsive=True,
                width=700,
            )
        else:
            st.write("Please generate images by running the analysis.")

if __name__ == "__main__":
    main()
