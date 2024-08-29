import numpy as np
from PIL import Image
import streamlit as st
from config import PRELOADED_IMAGES
import io

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
