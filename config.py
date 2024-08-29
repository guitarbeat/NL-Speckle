import streamlit as st

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

# Add any other configuration variables or constants here