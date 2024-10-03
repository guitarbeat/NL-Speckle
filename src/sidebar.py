import numpy as np
import streamlit as st
from PIL import Image
from typing import Optional
import uuid

import src.session_state as session_state
from src.session_state import PRELOADED_IMAGE_PATHS, AVAILABLE_COLOR_MAPS

RNG = np.random.default_rng(seed=42)

def setup_ui() -> Optional[bool]:
    apply_custom_css()
    try:
        st.logo("media/logo.png")
        with st.sidebar:
            st.title("Image Processing App")
            create_sidebar_sections()
        return True
    except Exception as e:
        st.error(f"Error setting up UI: {str(e)}")
        return None

def create_sidebar_sections():
    sections = [
        ("🖼️ Image Selection", image_selection, True),
        ("⚙️ Advanced Options", advanced_options, False),
    ]
    for title, func, expanded in sections:
        with st.expander(title, expanded=expanded):
            func()

def apply_custom_css():
    st.markdown(
        """
        <style>
            [data-testid="stSidebar"] {min-width: 300px;}
        </style>
        """,
        unsafe_allow_html=True,
    )

def image_selection() -> bool:
    image_source = st.radio("Select Image Source", ("Preloaded Images", "Upload Image"), key=f"image_source_radio_{uuid.uuid4()}")
    
    if image_source == "Preloaded Images":
        selected_image = st.selectbox("Select Image", list(PRELOADED_IMAGE_PATHS.keys()))
        image = load_image(PRELOADED_IMAGE_PATHS[selected_image])
        return process_loaded_image(image, selected_image)
    else:
        uploaded_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg", "tif", "tiff"])
        if uploaded_file is None:
            st.warning("Please upload an image.")
            return False
        image = load_image(uploaded_file)
        return process_loaded_image(image, uploaded_file.name)

def load_image(image_path: str) -> Optional[Image.Image]:
    try:
        return Image.open(image_path).convert("L")
    except (FileNotFoundError, IOError) as e:
        st.error(f"Error loading image: {e}. Please try again.")
        return None

def process_loaded_image(image: Optional[Image.Image], image_name: str) -> bool:
    if image is None:
        st.warning("Failed to load image. Please try again.")
        return False
    st.success("Image loaded successfully!")
    st.image(image, caption="Input Image", use_column_width=True)
    session_state.set_session_state("image", image)
    session_state.set_session_state("image_array", np.array(image))
    session_state.set_session_state("image_file", image_name)
    select_color_map()
    return True

def select_color_map():
    current_color_map = session_state.get_color_map()
    if current_color_map not in AVAILABLE_COLOR_MAPS:
        current_color_map = AVAILABLE_COLOR_MAPS[0]
    color_map = st.selectbox(
        "Select Color Map",
        AVAILABLE_COLOR_MAPS,
        index=AVAILABLE_COLOR_MAPS.index(current_color_map),
        key=f"color_map_select_{uuid.uuid4()}",
        help="Choose a color map to apply to the image",
    )
    session_state.set_session_state("color_map", color_map)

def advanced_options():
    normalization_and_noise_options()
    process_image()

def normalization_and_noise_options():
    tab1, tab2 = st.tabs(["Normalization", "Noise"])
    with tab1:
        normalization_option = st.selectbox(
            "Normalization",
            options=["None", "Percentile"],
            index=["None", "Percentile"].index(session_state.get_normalization_options().get("option", "None")),
            help="Choose the normalization method for the image",
        )
        session_state.set_session_state("normalization", {"option": normalization_option})
    
    with tab2:
        apply_gaussian_noise = st.checkbox(
            "Add Gaussian Noise",
            value=session_state.get_gaussian_noise_settings().get("enabled", False),
            help="Add Gaussian noise to the image",
        )
        session_state.set_session_state("gaussian_noise", {"enabled": apply_gaussian_noise})
        if apply_gaussian_noise:
            noise_mean = st.number_input("Noise Mean", min_value=0.0, max_value=1.0, value=session_state.get_gaussian_noise_settings().get("mean", 0.1), step=0.01, format="%.2f", help="Mean value for Gaussian noise")
            noise_std_dev = st.number_input("Noise Standard Deviation", min_value=0.0, max_value=1.0, value=session_state.get_gaussian_noise_settings().get("std_dev", 0.1), step=0.01, format="%.2f", help="Standard deviation for Gaussian noise")
            session_state.set_session_state("gaussian_noise", {"enabled": True, "mean": noise_mean, "std_dev": noise_std_dev})

def process_image():
    image_array = session_state.get_image_array()
    if image_array is None or image_array.size == 0:
        st.warning("No image loaded. Please select an image first.")
        return
    try:
        image_np = image_array.astype(np.float32) / 255.0
        if session_state.get_gaussian_noise_settings()["enabled"]:
            image_np = apply_gaussian_noise(image_np)
        if session_state.get_normalization_options()["option"] == "Percentile":
            image_np = apply_percentile_normalization(image_np)
        session_state.set_session_state("processed_image_np", image_np)
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        st.exception(e)

def apply_gaussian_noise(image_np: np.ndarray) -> np.ndarray:
    gaussian_noise = session_state.get_gaussian_noise_settings()
    noise = RNG.normal(
        gaussian_noise.get("mean", 0),
        gaussian_noise.get("std_dev", 0.1),
        image_np.shape,
    )
    return np.clip(image_np + noise, 0, 1)

def apply_percentile_normalization(image_np: np.ndarray) -> np.ndarray:
    normalization = session_state.get_normalization_options()
    p_low, p_high = np.percentile(
        image_np,
        [normalization.get("percentile_low", 2), normalization.get("percentile_high", 98)],
    )
    image_np = np.clip(image_np, p_low, p_high)
    return (image_np - p_low) / (p_high - p_low)
