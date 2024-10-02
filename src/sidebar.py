import numpy as np
import streamlit as st
from PIL import Image
from typing import Optional
import uuid

import src.session_state as session_state

RNG = np.random.default_rng(seed=42)

# Color Maps
AVAILABLE_COLOR_MAPS = [
    "viridis_r",
    "viridis",
    "gray",
    "plasma",
    "inferno",
    "magma",
    "pink",
    "hot",
    "cool",
    "YlOrRd",
]

# Preloaded Images
PRELOADED_IMAGE_PATHS = {
    "image50.png": "media/image50.png",
    "spatial.tif": "media/spatial.tif",
    "logo.jpg": "media/logo.jpg",
}
def setup_ui() -> Optional[bool]:
    apply_custom_css()
    try:
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
        ("🔧 Display Options", lambda: display_options("speckle"), True),
        ("🔍 NLM Options", nlm_options, False),
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
    image_source = st.radio(
        "Select Image Source",
        ("Preloaded Images", "Upload Image"),
        key=f"image_source_radio_{uuid.uuid4()}",
    )
    return load_preloaded_image() if image_source == "Preloaded Images" else load_uploaded_image()

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

def load_preloaded_image() -> bool:
    selected_image = st.selectbox("Select Image", list(PRELOADED_IMAGE_PATHS.keys()))
    image = load_image(PRELOADED_IMAGE_PATHS[selected_image])
    return process_loaded_image(image, selected_image)

def load_uploaded_image() -> bool:
    uploaded_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg", "tif", "tiff"])
    if uploaded_file is None:
        st.warning("Please upload an image.")
        return False
    image = load_image(uploaded_file)
    return process_loaded_image(image, uploaded_file.name)

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

def display_options(technique: str):
    kernel_size_and_per_pixel_toggle(technique)

def kernel_size_and_per_pixel_toggle(technique: str):
    image_array = session_state.get_image_array()
    if image_array is None:
        st.warning("Please load an image first.")
        return

    height, width = image_array.shape
    
    new_kernel_size = st.slider(
        "Kernel Size",
        min_value=3, max_value=min(height, width), value=session_state.kernel_size(), step=2,
        key=f"kernel_size_slider_{technique}",
        help="Size of the kernel used for processing",
    )
    if new_kernel_size != session_state.kernel_size():
        session_state.kernel_size(new_kernel_size)

    half_kernel = new_kernel_size // 2
    processable_height = height - 2 * half_kernel
    processable_width = width - 2 * half_kernel
    total_processable_pixels = processable_height * processable_width

    show_per_pixel = st.checkbox(
        "Show Per-Pixel Processing",
        value=session_state.get_session_state("show_per_pixel", False),
        key=f"show_per_pixel_{technique}",
        help="Enable to see per-pixel processing details",
    )
    session_state.set_show_per_pixel_processing(show_per_pixel)

    if show_per_pixel:
        max_pixels = st.number_input(
            "Maximum Pixels to Process",
            min_value=1,
            max_value=total_processable_pixels,
            value=min(10000, total_processable_pixels),
            step=1000,
            key=f"max_pixels_{technique}",
            help="Set the maximum number of pixels to process for per-pixel visualization",
        )
        st.info(f"Processing {max_pixels:,} out of {total_processable_pixels:,} processable pixels ({max_pixels/total_processable_pixels:.2%})")
    else:
        max_pixels = total_processable_pixels
        st.info(f"Processing all {total_processable_pixels:,} processable pixels")

    session_state.set_session_state("pixels_to_process", max_pixels)
    
    # Update the processable area in the session state
    processable_area = {
        "top": half_kernel,
        "bottom": height - half_kernel,
        "left": half_kernel,
        "right": width - half_kernel,
    }
    session_state.set_session_state("processable_area", processable_area)

def advanced_options():
    normalization_and_noise_options()
    process_image()

def normalization_and_noise_options():
    tab1, tab2 = st.tabs(["Normalization", "Noise"])
    with tab1:
        handle_normalization_options()
    with tab2:
        handle_noise_options()

def handle_normalization_options():
    normalization_option = st.selectbox(
        "Normalization",
        options=["None", "Percentile"],
        index=["None", "Percentile"].index(
            session_state.get_normalization_options().get("option", "None")
        ),
        help="Choose the normalization method for the image",
    )
    session_state.set_session_state("normalization", {"option": normalization_option})

def handle_noise_options():
    apply_gaussian_noise = st.checkbox(
        "Add Gaussian Noise",
        value=session_state.get_gaussian_noise_settings().get("enabled", False),
        help="Add Gaussian noise to the image",
    )
    session_state.set_session_state("gaussian_noise", {"enabled": apply_gaussian_noise})
    if apply_gaussian_noise:
        setup_gaussian_noise_params()

def setup_gaussian_noise_params():
    noise_mean = st.number_input(
        "Noise Mean",
        min_value=0.0, max_value=1.0,
        value=session_state.get_gaussian_noise_settings().get("mean", 0.1),
        step=0.01, format="%.2f",
        help="Mean value for Gaussian noise",
    )
    noise_std_dev = st.number_input(
        "Noise Standard Deviation",
        min_value=0.0, max_value=1.0,
        value=session_state.get_gaussian_noise_settings().get("std_dev", 0.1),
        step=0.01, format="%.2f",
        help="Standard deviation for Gaussian noise",
    )
    session_state.set_session_state("gaussian_noise", {
        "enabled": True,
        "mean": noise_mean,
        "std_dev": noise_std_dev
    })

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

def nlm_options():
    nlm_opts = session_state.get_nlm_options()
    use_whole_image = st.checkbox(
        "Use whole image as search window",
        value=nlm_opts["use_whole_image"],
        help="Use the entire image as the search window for NLM",
    )
    
    image_array = session_state.get_image_array()
    max_search_window, min_search_window = get_search_window_limits(image_array)
    
    search_window_size = get_search_window_size(use_whole_image, min_search_window, max_search_window, nlm_opts, image_array)

    filter_strength = st.slider(
        "Filter Strength",
        min_value=0.1, max_value=100.0, value=nlm_opts["filter_strength"],
        step=0.1, format="%.1f",
        help="Filter strength for NLM (higher values mean more smoothing)",
    )

    session_state.update_nlm_params(filter_strength, search_window_size, use_whole_image)

def get_search_window_limits(image_array):
    if image_array is not None:
        image_shape = image_array.shape
        max_search_window = min(101, *image_shape)
        min_search_window = min(21, max_search_window)
    else:
        max_search_window, min_search_window = 101, 21
    return max_search_window, min_search_window

def get_search_window_size(use_whole_image, min_search_window, max_search_window, nlm_opts, image_array):
    if not use_whole_image:
        search_window_size = st.slider(
            "Search Window Size",
            min_value=min_search_window, max_value=max_search_window,
            value=nlm_opts["search_window_size"], step=2,
            help="Size of the search window for NLM (must be odd)",
        )
        return search_window_size if search_window_size % 2 == 1 else search_window_size + 1
    else:
        return min(max(image_array.shape), 101) if image_array is not None else 101