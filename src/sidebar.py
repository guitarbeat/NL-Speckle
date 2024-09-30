###############################################################################
# sidebar.py
###############################################################################

"""
This module defines functions for the sidebar UI in the Streamlit application.
It includes available color maps and preloaded image paths.
"""

# Imports
import numpy as np
import streamlit as st
from PIL import Image
from typing import Optional
from session_state import initialize_session_state, update_pixel_processing_values, calculate_total_pixels

# Constants
rng = np.random.default_rng(seed=42)

###############################################################################
# Main UI Setup
###############################################################################

def setup_ui() -> Optional[bool]:
    initialize_session_state()
    apply_custom_css()

    if image_selection():
        update_pixel_processing_values()  # Call this only if an image is loaded
        display_options()
        nlm_options()
        advanced_options()
        return True
    else:
        return None

def apply_custom_css():
    st.sidebar.markdown("""
        <style>
            [data-testid="stSidebar"] {
                min-width: 300px;
            }
        </style>
    """, unsafe_allow_html=True)
###############################################################################
# Image Selection Methods
###############################################################################


def image_selection() -> bool:
    with st.sidebar.expander("🖼️ Image Selector", expanded=True):
        image_source = st.radio("Select Image Source", ("Preloaded Images", "Upload Image"))
        return load_preloaded_image() if image_source == "Preloaded Images" else load_uploaded_image()

def load_preloaded_image() -> bool:
    from main import PRELOADED_IMAGE_PATHS
    selected_image = st.selectbox("Select Image", list(PRELOADED_IMAGE_PATHS.keys()))
    try:
        image = Image.open(PRELOADED_IMAGE_PATHS[selected_image]).convert("L")
        st.session_state.image_file = selected_image
        return process_loaded_image(image)
    except (FileNotFoundError, IOError) as e:
        st.error(f"Error loading image: {e}. Please try again.")
        return False

def load_uploaded_image() -> bool:
    uploaded_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg", "tif", "tiff"])
    if uploaded_file is None:
        st.warning("Please upload an image.")
        return False
    try:
        image = Image.open(uploaded_file).convert("L")
        st.session_state.image_file = uploaded_file.name
        return process_loaded_image(image)
    except IOError as e:
        st.error(f"Error loading image: {e}. Please try again.")
        return False

def process_loaded_image(image: Image.Image) -> bool:
    st.image(image, caption="Input Image", use_column_width=True)
    st.session_state.image = image
    st.session_state.image_array = np.array(image)
    select_color_map()
    update_pixel_processing_values()
    return True


###############################################################################
# Display Options Methods
###############################################################################


def display_options():
    with st.sidebar.expander("🔧 Display Options", expanded=True):
        kernel_size_and_per_pixel_toggle()
        total_pixels = calculate_total_pixels()
        handle_pixel_processing(total_pixels)

def kernel_size_and_per_pixel_toggle():
    col1, col2 = st.columns(2)
    with col1:
        new_kernel_size = st.slider(
            "Kernel Size", min_value=3, max_value=21, 
            value=st.session_state.kernel_size, step=2, key="kernel_size_slider"
        )
        if new_kernel_size != st.session_state.kernel_size:
            st.session_state.kernel_size = new_kernel_size
            update_pixel_processing_values()
    with col2:
        st.toggle(
            "Show Per-Pixel Processing Steps", 
            value=st.session_state.show_per_pixel,
            key="show_per_pixel",
            on_change=lambda: setattr(st.session_state, 'show_per_pixel', not st.session_state.show_per_pixel)
        )

def handle_pixel_processing(total_pixels: int):
    if st.session_state.show_per_pixel:
        setup_pixel_processing(total_pixels)
    else:
        st.session_state.pixels_to_process = total_pixels

def setup_pixel_processing(total_pixels: int):
    col1, col2 = st.columns(2)
    with col1:
        st.session_state.desired_percentage = st.slider(
            "Percentage", min_value=1, max_value=100,
            value=st.session_state.desired_percentage, key="percentage_slider",
        )

    st.session_state.desired_exact_count = int(total_pixels * st.session_state.desired_percentage / 100)

    with col2:
        st.session_state.desired_exact_count = st.number_input(
            "Exact Pixels", min_value=1, max_value=total_pixels,
            value=st.session_state.desired_exact_count, key="exact_pixel_count",
        )

    st.session_state.desired_percentage = int((st.session_state.desired_exact_count / total_pixels) * 100)
    st.session_state.pixels_to_process = st.session_state.desired_exact_count

    st.write(f"Processing {st.session_state.desired_percentage}% ({st.session_state.desired_exact_count} of {total_pixels} pixels)")

# Display options methods
def select_color_map():
    from main import AVAILABLE_COLOR_MAPS
    st.session_state.color_map = st.selectbox(
        "Select Color Map", AVAILABLE_COLOR_MAPS,
        index=AVAILABLE_COLOR_MAPS.index(st.session_state.color_map),
        key="color_map_select",
    )

###############################################################################
# Advanced Options Methods
###############################################################################

def advanced_options():
    with st.sidebar.expander("🔬 Advanced Options", expanded=False):
        normalization_and_noise_options()
        sat_options()
        speckle_filter_selection()  # Add this line
    process_image()

def speckle_filter_selection():
    st.session_state.filters["speckle"] = st.multiselect(
        "Speckle Filter Selection",
        options=["Original Image", "Speckle Contrast"],
        default=st.session_state.filters["speckle"],
        key="speckle_filter_selection"
    )

def normalization_and_noise_options():
    tab1, tab2 = st.tabs(["Normalization", "Noise"])
    with tab1:
        st.session_state.normalization["option"] = st.selectbox(
            "Normalization", options=["None", "Percentile"],
            index=["None", "Percentile"].index(st.session_state.normalization["option"]),
            help="Choose the normalization method for the image",
        )
    with tab2:
        st.session_state.apply_gaussian_noise = st.checkbox(
            "Add Gaussian Noise", value=st.session_state.gaussian_noise["enabled"], 
            help="Add Gaussian noise to the image"
        )
        if st.session_state.gaussian_noise["enabled"]:
            setup_gaussian_noise_params()

def sat_options():
    with st.popover("SAT Options"):
        st.session_state.use_sat = st.toggle(
            "Use Summed-Area Tables (SAT)", value=st.session_state.use_sat,
            help="Enable Summed-Area Tables for faster Speckle Contrast calculation"
        )

def setup_gaussian_noise_params():
    st.session_state.noise_mean = st.sidebar.number_input(
        "Noise Mean", min_value=0.0, max_value=1.0,
        value=st.session_state.noise_mean, step=0.01, format="%.2f",
    )
    st.session_state.noise_std_dev = st.sidebar.number_input(
        "Noise Standard Deviation", min_value=0.0, max_value=1.0,
        value=st.session_state.noise_std_dev, step=0.01, format="%.2f",
    )

def process_image():
    image_np = st.session_state.image_array.astype(np.float32) / 255.0
    if st.session_state.gaussian_noise["enabled"]:
        image_np = apply_gaussian_noise(image_np)
    if st.session_state.normalization["option"] == "Percentile":
        image_np = apply_percentile_normalization(image_np)
    st.session_state.processed_image_np = image_np
    

def apply_gaussian_noise(image_np: np.ndarray) -> np.ndarray:
    noise = rng.normal(
        st.session_state.gaussian_noise["mean"],
        st.session_state.gaussian_noise["std_dev"],
        image_np.shape
    )
    return np.clip(image_np + noise, 0, 1)

def apply_percentile_normalization(image_np: np.ndarray) -> np.ndarray:
    p_low, p_high = np.percentile(
        image_np,
        [st.session_state.normalization["percentile_low"],
         st.session_state.normalization["percentile_high"]]
    )
    image_np = np.clip(image_np, p_low, p_high)
    return (image_np - p_low) / (p_high - p_low)

###############################################################################
# NLM Options Methods
###############################################################################

def nlm_options():
    with st.sidebar.expander("🔍 NLM Options", expanded=False):
        setup_search_window()
        setup_filter_strength()
        setup_nlm_filter_selection()  # Add this line

def setup_nlm_filter_selection():
    st.multiselect(
        "NLM Filter Selection",
        options=["Original Image", "Non-Local Means"],
        default=st.session_state.filters["nlm"],
        key="nlm_filter_selection"
    )

def setup_search_window():
    image_shape = st.session_state.image.size
    max_search_window = min(101, *image_shape)
    default_search_window = min(21, max_search_window)
    st.session_state.nlm_options["use_whole_image"] = st.checkbox(
        "Use whole image as search window", 
        value=st.session_state.nlm_options["use_whole_image"]  # Use the value from Session State
    )
    st.session_state.search_window_size = get_search_window_size(
        st.session_state.nlm_options["use_whole_image"],  # Use the value from Session State
        max_search_window, 
        default_search_window, 
        image_shape
    )

def setup_filter_strength():
    st.session_state.filter_strength = st.slider(
        "Filter Strength", min_value=0.1, max_value=100.0,
        value=st.session_state.nlm_options["filter_strength"], step=0.1, format="%.1f",
        help="Filter strength for NLM (higher values mean more smoothing)",
    )

def get_search_window_size(use_whole_image: bool, max_search_window: int, default_search_window: int, image_shape: tuple) -> int:
    if use_whole_image:
        return max(image_shape)
    
    search_window_size = st.slider(
        "Search Window Size", min_value=3, max_value=max_search_window,
        value=default_search_window, step=2,
        help="Size of the search window for NLM (must be odd)",
    )
    return search_window_size if search_window_size % 2 == 1 else search_window_size + 1

def extract_patch(image: np.ndarray, y: int, x: int, patch_size: int) -> np.ndarray:
    half_patch = patch_size // 2
    return image[
        y - half_patch : y + half_patch + 1, x - half_patch : x + half_patch + 1
    ]