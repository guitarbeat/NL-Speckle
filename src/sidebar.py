###############################################################################
# Imports and Constants
###############################################################################

import numpy as np
import streamlit as st
from PIL import Image
from typing import Optional
from src.session_state import (
    update_pixel_processing_values, 
    calculate_total_pixels, 
    set_value,
    get_value,
    update_nested_session_state, 

)
import uuid

RNG = np.random.default_rng(seed=42)

###############################################################################
# Main UI Setup
###############################################################################

def setup_ui() -> Optional[bool]:
    apply_custom_css()

    try:
        with st.sidebar:
            st.title("Image Processing App")
            
            with st.expander("🖼️ Image Selection", expanded=True):
                if image_selection():
                    update_pixel_processing_values()
            
            with st.expander("🔧 Display Options", expanded=True):
                display_options()
            
            
            with st.expander("🔍 NLM Options", expanded=False):
                nlm_options()
            
            with st.expander("⚙️ Advanced Options", expanded=False):
                advanced_options()
        
        return True
    except Exception as e:
        st.error(f"Error setting up UI: {str(e)}")
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
        image_source = st.radio("Select Image Source", ("Preloaded Images", "Upload Image"), key=f"image_source_radio_{uuid.uuid4()}")
        return load_preloaded_image() if image_source == "Preloaded Images" else load_uploaded_image()

def load_image(image_path: str) -> Image.Image:
    try:
        return Image.open(image_path).convert("L")
    except (FileNotFoundError, IOError) as e:
        st.error(f"Error loading image: {e}. Please try again.")
        return None

def process_loaded_image(image: Image.Image, image_name: str) -> bool:
    if image is None:
        st.warning("Failed to load image. Please try again.")
        return False
    
    st.success("Image loaded successfully!")
    st.image(image, caption="Input Image", use_column_width=True)
    set_value('image', image)
    set_value('image_array', np.array(image))
    set_value('image_file', image_name)
    select_color_map()
    update_pixel_processing_values()
    return True

def load_preloaded_image() -> bool:
    from main import PRELOADED_IMAGE_PATHS
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
    from main import AVAILABLE_COLOR_MAPS
    color_map = st.selectbox(
        "Select Color Map", AVAILABLE_COLOR_MAPS,
        index=AVAILABLE_COLOR_MAPS.index(get_value('color_map')),
        key="color_map_select",
        help="Choose a color map to apply to the image"
    )
    set_value('color_map', color_map)

###############################################################################
# Display Options Methods
###############################################################################

def display_options():
    with st.sidebar.expander("🔧 Display Options", expanded=True):
        kernel_size_and_per_pixel_toggle()
        total_pixels = calculate_total_pixels()
        handle_pixel_processing(total_pixels)

def kernel_size_and_per_pixel_toggle():
    new_kernel_size = st.slider(
        "Kernel Size", min_value=3, max_value=21, 
        value=get_value('kernel_size', 3), step=2, key="kernel_size_slider",
        help="Size of the kernel used for processing"
    )
    if new_kernel_size != get_value('kernel_size'):
        set_value('kernel_size', new_kernel_size)
        update_pixel_processing_values()

    show_per_pixel = st.toggle(
        "Show Per-Pixel Processing Steps", 
        value=get_value('show_per_pixel', False),
        key="show_per_pixel",
        help="Enable to view processing steps for individual pixels"
    )
    set_value('show_per_pixel', show_per_pixel)

def handle_pixel_processing(total_pixels: int):
    if get_value('show_per_pixel', False):
        setup_pixel_processing(total_pixels)
    else:
        set_value('pixels_to_process', total_pixels)

def setup_pixel_processing(total_pixels: int):
    st.write("Per-Pixel Processing Options:")

    desired_percentage = st.slider(
        "Percentage of Pixels to Process", 
        min_value=1, 
        max_value=100,
        value=get_value('desired_percentage', 100), 
        key="percentage_slider",
        help="Adjust the percentage of pixels to process"
    )
    set_value('desired_percentage', desired_percentage)

    desired_exact_count = int(total_pixels * desired_percentage / 100)
    set_value('desired_exact_count', desired_exact_count)

    exact_count = st.number_input(
        "Exact Number of Pixels to Process", 
        min_value=1, 
        max_value=total_pixels,
        value=desired_exact_count, 
        key="exact_pixel_count",
        help="Specify the exact number of pixels to process"
    )
    set_value('desired_exact_count', exact_count)

    # Update percentage based on exact count
    updated_percentage = int((exact_count / total_pixels) * 100)
    set_value('desired_percentage', updated_percentage)
    set_value('pixels_to_process', exact_count)

    st.info(f"Processing {updated_percentage}% ({exact_count} out of {total_pixels} total pixels)")

###############################################################################
# Filter Selection Methods
###############################################################################

###############################################################################
# Advanced Options Methods
###############################################################################

def advanced_options():
    with st.sidebar.expander("🔬 Advanced Options", expanded=False):
        normalization_and_noise_options()
        sat_options()
    process_image()

def normalization_and_noise_options():
    tab1, tab2 = st.tabs(["Normalization", "Noise"])
    with tab1:
        normalization_option = st.selectbox(
            "Normalization", options=["None", "Percentile"],
            index=["None", "Percentile"].index(get_value('normalization', {}).get("option", "None")),
            help="Choose the normalization method for the image",
        )
        update_nested_session_state(['normalization', 'option'], normalization_option)
    with tab2:
        apply_gaussian_noise = st.checkbox(
            "Add Gaussian Noise", 
            value=get_value('gaussian_noise', {}).get("enabled", False),
            help="Add Gaussian noise to the image"
        )
        update_nested_session_state(['gaussian_noise', 'enabled'], apply_gaussian_noise)
        if apply_gaussian_noise:
            setup_gaussian_noise_params()

def sat_options():
    with st.popover("SAT Options"):
        use_sat = st.toggle(
            "Use Summed-Area Tables (SAT)", 
            value=get_value('use_sat', False),
            help="Enable Summed-Area Tables for faster Speckle Contrast calculation"
        )
        set_value('use_sat', use_sat)

def setup_gaussian_noise_params():
    noise_mean = st.sidebar.number_input(
        "Noise Mean", min_value=0.0, max_value=1.0,
        value=get_value('gaussian_noise', {}).get("mean", 0.1),
        step=0.01, format="%.2f",
        help="Mean value for Gaussian noise"
    )
    update_nested_session_state(['gaussian_noise', 'mean'], noise_mean)

    noise_std_dev = st.sidebar.number_input(
        "Noise Standard Deviation", min_value=0.0, max_value=1.0,
        value=get_value('gaussian_noise', {}).get("std_dev", 0.1),
        step=0.01, format="%.2f",
        help="Standard deviation for Gaussian noise"
    )
    update_nested_session_state(['gaussian_noise', 'std_dev'], noise_std_dev)

def process_image():
    image_array = get_value('image_array')
    if image_array is None:
        st.warning("No image loaded. Please select an image first.")
        return
    
    try:
        image_np = image_array.astype(np.float32) / 255.0
        if get_value('gaussian_noise', {}).get("enabled", False):
            image_np = apply_gaussian_noise(image_np)
        if get_value('normalization', {}).get("option") == "Percentile":
            image_np = apply_percentile_normalization(image_np)
        set_value('processed_image_np', image_np)
        
        # Add these lines to set the image data for both speckle and nlm techniques
        set_value('speckle_image', image_np)
        set_value('nlm_image', image_np)
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        st.exception(e)

def apply_gaussian_noise(image_np: np.ndarray) -> np.ndarray:
    gaussian_noise = get_value('gaussian_noise', {})
    noise = RNG.normal(
        gaussian_noise.get("mean", 0),
        gaussian_noise.get("std_dev", 0.1),
        image_np.shape
    )
    return np.clip(image_np + noise, 0, 1)

def apply_percentile_normalization(image_np: np.ndarray) -> np.ndarray:
    normalization = get_value('normalization', {})
    p_low, p_high = np.percentile(
        image_np,
        [normalization.get("percentile_low", 2),
         normalization.get("percentile_high", 98)]
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

def setup_search_window():
    image_shape = get_value('image').size
    max_search_window = min(101, *image_shape)
    default_search_window = min(21, max_search_window)
    use_whole_image = st.checkbox(
        "Use whole image as search window", 
        value=get_value('nlm_options', {}).get("use_whole_image", False),
        help="Use the entire image as the search window for NLM"
    )
    update_nested_session_state(['nlm_options', 'use_whole_image'], use_whole_image)
    
    search_window_size = get_search_window_size(
        use_whole_image,
        max_search_window, 
        default_search_window, 
        image_shape
    )
    update_nested_session_state(['search_window_size'], search_window_size)

def setup_filter_strength():
    filter_strength = st.slider(
        "Filter Strength", min_value=0.1, max_value=100.0,
        value=get_value('nlm_options', {}).get("filter_strength", 10.0),
        step=0.1, format="%.1f",
        help="Filter strength for NLM (higher values mean more smoothing)",)
    update_nested_session_state(['nlm_options', 'filter_strength'], filter_strength)

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