"""
This module defines functions for the sidebar UI in the Streamlit application.
It includes available color maps and preloaded image paths.
"""

import numpy as np
import streamlit as st
from PIL import Image
from typing import Optional

# Create a Generator object with a seed
rng = np.random.default_rng(seed=42)


###############################################################################
#                              Main UI Setup                                  #
###############################################################################

def setup_ui() -> Optional[bool]:
    # Apply custom CSS for better spacing and wider sidebar
    _apply_custom_css()

    # Main setup logic
    if not _image_selection():
        return None

    _display_options()
    _nlm_options()
    _advanced_options()

    return True

def _apply_custom_css():
    st.sidebar.markdown("""
        <style>
            [data-testid="stSidebar"] {
                min-width: 300px;
            }
        </style>
    """, unsafe_allow_html=True)

###############################################################################
#                           Image Selection Methods                           #
###############################################################################

def _image_selection() -> bool:
    with st.sidebar.expander("🖼️ Image Selector", expanded=True):
        image_source = st.radio("Select Image Source", ("Preloaded Images", "Upload Image"))
        
        if image_source == "Preloaded Images":
            return _load_preloaded_image()
        else:
            return _load_uploaded_image()

def _load_preloaded_image() -> bool:
    from main import PRELOADED_IMAGE_PATHS
    selected_image = st.selectbox("Select Image", list(PRELOADED_IMAGE_PATHS.keys()))
    try:
        image = Image.open(PRELOADED_IMAGE_PATHS[selected_image]).convert("L")
        st.session_state.image_file = selected_image
        return _process_loaded_image(image)
    except (FileNotFoundError, IOError) as e:
        st.error(f"Error loading image: {e}. Please try again.")
        return False

def _load_uploaded_image() -> bool:
    uploaded_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg", "tif", "tiff"])
    if uploaded_file is None:
        st.warning("Please upload an image.")
        return False
    try:
        image = Image.open(uploaded_file).convert("L")
        st.session_state.image_file = uploaded_file.name
        return _process_loaded_image(image)
    except IOError as e:
        st.error(f"Error loading image: {e}. Please try again.")
        return False

def _process_loaded_image(image: Image.Image) -> bool:
    st.image(image, caption="Input Image", use_column_width=True)
    st.session_state.image = image
    st.session_state.image_array = np.array(image)
    _select_color_map()
    return True

###############################################################################
#                          Display Options Methods                            #
###############################################################################

def _display_options():
    with st.sidebar.expander("🔧 Display Options", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            st.session_state.kernel_size = st.slider(
                "Kernel Size",
                min_value=3,
                max_value=21,
                value=st.session_state.get("kernel_size", 3),
                step=2,
                key="kernel_size_slider",
            )
        with col2:
            show_per_pixel = st.toggle(
                "Show Per-Pixel Processing Steps", 
                value=st.session_state.get("show_per_pixel", False),
                key="show_per_pixel",
            )
        
        total_pixels = (st.session_state.image.width - st.session_state.kernel_size + 1) * (
            st.session_state.image.height - st.session_state.kernel_size + 1
        )
        st.session_state.total_pixels = total_pixels
        
        if show_per_pixel:
            _setup_pixel_processing(total_pixels)
        else:
            st.session_state.exact_pixel_count = total_pixels

def _setup_pixel_processing(total_pixels: int):
    # Initialize session state variables if they don't exist
    if "desired_percentage" not in st.session_state:
        st.session_state.desired_percentage = 100
    if "desired_exact_count" not in st.session_state:
        st.session_state.desired_exact_count = total_pixels

    def update_exact_count():
        st.session_state.desired_exact_count = max(1, int(
            total_pixels * st.session_state.desired_percentage / 100
        ))

    def update_percentage():
        st.session_state.desired_percentage = max(1, min(100, int(
            (st.session_state.desired_exact_count / total_pixels) * 100
        )))

    with st.popover("Pixel Processing Options"):
        col1, col2 = st.columns(2)
        with col1:
            percentage = st.slider(
                "Percentage",
                min_value=1,
                max_value=100,
                value=st.session_state.desired_percentage,
                key="percentage_slider",
                on_change=update_exact_count,
            )

        with col2:
            exact_count = st.number_input(
                "Exact Pixels",
                min_value=1,
                max_value=total_pixels,
                value=st.session_state.desired_exact_count,
                key="exact_pixel_count",
                on_change=update_percentage,
            )

    # Update session state with the widget values
    st.session_state.desired_percentage = percentage
    st.session_state.desired_exact_count = exact_count
    st.session_state.pixels_to_process = exact_count

    # Display the selected values outside the popover
    st.write(f"Processing {percentage}% ({exact_count} pixels)")

# Display options methods
def _select_color_map():
    from main import AVAILABLE_COLOR_MAPS
    if "color_map" not in st.session_state:
        st.session_state.color_map = "gray"

    st.session_state.color_map = st.selectbox(
        "Select Color Map",
        AVAILABLE_COLOR_MAPS,
        index=AVAILABLE_COLOR_MAPS.index(st.session_state.color_map),
        key="color_map_select",
    )

###############################################################################
#                         Advanced Options Methods                            #
###############################################################################

def _advanced_options():
    with st.sidebar.expander("🔬 Advanced Options", expanded=False):
        tab1, tab2 = st.tabs(["Normalization", "Noise"])
        
        with tab1:
            st.session_state.normalization_option = st.selectbox(
                "Normalization",
                options=["None", "Percentile"],
                index=0,
                help="Choose the normalization method for the image",
            )

        with tab2:
            st.session_state.apply_gaussian_noise = st.checkbox(
                "Add Gaussian Noise", value=False, help="Add Gaussian noise to the image"
            )
            if st.session_state.apply_gaussian_noise:
                _setup_gaussian_noise_params()

        with st.popover("SAT Options"):
            st.session_state.use_sat = st.toggle(
                "Use Summed-Area Tables (SAT)",
                value=False,
                help="Enable Summed-Area Tables for faster Speckle Contrast calculation"
            )

    _process_image()

def _setup_gaussian_noise_params():
    st.session_state.noise_mean = st.sidebar.number_input(
        "Noise Mean",
        min_value=0.0,
        max_value=1.0,
        value=0.1,
        step=0.01,
        format="%.2f",
    )
    st.session_state.noise_std_dev = st.sidebar.number_input(
        "Noise Standard Deviation",
        min_value=0.0,
        max_value=1.0,
        value=0.1,
        step=0.01,
        format="%.2f",
    )

def _process_image():
    image_np = st.session_state.image_array / 255.0
    if st.session_state.apply_gaussian_noise:
        noise = rng.normal(st.session_state.noise_mean, st.session_state.noise_std_dev, image_np.shape)
        image_np = np.clip(image_np + noise, 0, 1)
    if st.session_state.normalization_option == "Percentile":
        p_low, p_high = np.percentile(image_np, [2, 98])
        image_np = np.clip(image_np, p_low, p_high)
        image_np = (image_np - p_low) / (p_high - p_low)
    st.session_state.processed_image_np = image_np

###############################################################################
#                             NLM Options Method                              #
###############################################################################

def _nlm_options():
    with st.sidebar.expander("🔍 NLM Options", expanded=False):
        image_shape = st.session_state.image.size
        max_search_window = min(101, *image_shape)
        default_search_window = min(21, max_search_window)
        st.session_state.use_whole_image = st.checkbox(
            "Use whole image as search window", value=True
        )

        st.session_state.search_window_size = _get_search_window_size(
            st.session_state.use_whole_image, 
            max_search_window, 
            default_search_window, 
            image_shape
        )

        st.session_state.filter_strength = st.slider(
            "Filter Strength",
            min_value=0.1,
            max_value=100.0,
            value=10.0,
            step=0.1,
            format="%.1f",
            help="Filter strength for NLM (higher values mean more smoothing)",
        )

def _get_search_window_size(use_whole_image: bool, max_search_window: int, default_search_window: int, image_shape: tuple) -> int:
    if use_whole_image:
        return max(image_shape)
    
    search_window_size = st.slider(
        "Search Window Size",
        min_value=3,
        max_value=max_search_window,
        value=default_search_window,
        step=2,
        help="Size of the search window for NLM (must be odd)",
    )
    return search_window_size if search_window_size % 2 == 1 else search_window_size + 1