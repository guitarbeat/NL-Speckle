import streamlit as st
from typing import Optional, Any, Dict, Tuple
import numpy as np
from PIL import Image

# Constants
PRELOADED_IMAGES = {
    "image50.png": "media/image50.png",
    "spatial.tif": "media/spatial.tif",
    "logo.jpg": "media/logo.jpg"
}
COLOR_MAPS = ["viridis", "plasma", "inferno", "magma", "cividis", "gray", "pink"]
FILTER_OPTIONS = {
    "speckle": ["Mean Filter", "Std Dev Filter", "Speckle Contrast"],
    "nlm": ["Weight Map", "NL-Means Image", "Difference Map"]
}

@st.cache_data
def load_image(image_source: str, selected_image: Optional[str] = None, uploaded_file: Any = None) -> Image.Image:
    if image_source == "Preloaded Image" and selected_image:
        return Image.open(PRELOADED_IMAGES[selected_image]).convert('L')
    elif image_source == "Upload Image" and uploaded_file:
        return Image.open(uploaded_file).convert('L')
    st.warning('Please upload or select an image.')
    st.stop()

def create_ui_elements(technique: str, tab: st.delta_generator.DeltaGenerator, show_full_processed: bool) -> Dict[str, Any]:
    with tab:
        placeholders = {'formula': st.empty(), 'original_image': st.empty()}
        selected_filters = st.multiselect("Select views to display", FILTER_OPTIONS[technique],
                                          default={"speckle": ["Speckle Contrast"], "nlm": ["NL-Means Image"]}[technique])
        
        columns = st.columns(len(selected_filters) + 1)
        for i, filter_name in enumerate(['Original Image'] + selected_filters):
            with columns[i]:
                key = filter_name.lower().replace(" ", "_")
                placeholders[key] = st.empty() if show_full_processed else st.expander(filter_name, expanded=True).empty()
                if not show_full_processed:
                    placeholders[f'zoomed_{key}'] = st.expander(f"Zoomed-in {filter_name.split()[0]}", expanded=False).empty()
        
        if not show_full_processed:
            placeholders['zoomed_kernel'] = placeholders.get('zoomed_kernel', st.empty())
        
        return placeholders

def setup_image_source() -> Tuple[str, Optional[str], Optional[Any], Image.Image]:
    st.markdown("### 📷 Image Source")
    image_source = st.radio("Choose Image Source", ["Preloaded Image", "Upload Image"])
    selected_image = st.selectbox("Select Image", list(PRELOADED_IMAGES)) if image_source == "Preloaded Image" else None
    uploaded_file = st.file_uploader("Upload Image") if image_source == "Upload Image" else None
    image = load_image(image_source, selected_image, uploaded_file)
    st.image(image, "Input Image", use_column_width=True)
    return image_source, selected_image, uploaded_file, image

def setup_processing_parameters(image: Image.Image) -> Dict[str, Any]:
    with st.expander("⚙️ Processing Parameters", expanded=True):
        kernel_size = st.number_input('Kernel Size', min_value=3, max_value=21, value=7, step=2)
        use_full_image = st.checkbox("Use Full Image for Search", value=False)
        search_window_size = (
            st.number_input("Search Window Size", 
                            min_value=kernel_size + 2, 
                            max_value=min(max(image.width, image.height) // 2, 35),
                            value=kernel_size + 2,
                            step=2,
                            help="Size of the search window for NL-Means denoising")
            if not use_full_image else None
        )
        filter_strength = st.number_input("Filter Strength (h)", min_value=0.01, max_value=30.0, value=0.10, step=0.01, format="%.2f")
        cmap = st.selectbox("🎨 Color Map", COLOR_MAPS, index=0)
    return {
        'kernel_size': kernel_size,
        'use_full_image': use_full_image,
        'search_window_size': search_window_size,
        'filter_strength': filter_strength,
        'cmap': cmap
    }

def setup_display_options(image: Image.Image, kernel_size: int) -> Dict[str, Any]:
    with st.expander("🖼️ Display Options", expanded=True):
        show_full_processed = st.checkbox("Show Fully Processed Image", value=True)
        if not show_full_processed:
            col1, col2 = st.columns(2)
            play_pause = col1.button("▶️/⏸️", use_container_width=True)
            reset = col2.button("🔄 Reset", use_container_width=True)
            max_pixels = (image.width - kernel_size + 1) * (image.height - kernel_size + 1)
            pixels_to_process = st.slider("Pixels to process", min_value=1, max_value=max_pixels, 
                                          value=st.session_state.get('current_position', 1), step=1, key="pixels_slider")
            st.number_input("Pixels to process", min_value=1, max_value=max_pixels, 
                            value=st.session_state.get('current_position', 1), step=1, key="pixels_input")
            st.session_state.current_position = pixels_to_process
        else:
            play_pause, reset = False, False
            pixels_to_process = (image.width - kernel_size + 1) * (image.height - kernel_size + 1)
    return {
        'show_full_processed': show_full_processed,
        'play_pause': play_pause,
        'reset': reset,
        'pixels_to_process': pixels_to_process
    }

def setup_advanced_options(image: Image.Image) -> np.ndarray:
    with st.expander("🔬 Advanced Options"):
        add_noise = st.checkbox("Toggle Gaussian Noise")
        if add_noise:
            noise_mean = st.number_input("Noise Mean", min_value=0.0, max_value=1.0, value=0.0, step=0.01, format="%.2f")
            noise_std = st.number_input("Noise Std", min_value=0.0, max_value=1.0, value=0.1, step=0.01, format="%.2f")
            image_np = np.clip(np.array(image) / 255.0 + np.random.normal(noise_mean, noise_std, np.array(image).shape), 0, 1)
        else:
            image_np = np.array(image) / 255.0
    return image_np

def setup_sidebar() -> Dict[str, Any]:
    if 'current_position' not in st.session_state:
        st.session_state.current_position = 1

    with st.sidebar:
        st.title("Image Processing Settings")
        
        # Image Source
        image_source, selected_image, uploaded_file, image = setup_image_source()
        
        # Processing Parameters
        processing_params = setup_processing_parameters(image)
        
        # Display Options
        display_options = setup_display_options(image, processing_params['kernel_size'])
        
        # Advanced Options
        image_np = setup_advanced_options(image)

    # Update session state
    st.session_state.update({
        'image_np': image_np,
        'kernel_size': processing_params['kernel_size'],
        'cmap': processing_params['cmap'],
        'search_window_size': processing_params['search_window_size'],
        'filter_strength': processing_params['filter_strength'],
        'show_full_processed': display_options['show_full_processed']
    })

    # Return consolidated parameters
    return {
        "image": image,
        "image_np": image_np,
        "show_full_processed": display_options['show_full_processed'],
        "animation_params": {
            "play_pause": display_options['play_pause'],
            "reset": display_options['reset'],
            "pixels_to_process": display_options['pixels_to_process']
        },
        "kernel_size": processing_params['kernel_size'],
        "search_window_size": processing_params['search_window_size'],
        "filter_strength": processing_params['filter_strength'],
        "cmap": processing_params['cmap'],
        "max_pixels": display_options['pixels_to_process'],
        "technique": st.session_state.get('technique', 'speckle')
    }