import streamlit as st
from PIL import Image
import numpy as np
import time
from typing import Dict, Any, Tuple
from image_processing import handle_image_analysis, handle_image_comparison, create_placeholders_and_sections
import streamlit_nested_layout # type: ignore

# Constants
TABS = ["Speckle Contrast Calculation", "Non-Local Means Denoising", "Speckle Contrast Comparison"]
PAGE_CONFIG = {
    "page_title": "Speckle Contrast Visualization",
    "layout": "wide",
    "page_icon": "favicon.png",
    "initial_sidebar_state": "expanded"
}
PRELOADED_IMAGES = {
    "image50.png": "media/image50.png",
    "spatial.tif": "media/spatial.tif",
    "logo.jpg": "media/logo.jpg"
}
COLOR_MAPS = ["viridis", "plasma", "inferno", "magma", "cividis", "gray", "pink"]

@st.cache_data
def load_image(image_source: str, selected_image: str = None, uploaded_file: Any = None) -> Image.Image:
    if image_source == "Preloaded Image" and selected_image:
        return Image.open(PRELOADED_IMAGES[selected_image]).convert('L')
    elif image_source == "Upload Image" and uploaded_file:
        return Image.open(uploaded_file).convert('L')
    st.warning('Please upload or select an image.')
    st.stop()

def setup_sidebar() -> Dict[str, Any]:
    with st.sidebar:
        st.title("Image Processing Settings")
        
        show_full_processed = st.checkbox(
            "🖼️ Show Fully Processed Image", 
            value=False,
            help="Toggle to switch between progressive processing and full image processing."
        )
        st.markdown("---")
        
        if show_full_processed:
            st.info("Full image processing mode is active. The entire image will be processed at once.")
        else:
            st.info("Progressive processing mode is active. You can control the number of pixels processed.")
        
        image_source, selected_image, uploaded_file = setup_image_source()
        image = load_image(image_source, selected_image, uploaded_file)
        st.image(image, "Input Image", use_column_width=True)

        processing_params = setup_processing_parameters(image)
        image_np = apply_noise(np.array(image) / 255.0)

    return {
        "image": image,
        "image_np": image_np,
        "show_full_processed": show_full_processed,
        **processing_params
    }

def setup_image_source() -> Tuple[str, str, Any]:
    st.markdown("### 🔍 Image Source")
    image_source = st.radio("Choose Image Source", ["Preloaded Image", "Upload Image"])
    selected_image = st.selectbox("Select Image", list(PRELOADED_IMAGES)) if image_source == "Preloaded Image" else None
    uploaded_file = st.file_uploader("Upload Image") if image_source == "Upload Image" else None
    return image_source, selected_image, uploaded_file

def setup_processing_parameters(image: Image.Image) -> Dict[str, Any]:
    st.markdown("### ⚙️ Processing Parameters")
    with st.form("processing_params"):
        kernel_size = st.slider('Kernel Size', 3, 21, 7, 2)
        stride = st.slider('Stride', 1, 5, 1)
        use_full_image = st.checkbox("Use Full Image for Search", value=False)
        search_window_size = "full" if use_full_image else st.slider("Search Window Size", kernel_size + 2, min(max(image.width, image.height) // 2, 35), kernel_size + 2, step=2)
        filter_strength = st.slider("Filter Strength (h)", 0.01, 30.0, 0.10)
        cmap = st.selectbox("🎨 Color Map", COLOR_MAPS, index=0)
        st.form_submit_button("Apply Settings")
    
    return {
        "kernel_size": kernel_size,
        "stride": stride,
        "search_window_size": search_window_size,
        "filter_strength": filter_strength,
        "cmap": cmap
    }

def apply_noise(image_np: np.ndarray) -> np.ndarray:
    st.markdown("### 🔬 Advanced Options")
    if st.checkbox("Toggle Gaussian Noise"):
        noise_mean = st.slider("Noise Mean", 0.0, 1.0, 0.0, 0.01)
        noise_std = st.slider("Noise Std", 0.0, 1.0, 0.1, 0.01)
        image_np = np.clip(image_np + np.random.normal(noise_mean, noise_std, image_np.shape), 0, 1)
    return image_np

def update_images(params: Dict[str, Any], placeholders: Dict[str, Any]) -> None:
    speckle_results = handle_image_analysis(params['tabs'][0], **params['analysis_params'], technique="speckle", placeholders=placeholders['speckle'], show_full_processed=params['show_full_processed'])
    nlm_results = handle_image_analysis(params['tabs'][1], **params['analysis_params'], technique="nlm", placeholders=placeholders['nlm'], show_full_processed=params['show_full_processed'])
    st.session_state.processed_pixels = params['analysis_params']['max_pixels']
    st.session_state.speckle_results = speckle_results
    st.session_state.nlm_results = nlm_results

def setup_animation_controls(max_processable_pixels: int) -> Dict[str, Any]:
    with st.expander("Animation Settings", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            play_pause = st.button("⏯️ Play/Pause", use_container_width=True)
        with col2:
            reset = st.button("🔄 Reset", use_container_width=True)
        
        if 'current_position' not in st.session_state:
            st.session_state.current_position = 1
        
        pixels_to_process = st.slider("Pixels to process", 1, max_processable_pixels, 
                                      st.session_state.current_position)
        
        st.session_state.current_position = pixels_to_process
        
    return {
        "play_pause": play_pause,
        "reset": reset,
        "pixels_to_process": pixels_to_process
    }

def handle_animation(animation_params: Dict[str, Any], max_processable_pixels: int, update_func: callable):
    if animation_params['play_pause']:
        st.session_state.animate = not st.session_state.get('animate', False)
    
    if animation_params['reset']:
        st.session_state.current_position = 1
        st.session_state.animate = False

    if st.session_state.get('animate', False):
        for i in range(st.session_state.current_position, max_processable_pixels + 1):
            st.session_state.current_position = i
            update_func(i)
            time.sleep(0.01)
            if not st.session_state.get('animate', False):
                break

def main():
    st.set_page_config(**PAGE_CONFIG)
    st.logo("media/logo.png")

    sidebar_params = setup_sidebar()
    tabs = st.tabs(TABS)
    
    speckle_placeholders = create_placeholders_and_sections("speckle", tabs[0], sidebar_params['show_full_processed'])
    nlm_placeholders = create_placeholders_and_sections("nlm", tabs[1], sidebar_params['show_full_processed'])

    max_processable_pixels = sidebar_params['image'].width * sidebar_params['image'].height

    if not sidebar_params['show_full_processed']:
        animation_params = setup_animation_controls(max_processable_pixels)
    else:
        animation_params = {"play_pause": False, "reset": False, "pixels_to_process": max_processable_pixels}

    analysis_params = {
        "image_np": sidebar_params['image_np'],
        "kernel_size": sidebar_params['kernel_size'],
        "stride": sidebar_params['stride'],
        "search_window_size": sidebar_params['search_window_size'],
        "filter_strength": sidebar_params['filter_strength'],
        "cmap": sidebar_params['cmap'],
        "max_pixels": animation_params['pixels_to_process']
    }

    def update_func(i):
        update_images({"tabs": tabs, "analysis_params": {**analysis_params, "max_pixels": i}, "show_full_processed": sidebar_params['show_full_processed']}, 
                      {"speckle": speckle_placeholders, "nlm": nlm_placeholders})

    if not sidebar_params['show_full_processed']:
        handle_animation(animation_params, max_processable_pixels, update_func)
    else:
        update_func(max_processable_pixels)

    if 'speckle_results' in st.session_state and 'nlm_results' in st.session_state:
        handle_image_comparison(tab=tabs[2], cmap_name=sidebar_params['cmap'], images={
            'Unprocessed Image': sidebar_params['image_np'],
            'Standard Deviation': st.session_state.speckle_results[1],
            'Speckle Contrast': st.session_state.speckle_results[2],
            'Mean Filter': st.session_state.speckle_results[0],
            'NL-Means Image': st.session_state.nlm_results[0]
        })
    else:
        with tabs[2]:
            st.warning("Please process the image before viewing comparisons.")

if __name__ == "__main__":
    main()