import streamlit as st
from PIL import Image
import numpy as np
import time
from typing import Dict, Any
from image_processing import handle_image_analysis, handle_image_comparison, create_placeholders_and_sections
import streamlit_nested_layout # type: ignore  # noqa: F401

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

def update_func(dummy=None):
    time.sleep(0.1)  # Add a small delay
    st.session_state.current_position = st.session_state.get('current_position', 1)
    update_images({"tabs": st.session_state.tabs, "analysis_params": {**st.session_state.analysis_params, "max_pixels": st.session_state.current_position}, "show_full_processed": st.session_state.sidebar_params['show_full_processed']}, 
                  {"speckle": st.session_state.speckle_placeholders, "nlm": st.session_state.nlm_placeholders})
    
def calculate_max_processable_pixels(image_width, image_height, kernel_size):
    return (image_width - kernel_size + 1) * (image_height - kernel_size + 1)

def setup_sidebar() -> Dict[str, Any]:
    # Initialize current_position if it doesn't exist
    if 'current_position' not in st.session_state:
        st.session_state.current_position = 1

    def update_pixels(key):
        if key == 'slider':
            st.session_state.pixels_input = st.session_state.pixels_slider
        else:
            st.session_state.pixels_slider = st.session_state.pixels_input
        st.session_state.current_position = st.session_state.pixels_slider
        update_func()

    with st.sidebar:
        st.title("Image Processing Settings")

        st.markdown("### 📷 Image Source")
        image_source = st.radio("Choose Image Source", ["Preloaded Image", "Upload Image"])
        
        if image_source == "Preloaded Image":
            selected_image = st.selectbox("Select Image", list(PRELOADED_IMAGES))
            uploaded_file = None
        else:
            uploaded_file = st.file_uploader("Upload Image")
            selected_image = None

        image = load_image(image_source, selected_image, uploaded_file)
        st.image(image, "Input Image", use_column_width=True)

        with st.expander("⚙️ Processing Parameters", expanded=True):
            kernel_size = st.number_input('Kernel Size', min_value=3, max_value=21, value=7, step=2,
                                          help="Size of the kernel for image processing")
            
            # Calculate max processable pixels
            max_pixels = calculate_max_processable_pixels(image.width, image.height, kernel_size)
            
            use_full_image = st.checkbox("Use Full Image for Search", value=False)
            if not use_full_image:
                search_window_size = st.number_input("Search Window Size", 
                                                     min_value=kernel_size + 2, 
                                                     max_value=min(max(image.width, image.height) // 2, 35),
                                                     value=kernel_size + 2,
                                                     step=2,
                                                     help="Size of the search window for NL-Means denoising")
            else:
                search_window_size = "full"
            
            filter_strength = st.number_input("Filter Strength (h)", 
                                              min_value=0.01, 
                                              max_value=30.0, 
                                              value=0.10, 
                                              step=0.01,
                                              format="%.2f",
                                              help="Strength of the NL-Means filter")
            
            cmap = st.selectbox("🎨 Color Map", COLOR_MAPS, index=0)

        with st.expander("🖼️ Display Options", expanded=True):
            show_full_processed = st.checkbox(
                "Show Fully Processed Image", 
                value=True,
                help="Toggle to switch between progressive processing and full image processing."
            )
            
            if not show_full_processed:
                col1, col2 = st.columns(2)
                play_pause = col1.button("▶️/⏸️", use_container_width=True) 
                reset = col2.button("🔄 Reset", use_container_width=True)
                
                pixels_to_process = st.slider(
                    "Pixels to process (slider)",
                    min_value=1,
                    max_value=max_pixels,
                    value=st.session_state.get('current_position', 1),
                    step=1,
                    key="pixels_slider",
                    on_change=update_pixels,
                    args=('slider',)
                )
                
                st.number_input(
                    "Pixels to process (input)",
                    min_value=1,
                    max_value=max_pixels,
                    value=st.session_state.get('current_position', 1),
                    step=1,
                    key="pixels_input",
                    on_change=update_pixels,
                    args=('input',)
                )
                
                st.session_state.current_position = pixels_to_process
            else:
                play_pause, reset, pixels_to_process = False, False, max_pixels

        with st.expander("🔬 Advanced Options"):
            add_noise = st.checkbox("Toggle Gaussian Noise")
            if add_noise:
                noise_mean = st.number_input("Noise Mean", 
                                             min_value=0.0, 
                                             max_value=1.0, 
                                             value=0.0, 
                                             step=0.01,
                                             format="%.2f")
                noise_std = st.number_input("Noise Std", 
                                            min_value=0.0, 
                                            max_value=1.0, 
                                            value=0.1, 
                                            step=0.01,
                                            format="%.2f")
                image_np = np.clip(np.array(image) / 255.0 + np.random.normal(noise_mean, noise_std, np.array(image).shape), 0, 1)
            else:
                image_np = np.array(image) / 255.0

    return {
        "image": image,
        "image_np": image_np,
        "show_full_processed": show_full_processed,
        "animation_params": {
            "play_pause": play_pause,
            "reset": reset,
            "pixels_to_process": pixels_to_process  
        },
        "kernel_size": kernel_size,
        "search_window_size": search_window_size,
        "filter_strength": filter_strength,
        "cmap": cmap,
        "max_pixels": max_pixels if show_full_processed else pixels_to_process,
    }

def update_images(params: Dict[str, Any], placeholders: Dict[str, Any]) -> None:
    speckle_results = handle_image_analysis(params['tabs'][0], **params['analysis_params'], technique="speckle", placeholders=placeholders['speckle'], show_full_processed=params['show_full_processed'])
    nlm_results = handle_image_analysis(params['tabs'][1], **params['analysis_params'], technique="nlm", placeholders=placeholders['nlm'], show_full_processed=params['show_full_processed'])
    st.session_state.processed_pixels = params['analysis_params']['max_pixels']
    st.session_state.speckle_results = speckle_results
    st.session_state.nlm_results = nlm_results

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

    analysis_params = {
        "image_np": sidebar_params['image_np'],
        "kernel_size": sidebar_params['kernel_size'],
        "search_window_size": sidebar_params['search_window_size'],
        "filter_strength": sidebar_params['filter_strength'],
        "cmap": sidebar_params['cmap'],
        "max_pixels": sidebar_params['max_pixels'],  # Use max_pixels instead of max_processable_pixels
        "height": sidebar_params['image_np'].shape[0],
        "width": sidebar_params['image_np'].shape[1]
    }

    # Store necessary variables in session state
    st.session_state.tabs = tabs
    st.session_state.speckle_placeholders = speckle_placeholders
    st.session_state.nlm_placeholders = nlm_placeholders
    st.session_state.sidebar_params = sidebar_params
    st.session_state.analysis_params = analysis_params

    if not sidebar_params['show_full_processed']:
        handle_animation(sidebar_params['animation_params'], sidebar_params['max_pixels'], update_func)

    # Always call update_func, regardless of show_full_processed
    update_func()

    # Process images for both techniques
    speckle_results = handle_image_analysis(
        tab=tabs[0],
        technique="speckle",
        placeholders=speckle_placeholders,
        **analysis_params
    )
    
    nlm_results = handle_image_analysis(
        tab=tabs[1],
        technique="nlm",
        placeholders=nlm_placeholders,
        **analysis_params
    )

    # Store results in session state
    st.session_state.speckle_results = speckle_results
    st.session_state.nlm_results = nlm_results

    # Handle image comparison
    if speckle_results is not None and nlm_results is not None:
        handle_image_comparison(tab=tabs[2], cmap_name=sidebar_params['cmap'], images={
            'Unprocessed Image': sidebar_params['image_np'],
            'Standard Deviation': speckle_results[1],
            'Speckle Contrast': speckle_results[2],
            'Mean Filter': speckle_results[0],
            'NL-Means Image': nlm_results[0]
        })
    else:
        with tabs[2]:
            st.warning("Please process the image before viewing comparisons.")

    # Add a section to display processing statistics
    with st.expander("Processing Statistics", expanded=False):
        st.write(f"Image size: {analysis_params['width']}x{analysis_params['height']}")
        st.write(f"Kernel size: {analysis_params['kernel_size']}x{analysis_params['kernel_size']}")
        st.write(f"Max processable pixels: {sidebar_params['max_pixels']}")
        st.write(f"Actual processed pixels: {min(sidebar_params['max_pixels'], (analysis_params['width'] - analysis_params['kernel_size'] + 1) * (analysis_params['height'] - analysis_params['kernel_size'] + 1))}")

if __name__ == "__main__":
    main()