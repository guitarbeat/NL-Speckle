import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from streamlit_image_comparison import image_comparison
from typing import Dict, Any
from PIL import Image

# Constants
FILTER_OPTIONS = {
    "speckle": ["Mean Filter", "Std Dev Filter", "Speckle Contrast"],
    "nlm": ["Weight Map", "NL-Means Image", "Difference Map"]
}
COLOR_MAPS = ["gray", "viridis", "plasma", "inferno", "magma", "cividis", "pink"]
PRELOADED_IMAGES = {
    "image50.png": "media/image50.png",
    "spatial.tif": "media/spatial.tif",
    "logo.jpg": "media/logo.jpg"
}

def load_image():
    st.sidebar.markdown("### 📷 Image Source")
    image_source = st.sidebar.radio("Select Image Source", ("Preloaded Images", "Upload Image"))
    
    try:
        if image_source == "Preloaded Images":
            selected_image = st.sidebar.selectbox("Select Image", list(PRELOADED_IMAGES))
            image = Image.open(PRELOADED_IMAGES[selected_image]).convert('L')
        else:
            uploaded_file = st.sidebar.file_uploader("Upload Image")
            image = Image.open(uploaded_file).convert('L') if uploaded_file else None

        if image is None:
            st.sidebar.warning('Please select or upload an image.')
            return None

        st.sidebar.image(image, "Input Image", use_column_width=True)
        return image
    except Exception as e:
        st.sidebar.error(f"Failed to load the image: {str(e)}")
        return None

def setup_color_map():
    st.sidebar.markdown("### 🎨 Color Map")
    cmap = st.session_state.get('cmap', COLOR_MAPS[0])
    cmap = st.sidebar.selectbox("Select Color Map", COLOR_MAPS, index=COLOR_MAPS.index(cmap))
    st.session_state.cmap = cmap
    return cmap

def get_display_options(image, kernel_size):
    try:
        col1, col2 = st.columns(2)
        show_per_pixel = col1.toggle("Show Per-Pixel Processing Steps", value=False)
        max_pixels = (image.width - kernel_size + 1) * (image.height - kernel_size + 1)
        
        pixels_to_process = max_pixels
        if show_per_pixel:
            percentage = col2.slider("Percentage of Pixels to Process", 
                                     min_value=1, max_value=100, value=100, step=1,
                                     key=f"percentage_slider_{kernel_size}")  # Add kernel_size to the key
            st.session_state.percentage = percentage
            pixels_to_process = int(max_pixels * percentage / 100)

        return {"show_per_pixel": show_per_pixel, "max_pixels": max_pixels, "pixels_to_process": pixels_to_process}
    except Exception as e:
        st.error(f"Failed to set display options: {str(e)}")
        return {"show_per_pixel": False, "max_pixels": max_pixels, "pixels_to_process": max_pixels}

def get_advanced_options(image):
    try:
        with st.sidebar.expander("🔬 Advanced Options"):
            add_noise = st.checkbox("Add Gaussian Noise", value=False,
                                    help="Add Gaussian noise to the image")
            if add_noise:
                noise_mean = st.number_input("Noise Mean", min_value=0.0, max_value=1.0, value=0.0, step=0.01, format="%.2f")
                noise_std = st.number_input("Noise Std", min_value=0.0, max_value=1.0, value=0.1, step=0.01, format="%.2f")
                image_np = np.clip(np.array(image) / 255.0 + np.random.normal(noise_mean, noise_std, np.array(image).shape), 0, 1)
            else:
                image_np = np.array(image) / 255.0
        return {"image_np": image_np, "add_noise": add_noise}
    except Exception as e:
        st.sidebar.error(f"Failed to set advanced options: {str(e)}")
        return {"image_np": np.array(image) / 255.0, "add_noise": False}

def setup_sidebar():
    try:
        st.sidebar.title("Image Processing Settings")
        image = load_image()
        if image is None:
            return None

        cmap = setup_color_map()
        display_options = get_display_options(image, 7)  # Use default kernel_size of 7
        advanced_options = get_advanced_options(image)

        return {
            "image": image,
            "cmap": cmap,
            **display_options,
            **advanced_options
        }
    except Exception as e:
        st.sidebar.error(f"An error occurred while setting up the sidebar: {str(e)}")
        return None

def get_technique_params(technique: str, analysis_params: Dict[str, Any]) -> Dict[str, Any]:
    kernel_size = st.slider('Kernel Size', min_value=3, max_value=21, value=7, step=2, key=f'kernel_size_{technique}')
    
    # Recalculate max_pixels based on the new kernel size
    image_height, image_width = analysis_params['image_np'].shape[:2]
    max_pixels = (image_width - kernel_size + 1) * (image_height - kernel_size + 1)
    
    pixels_to_process = max_pixels
    if analysis_params.get('show_per_pixel', False):
        percentage = st.session_state.get('percentage', 100)
        pixels_to_process = int(max_pixels * percentage / 100)
    
    st.write(f"Processing {pixels_to_process:,} out of {max_pixels:,} pixels")

    technique_params = {
        'kernel_size': kernel_size,
        'max_pixels': max_pixels,
        'pixels_to_process': pixels_to_process,
        'cmap': analysis_params.get('cmap', 'gray'),
    }
    
    if technique == "nlm":
        technique_params.update(get_nlm_specific_params(kernel_size))
    
    return technique_params

def get_nlm_specific_params(kernel_size: int) -> Dict[str, Any]:
    params = {
        'filter_strength': st.slider('Filter Strength (h)', min_value=0.01, max_value=30.0, value=0.10, step=0.01, key='filter_strength_nlm'),
    }
    
    max_window_size = st.session_state.analysis_params['image_np'].shape[0]  # Assuming square image
    params['search_window_size'] = st.slider('Search Window Size', 
                                             min_value=kernel_size, 
                                             max_value=max_window_size, 
                                             value=min(51, max_window_size), 
                                             step=2, 
                                             key='search_window_size_nlm')
    
    return params

def create_process_params(analysis_params: Dict[str, Any], technique: str, technique_params: Dict[str, Any]) -> Dict[str, Any]:
    return {
        'image_np': analysis_params['image_np'],
        'technique': technique,
        'analysis_params': technique_params,
        'pixels_to_process': technique_params['max_pixels'],
        'update_state': True,
        'handle_visualization': True,
        'show_per_pixel': analysis_params['show_per_pixel']
    }

def extract_kernel_from_image(image_np, end_x, end_y, kernel_size):
    half_kernel = kernel_size // 2
    height, width = image_np.shape

    y_start, y_end = max(0, end_y - half_kernel), min(height, end_y + half_kernel + 1)
    x_start, x_end = max(0, end_x - half_kernel), min(width, end_x + half_kernel + 1)

    kernel_values = image_np[y_start:y_end, x_start:x_end]
    
    if kernel_values.size == 0:
        raise ValueError(f"Extracted kernel at ({end_x}, {end_y}) is empty. Image shape: {image_np.shape}, Kernel size: {kernel_size}")

    if kernel_values.shape != (kernel_size, kernel_size):
        pad_width = [
            (max(0, half_kernel - end_y), max(0, end_y + half_kernel + 1 - height)),
            (max(0, half_kernel - end_x), max(0, end_x + half_kernel + 1 - width))
        ]
        kernel_values = np.pad(kernel_values, pad_width, mode='edge')

    return kernel_values.astype(float), float(image_np[end_y, end_x])

def update_session_state(technique: str, pixels_to_process: int, results: Any):
    st.session_state.processed_pixels = pixels_to_process
    st.session_state[f"{technique}_results"] = results
    print(f"NLM results stored in session state: {st.session_state.get('nlm_results')}")

def prepare_comparison_images():
    speckle_results = st.session_state.get("speckle_results")
    nlm_results = st.session_state.get("nlm_results")
    analysis_params = st.session_state.analysis_params

    if speckle_results is not None and nlm_results is not None:
        return {
            'Unprocessed Image': analysis_params['image_np'],
            'Standard Deviation': speckle_results.std_dev_filter,
            'Speckle Contrast': speckle_results.speckle_contrast_filter,
            'Mean Filter': speckle_results.mean_filter,
            'NL-Means Image': nlm_results.denoised_image,
            'NLM Weight Map': nlm_results.weight_map_for_end_pixel,
            'NLM Difference Map': nlm_results.difference_map
        }
    else:
        return None

def normalize_image(img: np.ndarray, cmap_name: str) -> np.ndarray:
    normalized = (img - np.min(img)) / (np.max(img) - np.min(img))
    colored = plt.get_cmap(cmap_name)(normalized)[:, :, :3]
    return (colored * 255).astype(np.uint8)

def render_image_comparison_ui(available_images: list[str]) -> tuple[str, str]:
    col1, col2 = st.columns(2)
    image_choice_1 = col1.selectbox('Select first image to compare:', [''] + available_images, index=0)
    image_choice_2 = col2.selectbox('Select second image to compare:', [''] + available_images, index=0)
    return image_choice_1, image_choice_2

def display_comparison(img1: np.ndarray, img2: np.ndarray, label1: str, label2: str):
    if label1 != label2:
        image_comparison(img1=img1, img2=img2, label1=label1, label2=label2, make_responsive=True)
        st.subheader("Selected Images")
        st.image([img1, img2], caption=[label1, label2])
    else:
        st.error("Please select two different images for comparison.")
        st.image(np.abs(img1 - img2), caption="Difference Map", use_column_width=True)

def handle_image_comparison(tab, cmap_name: str, images: dict[str, np.ndarray]):
    with tab:
        st.header("Image Comparison")
        if not images:
            st.warning("No images available for comparison.")
            return

        available_images = list(images.keys())
        image_choice_1, image_choice_2 = render_image_comparison_ui(available_images)

        if image_choice_1 and image_choice_2:
            img1, img2 = images[image_choice_1], images[image_choice_2]
            img1_uint8, img2_uint8 = map(lambda img: normalize_image(img, cmap_name), [img1, img2])
            display_comparison(img1_uint8, img2_uint8, image_choice_1, image_choice_2)
        else:
            st.info("Select two images to compare.")

def create_technique_ui_elements(technique, tab, show_per_pixel):
    with tab:
        placeholders = {'formula': st.empty(), 'original_image': st.empty()}
        filter_options = FILTER_OPTIONS[technique]
        selected_filters = st.multiselect("Select views to display", filter_options,
                                          default=[filter_options[0]])
        
        columns = st.columns(len(selected_filters) + 1)
        for i, filter_name in enumerate(['Original Image'] + selected_filters):
            with columns[i]:
                key = filter_name.lower().replace(" ", "_")
                placeholders[key] = st.empty()
                placeholders[f'zoomed_{key}'] = st.expander(f"Zoomed-in {filter_name}", expanded=False).empty()
        
        placeholders['zoomed_kernel'] = st.empty()
        
    return placeholders