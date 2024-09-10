import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import streamlit as st
import matplotlib.pyplot as plt
from streamlit_image_comparison import image_comparison
import logging

from analysis.speckle import process_speckle, visualize_speckle_results
from analysis.nlm import process_nlm,  visualize_nlm_results
from utils import calculate_processing_details 

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

FILTER_OPTIONS = {
    "speckle": ["Mean Filter", "Std Dev Filter", "Speckle Contrast"],
    "nlm": ["Weight Map", "NL-Means Image", "Difference Map"]
}

# ----------------------------- UI Elements ----------------------------- #

def create_ui_elements(technique: str, tab: st.delta_generator.DeltaGenerator, show_full_processed: bool) -> Dict[str, Any]:
    try:
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
    except Exception:
        st.error(f"Failed to create UI elements for {technique}. Please try again.")
        return {}

# ----------------------------- Image Processing ----------------------------- #

def process_and_visualize_image(params: Dict[str, Any]) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    try:
        image_np = params['image_np']
        technique = params['technique']
        analysis_params = params['analysis_params']
        show_full_processed = params['show_full_processed']
        
        # Process image
        if technique == "speckle":
            results = process_speckle(image_np, analysis_params['kernel_size'], analysis_params['max_pixels'])
        elif technique == "nlm":
            results = process_nlm(image_np, analysis_params['kernel_size'], analysis_params['max_pixels'], 
                                  analysis_params['search_window_size'], analysis_params['filter_strength'])
        else:
            raise ValueError(f"Unknown technique: {technique}")
        
        # Visualize results
        if params['handle_visualization']:
            details = calculate_processing_details(image_np, analysis_params['kernel_size'], analysis_params['max_pixels'])
            kernel_matrix, original_value = extract_kernel_info(
                image_np, details['last_x'], details['last_y'], analysis_params['kernel_size']
            )
            
            placeholders = st.session_state.get(f'{technique}_placeholders', {})
            
            visualization_params = {
                'image_np': image_np,
                'results': results,
                'placeholders': placeholders,
                'params': {
                    'analysis_params': analysis_params,
                    'show_full_processed': show_full_processed
                },
                'first_pixel': (details['last_x'], details['last_y']),
                'kernel_size': analysis_params['kernel_size'],
                'kernel_matrix': kernel_matrix,
                'original_value': original_value
            }
            
            if technique == "speckle":
                visualize_speckle_results(**visualization_params)
            elif technique == "nlm":
                visualize_nlm_results(**visualization_params)
        
        # Update state
        if params['update_state']:
            st.session_state.processed_pixels = params['pixels_to_process']
            st.session_state[f"{technique}_results"] = results
        
        return params, results
    except Exception as e:
        logger.error(f"Error during image processing and visualization: {str(e)}")
        st.error(f"Error during image processing and visualization: {str(e)}")
        st.exception(e)
        return params, None

def process_techniques(analysis_params: Dict[str, Any]) -> None:
    for technique in ["speckle", "nlm"]:
        try:
            tab = st.session_state.tabs[0 if technique == "speckle" else 1]
            placeholders = create_ui_elements(technique, tab, analysis_params['show_full_processed'])
            st.session_state[f"{technique}_placeholders"] = placeholders
            
            pixels_to_process = (analysis_params['max_pixels'] if analysis_params['show_full_processed'] 
                                 else st.session_state.pixels_to_process)
            
            params = {
                'image_np': analysis_params['image_np'],
                'technique': technique,
                'analysis_params': {
                    'kernel_size': analysis_params['kernel_size'],
                    'max_pixels': pixels_to_process,
                    'cmap': analysis_params['cmap'],
                    'search_window_size': analysis_params.get('search_window_size'),
                    'filter_strength': analysis_params.get('filter_strength')
                },
                'show_full_processed': analysis_params['show_full_processed'],
                'pixels_to_process': pixels_to_process,
                'update_state': True,
                'handle_visualization': True
            }
            
            _, results = process_and_visualize_image(params)
            st.session_state[f"{technique}_results"] = results
        except Exception as e:
            logger.error(f"Error processing {technique} technique: {str(e)}")
            st.error(f"Failed to process {technique} technique. Please check your inputs and try again.")

# ----------------------------- Image Comparison ----------------------------- #

def handle_image_comparison(tab: st.delta_generator.DeltaGenerator, cmap_name: str, images: Dict[str, np.ndarray]):
    with tab:
        st.header("Image Comparison")
        if not images:
            st.warning("No images available for comparison.")
            return
        
        available_images = list(images.keys())
        col1, col2 = st.columns(2)
        image_choice_1 = col1.selectbox('Select first image to compare:', [''] + available_images, index=0)
        image_choice_2 = col2.selectbox('Select second image to compare:', [''] + available_images, index=0)
        
        if image_choice_1 and image_choice_2:
            img1, img2 = images[image_choice_1], images[image_choice_2]
            img1_uint8, img2_uint8 = map(lambda img: (plt.get_cmap(cmap_name)((img - np.min(img)) / (np.max(img) - np.min(img)))[:, :, :3] * 255).astype(np.uint8), [img1, img2])
            
            if image_choice_1 != image_choice_2:
                image_comparison(img1=img1_uint8, img2=img2_uint8, label1=image_choice_1, label2=image_choice_2, make_responsive=True)
                st.subheader("Selected Images")
                st.image([img1_uint8, img2_uint8], caption=[image_choice_1, image_choice_2])
            else:
                st.error("Please select two different images for comparison.")
                st.image(np.abs(img1 - img2), caption="Difference Map", use_column_width=True)
        else:
            st.info("Select two images to compare.")

# ----------------------------- Kernel Extraction ----------------------------- #

def extract_kernel_info(image_np: np.ndarray, last_x: int, last_y: int, kernel_size: int) -> Tuple[List[List[float]], float]:
    """
    Extract and prepare kernel information from the image.

    Args:
    image_np (np.ndarray): The input image as a NumPy array.
    last_x (int): The x-coordinate of the center pixel.
    last_y (int): The y-coordinate of the center pixel.
    kernel_size (int): The size of the kernel.

    Returns:
    Tuple[List[List[float]], float]: A tuple containing the kernel matrix and the original center pixel value.
    """
    half_kernel = kernel_size // 2
    height, width = image_np.shape

    # Calculate kernel boundaries
    y_start = max(0, last_y - half_kernel)
    y_end = min(height, last_y + half_kernel + 1)
    x_start = max(0, last_x - half_kernel)
    x_end = min(width, last_x + half_kernel + 1)

    # Extract kernel values
    kernel_values = image_np[y_start:y_end, x_start:x_end]
    
    if kernel_values.size == 0:
        raise ValueError(f"Extracted kernel at ({last_x}, {last_y}) is empty. Image shape: {image_np.shape}, Kernel size: {kernel_size}")

    # Pad the kernel if necessary
    if kernel_values.shape != (kernel_size, kernel_size):
        pad_top = max(0, half_kernel - last_y)
        pad_bottom = max(0, last_y + half_kernel + 1 - height)
        pad_left = max(0, half_kernel - last_x)
        pad_right = max(0, last_x + half_kernel + 1 - width)
        kernel_values = np.pad(kernel_values, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='edge')

    # Convert to list of lists and ensure float type
    kernel_matrix = [[float(kernel_values[i, j]) for j in range(kernel_size)] for i in range(kernel_size)]
    
    # Get the original center pixel value
    original_value = float(image_np[last_y, last_x])

    return kernel_matrix, original_value