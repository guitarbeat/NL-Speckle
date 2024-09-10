# Import Libraries
import numpy as np
from typing import Optional, Dict, Any, Tuple, List
import matplotlib.pyplot as plt
import streamlit as st
import time
from analysis.speckle import process_speckle
from analysis.nlm import process_nlm
from ui_components import create_ui_elements
from visualization import create_combined_plot, display_formula
from image_processing import calculate_processing_details, extract_kernel_info

# Main processing functions
def process_and_display_images(analysis_params: Dict[str, Any]) -> None:
    for technique in ["speckle", "nlm"]:
        tab = st.session_state.tabs[0 if technique == "speckle" else 1]
        placeholders = create_ui_elements(technique, tab, analysis_params['show_full_processed'])
        st.session_state[f"{technique}_placeholders"] = placeholders
        
        params, results = update_images_analyze_and_visualize(
            image_np=analysis_params['image_np'],
            kernel_size=analysis_params['kernel_size'],
            cmap=analysis_params['cmap'],
            technique=technique,
            search_window_size=analysis_params.get('search_window_size'),
            filter_strength=analysis_params.get('filter_strength'),
            show_full_processed=analysis_params['show_full_processed'],
            update_state=True,
            handle_visualization=True
        )
        st.session_state[f"{technique}_results"] = results


# Animation Handling

def handle_animation(sidebar_params: Dict[str, Any], analysis_params: Dict[str, Any]) -> None:
    if sidebar_params['show_full_processed']:
        return

    animation_params = sidebar_params['animation_params']
    
    if animation_params['play_pause']:
        st.session_state.animate = not st.session_state.get('animate', False)
    
    if animation_params['reset']:
        st.session_state.current_position = 1
        st.session_state.animate = False
    
    if st.session_state.get('animate', False):
        max_pixels = analysis_params['max_pixels']
        for i in range(st.session_state.current_position, max_pixels + 1):
            if not st.session_state.get('animate', False):
                break
            
            st.session_state.current_position = i
            # Create a new dictionary with only the expected arguments
            update_params = {
                'image_np': analysis_params['image_np'],
                'kernel_size': analysis_params['kernel_size'],
                'cmap': analysis_params['cmap'],
                'technique': analysis_params['technique'],
                'search_window_size': analysis_params.get('search_window_size'),
                'filter_strength': analysis_params.get('filter_strength'),
                'show_full_processed': analysis_params['show_full_processed']
            }
            update_images_analyze_and_visualize(**update_params)
            time.sleep(0.01)

# ----------------------------- Helper Functions ----------------------------- #


# Helper function (moved from being a separate function)
def prepare_filter_options_and_params(
    technique: str, 
    results: Dict[str, Any], 
    first_pixel: Tuple[int, int], 
    filter_strength: float, 
    search_window_size: int
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    first_x, first_y = first_pixel
    
    if technique == "speckle":
        return {
            "Mean Filter": results['mean_filter'],
            "Std Dev Filter": results['std_dev_filter'],
            "Speckle Contrast": results['speckle_contrast_filter']
        }, {
            "std": results['first_pixel_stats']['std_dev'],
            "mean": results['first_pixel_stats']['mean'],
            "sc": results['first_pixel_stats']['speckle_contrast'],
            "total_pixels": results['additional_info']['pixels_processed']
        }
    elif technique == "nlm":
        return {
            "NL-Means Image": results['processed_image'],
            "Weight Map": results['normalized_weight_map'],
            "Difference Map": np.abs(results['processed_image'] - results['additional_info']['image_dimensions'][0])
        }, {
            "filter_strength": filter_strength,
            "search_size": search_window_size,
            "total_pixels": results['additional_info']['pixels_processed'],
            "nlm_value": results['processed_image'][first_y, first_x]
        }
    else:
        return {}, {}


# ----------------------------- Image Processing ----------------------------- #

def update_images_analyze_and_visualize(
    image_np: np.ndarray,
    kernel_size: int,
    cmap: str,
    technique: str,
    search_window_size: Optional[int],
    filter_strength: float,
    show_full_processed: bool,
    update_state: bool = True,
    handle_visualization: bool = True
) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    if update_state:
        time.sleep(0.1)  # Add a small delay
        st.session_state.current_position = st.session_state.get('current_position', 1)
    
    max_pixels = st.session_state.current_position if not show_full_processed else None
    details = calculate_processing_details(image_np, kernel_size, max_pixels)
    
    params = {
        "tabs": st.session_state.get('tabs', []),
        "analysis_params": {
            "image_np": image_np,
            "kernel_size": kernel_size,
            "max_pixels": details['pixels_to_process'],
            "cmap": cmap,
            "search_window_size": search_window_size,
            "filter_strength": filter_strength
        },
        "show_full_processed": show_full_processed
    }
    
    results = None
    
    if handle_visualization:
        try:
            if technique == "speckle":
                results = process_speckle(image_np, kernel_size, details['pixels_to_process'])
            elif technique == "nlm":
                results = process_nlm(image_np, kernel_size, details['pixels_to_process'], search_window_size, filter_strength)
            else:
                raise ValueError(f"Unknown technique: {technique}")
            
            kernel_matrix, original_value = extract_kernel_info(image_np, details['last_x'], details['last_y'], kernel_size)
            
            placeholders = {"speckle": st.session_state.get('speckle_placeholders', {}),
                            "nlm": st.session_state.get('nlm_placeholders', {})}[technique]
            visualize_results(image_np, results, technique, placeholders, params, 
                              (details['last_x'], details['last_y']), kernel_size, 
                              kernel_matrix, original_value)
            
        except Exception as e:
            st.error(f"Error during image analysis and visualization: {str(e)}")
            st.exception(e)  # This will print the full traceback
            return params, None
    
    if update_state:
        st.session_state.processed_pixels = params['analysis_params']['max_pixels']
        st.session_state[f"{technique}_results"] = results
    return params, results

# Visualization

def visualize_results(
    image_np: np.ndarray,
    results: Dict[str, Any],
    technique: str,
    placeholders: Dict[str, Any],
    params: Dict[str, Any],
    first_pixel: Tuple[int, int],
    kernel_size: int,
    kernel_matrix: List[List[float]],
    original_value: float
):
    first_x, first_y = first_pixel
    vmin, vmax = np.min(image_np), np.max(image_np)
    show_full_processed = params['show_full_processed']
    cmap = params['analysis_params']['cmap']
    search_window_size = params['analysis_params'].get('search_window_size')
    filter_strength = params['analysis_params'].get('filter_strength')

    visualize_image(image_np, placeholders['original_image'], first_x, first_y, kernel_size, cmap, 
                    show_full_processed, vmin, vmax, "Original Image", technique, search_window_size)
    
    if not show_full_processed:
        visualize_image(image_np, placeholders['zoomed_original_image'], first_x, first_y, kernel_size, 
                        cmap, show_full_processed, vmin, vmax, "Zoomed-In Original Image", zoom=True)

    filter_options, specific_params = prepare_filter_options_and_params(
        technique, results, (first_x, first_y), filter_strength, search_window_size
    )
    
    for filter_name, filter_data in filter_options.items():
        key = filter_name.lower().replace(" ", "_")
        if key in placeholders:
            visualize_image(filter_data, placeholders[key], first_x, first_y, kernel_size, cmap, 
                            show_full_processed, np.min(filter_data), np.max(filter_data), filter_name)
            
            if not show_full_processed:
                zoomed_key = f'zoomed_{key}'
                if zoomed_key in placeholders:
                    visualize_image(filter_data, placeholders[zoomed_key], first_x, first_y, kernel_size, 
                                    cmap, show_full_processed, np.min(filter_data), np.max(filter_data), 
                                    f"Zoomed-In {filter_name}", zoom=True)

    specific_params.update({
        'x': first_x, 'y': first_y, 'input_x': first_x, 'input_y': first_y,
        'kernel_size': kernel_size, 'kernel_matrix': kernel_matrix, 'original_value': original_value
    })

    display_formula(placeholders['formula'], technique, **specific_params)

    plt.close('all')

def visualize_image(
    image: np.ndarray,
    placeholder: Any,
    x: int,
    y: int,
    kernel_size: int,
    cmap: str,
    show_full: bool,
    vmin: float,
    vmax: float,
    title: str,
    technique: str = None,
    search_window_size: int = None,
    zoom: bool = False
):
    if show_full and not zoom:
        fig = plt.figure()
        plt.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax)
        plt.axis('off')
        plt.title(title)
    else:
        if zoom:
            zoom_size = kernel_size
            ky = int(max(0, y - zoom_size // 2))
            kx = int(max(0, x - zoom_size // 2))
            image = image[ky:min(image.shape[0], ky + zoom_size),
                          kx:min(image.shape[1], kx + zoom_size)]
            x, y = zoom_size // 2, zoom_size // 2
        
        fig = create_combined_plot(image, x, y, kernel_size, title, cmap, 
                                   search_window_size if technique == "nlm" else None, 
                                   zoom=zoom, vmin=vmin, vmax=vmax)
    
    placeholder.pyplot(fig)
