import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, Tuple, Dict, Any
from matplotlib.collections import LineCollection

from analysis.speckle import process_speckle, SpeckleResult
from analysis.nlm import process_nlm, NLMResult
from frontend.ui_elements import (
    create_technique_ui_elements, extract_kernel_from_image, create_process_params, 
    update_session_state
)

# Constants
KERNEL_COLOR = 'red'
SEARCH_WINDOW_COLOR = 'blue'
PIXEL_VALUE_COLOR = 'red'
ZOOMED_IMAGE_SIZE = (8, 8)

def process_image(params: Dict[str, Any]) -> Tuple[Dict[str, Any], Any]:
    try:
        image_np = params['image_np']
        technique = params['technique']
        analysis_params = params['analysis_params']
        pixels_to_process = analysis_params['pixels_to_process']
        
        results = run_analysis_technique(image_np, technique, analysis_params)
        
        if params.get('return_processed_only', False):
            return params, results
        
        if params['handle_visualization']:
            visualize_results(image_np, technique, analysis_params, results, params.get('show_per_pixel', False))
        
        if params['update_state']:
            update_session_state(technique, pixels_to_process, results)
        
        return params, results
    except Exception as e:
        st.error(f"Error in process_image: {str(e)}")
        return params, None

def run_analysis_technique(image_np: np.ndarray, technique: str, analysis_params: Dict[str, Any]) -> Any:
    if technique == "speckle":
        return process_speckle(image_np, analysis_params['kernel_size'], analysis_params['max_pixels'])
    elif technique == "nlm":
        return process_nlm(image_np, analysis_params['kernel_size'], analysis_params['max_pixels'], 
                           analysis_params['search_window_size'], analysis_params['filter_strength'])
    else:
        raise ValueError(f"Unknown technique: {technique}")

def prepare_filter_options_and_parameters(results, end_processed_pixel):
    end_x, end_y = end_processed_pixel
    
    st.error(f"Debug: Results type: {type(results)}")
    st.error(f"Debug: Results attributes: {dir(results)}")
    
    if isinstance(results, NLMResult):
        st.error("Debug: Processing NLMResult")
        filter_options = {
            "NL-Means Image": results.denoised_image,
            "Weight Map": results.weight_map_for_end_pixel,
            "Difference Map": results.difference_map
        }
        specific_params = {
            'filter_strength': results.filter_strength,
            'search_window_size': results.search_window_size,
            'pixels_processed': results.pixels_processed,
            'kernel_size': results.kernel_size,
            'nlm_value': results.denoised_image[end_y, end_x] if results.denoised_image is not None else None
        }
    elif isinstance(results, SpeckleResult):
        st.error("Debug: Processing SpeckleResult")
        filter_options = {
            "Mean Filter": results.mean_filter,
            "Std Dev Filter": results.std_dev_filter,
            "Speckle Contrast": results.speckle_contrast_filter,
        }
        specific_params = {
            'kernel_size': results.kernel_size,
            'pixels_processed': results.pixels_processed,
            'mean': results.mean_filter[end_y, end_x] if results.mean_filter is not None else None,
            'std': results.std_dev_filter[end_y, end_x] if results.std_dev_filter is not None else None,
            'sc': results.speckle_contrast_filter[end_y, end_x] if results.speckle_contrast_filter is not None else None,
        }
    else:
        st.error(f"Debug: Unknown result type: {type(results)}")
        filter_options = {}
        specific_params = {}
    
    st.error(f"Debug: Filter options before filtering: {filter_options}")
    filter_options = {k: v for k, v in filter_options.items() if v is not None}
    st.error(f"Debug: Filter options after filtering: {filter_options}")
    
    if isinstance(results, SpeckleResult) or hasattr(results, 'kernel_size'):
        specific_params['total_pixels'] = results.kernel_size ** 2
    
    st.error(f"Debug: Specific params: {specific_params}")
    return filter_options, {k: v for k, v in specific_params.items() if v is not None}

def visualize_analysis_results(
    image_np: np.ndarray,
    results: Any,
    placeholders: Dict[str, Any],
    params: Dict[str, Any],
    end_processed_pixel: Tuple[int, int],
    kernel_size: int,
    kernel_matrix: np.ndarray,
    original_value: float,
    analysis_type: str
) -> None:
    end_x, end_y = end_processed_pixel
    vmin, vmax = np.min(image_np), np.max(image_np)
    search_window_size = params['analysis_params'].get('search_window_size') if analysis_type == 'nlm' else None
    show_per_pixel = params['show_per_pixel']

    from frontend.formula import display_analysis_formula

    visualize_image(image_np, placeholders['original_image'], end_x, end_y, kernel_size, 
                    show_full=True, vmin=vmin, vmax=vmax, title="Original Image", 
                    technique=analysis_type, search_window_size=search_window_size, 
                    show_kernel=show_per_pixel, show_per_pixel=show_per_pixel)
    
    if show_per_pixel:
        visualize_image(image_np, placeholders['zoomed_original_image'], end_x, end_y, kernel_size, 
                        show_full=False, vmin=vmin, vmax=vmax, title="Zoomed-In Original Image", 
                        technique=analysis_type, search_window_size=search_window_size, 
                        zoom=True, show_kernel=True, show_per_pixel=show_per_pixel)

    filter_options, specific_params = prepare_filter_options_and_parameters(results, (end_x, end_y))
    
    visualize_filter_results(filter_options, placeholders, params, (end_x, end_y), kernel_size, analysis_type, search_window_size)

    if show_per_pixel:
        specific_params.update({
            'x': end_x, 'y': end_y, 'input_x': end_x, 'input_y': end_y,
            'kernel_size': kernel_size, 'kernel_matrix': kernel_matrix, 'original_value': original_value
        })
        display_analysis_formula(specific_params, placeholders, analysis_type)

    st.error(f"Debug: Calling visualize_analysis_results with params: {specific_params}")
    visualize_analysis_results(**specific_params)

def visualize_results(image_np: np.ndarray, technique: str, analysis_params: Dict[str, Any], results: Any, show_per_pixel: bool):
    from utils import calculate_processing_details

    st.error(f"Debug: Visualizing results for technique: {technique}")
    st.error(f"Debug: Analysis params: {analysis_params}")
    st.error(f"Debug: Results type: {type(results)}")

    details = calculate_processing_details(image_np, analysis_params['kernel_size'], analysis_params['max_pixels'])
    
    if isinstance(results, NLMResult):
        end_x, end_y = results.processing_end_coord
        st.error(f"Debug: NLMResult end coordinates: ({end_x}, {end_y})")
    elif isinstance(results, SpeckleResult):
        end_x, end_y = results.processing_start_coord
        st.error(f"Debug: SpeckleResult start coordinates: ({end_x}, {end_y})")
    else:
        end_x, end_y = details.end_x, details.end_y
        st.error(f"Debug: Using default end coordinates: ({end_x}, {end_y})")
    
    kernel_matrix, original_value = extract_kernel_from_image(
        image_np, end_x, end_y, analysis_params['kernel_size']
    )
    
    placeholders = st.session_state.get(f'{technique}_placeholders', {})
    st.error(f"Debug: Placeholders: {placeholders.keys()}")
    
    visualization_params = {
        'image_np': image_np,
        'results': results,
        'placeholders': placeholders,
        'params': {
            'analysis_params': analysis_params,
            'show_per_pixel': show_per_pixel
        },
        'end_processed_pixel': (end_x, end_y),
        'kernel_size': analysis_params['kernel_size'],
        'kernel_matrix': kernel_matrix,
        'original_value': original_value,
        'analysis_type': technique
    }
    
    st.error(f"Debug: Calling visualize_analysis_results with params: {visualization_params}")
    visualize_analysis_results(**visualization_params)

def visualize_filter_results(filter_options, placeholders, params, end_processed_pixel, kernel_size, analysis_type, search_window_size):
    show_per_pixel = params['show_per_pixel']
    end_x, end_y = end_processed_pixel

    for filter_name, filter_data in filter_options.items():
        if filter_data is not None:
            key = filter_name.lower().replace(" ", "_")
            if key in placeholders:
                visualize_image(filter_data, placeholders[key], end_x, end_y, kernel_size, 
                                show_full=True, vmin=np.min(filter_data), vmax=np.max(filter_data), 
                                title=filter_name, technique=analysis_type, search_window_size=search_window_size,
                                show_kernel=show_per_pixel, show_per_pixel=show_per_pixel)
                
                if show_per_pixel:
                    zoomed_key = f'zoomed_{key}'
                    if zoomed_key in placeholders:
                        visualize_image(filter_data, placeholders[zoomed_key], end_x, end_y, kernel_size, 
                                        show_full=False, vmin=np.min(filter_data), vmax=np.max(filter_data), 
                                        title=f"Zoomed-In {filter_name}", technique=analysis_type, 
                                        search_window_size=search_window_size, zoom=True, show_kernel=True, 
                                        show_per_pixel=show_per_pixel)

def visualize_image(
    image: np.ndarray,
    placeholder,
    x: Optional[int],
    y: Optional[int],
    kernel_size: int,
    show_full: bool,
    vmin: Optional[float],
    vmax: Optional[float],
    title: str,
    technique: Optional[str] = None,
    search_window_size: Optional[int] = None,
    zoom: bool = False,
    show_kernel: bool = False,
    show_per_pixel: bool = False,
) -> None:
    show_search_window = technique == "nlm" and search_window_size is not None and show_per_pixel

    if zoom:
        image, x, y = get_zoomed_image_section(image, x, y, kernel_size)
    
    fig = create_image_plot(
        image, x, y, kernel_size, title, 
        plot_search_window=search_window_size if show_search_window else None,
        zoom=zoom, vmin=vmin, vmax=vmax, show_kernel=show_kernel
    )
    
    placeholder.pyplot(fig)

def create_image_plot(
    plot_image: np.ndarray,
    plot_x: int,
    plot_y: int,
    plot_kernel_size: int,
    title: str,
    plot_search_window: Optional[int] = None,
    zoom: bool = False,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    show_kernel: bool = False
) -> plt.Figure:
    fig, ax = plt.subplots(1, 1, figsize=ZOOMED_IMAGE_SIZE)
    ax.imshow(plot_image, vmin=vmin, vmax=vmax, cmap=st.session_state.cmap)
    ax.set_title(title)
    ax.axis('off')
    
    if show_kernel:
        draw_kernel_overlay(ax, plot_x, plot_y, plot_kernel_size) 
    
    if plot_search_window is not None:
        draw_search_window_overlay(ax, plot_image, plot_x, plot_y, plot_search_window)
    
    if zoom:
        draw_pixel_value_annotations(ax, plot_image)
    
    fig.tight_layout(pad=2)
    return fig

def draw_kernel_overlay(ax: plt.Axes, x: int, y: int, kernel_size: int) -> None:
    kx, ky = x - kernel_size // 2, y - kernel_size // 2
    ax.add_patch(plt.Rectangle((kx - 0.5, ky - 0.5), kernel_size, kernel_size, 
                               edgecolor=KERNEL_COLOR, linewidth=1, facecolor="none"))
    lines = ([[(kx + i - 0.5, ky - 0.5), (kx + i - 0.5, ky + kernel_size - 0.5)] for i in range(1, kernel_size)] +
             [[(kx - 0.5, ky + i - 0.5), (kx + kernel_size - 0.5, ky + i - 0.5)] for i in range(1, kernel_size)])
    ax.add_collection(LineCollection(lines, colors=KERNEL_COLOR, linestyles=':', linewidths=0.5))
   
    ax.add_patch(plt.Rectangle((x - 0.5, y - 0.5), 1, 1,
                               edgecolor=SEARCH_WINDOW_COLOR, linewidth=0.5, facecolor=SEARCH_WINDOW_COLOR, alpha=0.5))

def draw_search_window_overlay(ax: plt.Axes, image: np.ndarray, x: int, y: int, search_window: int) -> None:
    height, width = image.shape[:2]
    if search_window >= max(height, width):
        rect = plt.Rectangle((-0.5, -0.5), width, height, 
                             edgecolor=SEARCH_WINDOW_COLOR, linewidth=2, facecolor="none")
    else:
        half_window = search_window // 2
        sw_left = max(-0.5, x - half_window - 0.5)
        sw_top = max(-0.5, y - half_window - 0.5)
        sw_width = min(width - sw_left, search_window)
        sw_height = min(height - sw_top, search_window)
        rect = plt.Rectangle((sw_left, sw_top), sw_width, sw_height,
                             edgecolor=SEARCH_WINDOW_COLOR, linewidth=2, facecolor="none")
    ax.add_patch(rect)

def draw_pixel_value_annotations(ax: plt.Axes, image: np.ndarray) -> None:
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            ax.text(j, i, f"{image[i, j]:.2f}", ha="center", va="center", color=PIXEL_VALUE_COLOR, fontsize=8)

def get_zoomed_image_section(image: np.ndarray, x: int, y: int, zoom_size: int) -> Tuple[np.ndarray, int, int]:
    ky = max(0, y - zoom_size // 2)
    kx = max(0, x - zoom_size // 2)
    zoomed_image = image[ky:min(image.shape[0], ky + zoom_size),
                         kx:min(image.shape[1], kx + zoom_size)]
    return zoomed_image, zoom_size // 2, zoom_size // 2
