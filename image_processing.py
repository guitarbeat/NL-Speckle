import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from typing import Tuple, List, Optional, Union, Any, Dict, Callable
from numba import jit
import time
from PIL import Image
from constants import PRELOADED_IMAGES


# ---------------------------- Image Processing Functions ---------------------------- #

@jit(nopython=True, parallel=True)
def process_pixels(image: np.ndarray, output_arrays: Tuple[np.ndarray, ...], kernel_size: int, pixels_to_process: int, 
                   height: int, width: int, first_x: int, first_y: int, calculation):
    half_kernel = kernel_size // 2

    for pixel in range(pixels_to_process):
        row = first_y + pixel // (width - kernel_size + 1)
        col = first_x + pixel % (width - kernel_size + 1)

        local_window = image[row-half_kernel:row+half_kernel+1, col-half_kernel:col+half_kernel+1]
        results = calculation(local_window)

        for i, result in enumerate(results):
            output_arrays[i][row, col] = result

    return output_arrays

def process_nlm(image: np.ndarray, denoised_image: np.ndarray, weight_sum_map: np.ndarray, kernel_size: int, 
               search_size: Optional[int], filter_strength: float, pixels_to_process: int, height: int, width: int, 
               first_x: int, first_y: int, calculation: Callable):
    half_kernel = kernel_size // 2

    @jit(nopython=True)
    def nlm_pixel_process(center_row, center_col, image, denoised_image, weight_sum_map, kernel_size, search_size, filter_strength, height, width, calculation):
        center_patch = image[center_row-half_kernel:center_row+half_kernel+1, center_col-half_kernel:center_col+half_kernel+1]
        
        denoised_value = 0.0
        weight_sum = 0.0

        if search_size is None:
            search_y_start, search_y_end = 0, height
            search_x_start, search_x_end = 0, width
        else:
            search_y_start = max(0, center_row - search_size // 2)
            search_y_end = min(height, center_row + search_size // 2 + 1)
            search_x_start = max(0, center_col - search_size // 2)
            search_x_end = min(width, center_col + search_size // 2 + 1)

        for i in range(search_y_start, search_y_end):
            for j in range(search_x_start, search_x_end):
                if j < half_kernel or j >= width - half_kernel or i < half_kernel or i >= height - half_kernel:
                    continue
                
                comparison_patch = image[i-half_kernel:i+half_kernel+1, j-half_kernel:j+half_kernel+1]
                weight = calculation(center_patch, comparison_patch, filter_strength)
                
                denoised_value += image[i, j] * weight
                weight_sum += weight
                weight_sum_map[i, j] += weight

        denoised_image[center_row, center_col] = denoised_value / weight_sum if weight_sum > 0 else image[center_row, center_col]

    for pixel in range(pixels_to_process):
        row = first_y + pixel // (width - kernel_size + 1)
        col = first_x + pixel % (width - kernel_size + 1)
        nlm_pixel_process(row, col, image, denoised_image, weight_sum_map, kernel_size, search_size, filter_strength, height, width, calculation)

    return denoised_image, weight_sum_map, first_x, first_y, weight_sum_map[first_y, first_x]

def process_speckle(image: np.ndarray, kernel_size: int, pixels_to_process: int, height: int, width: int, 
                    first_x: int, first_y: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, int, float, float, float]:
    mean_filter = np.zeros((height, width), dtype=np.float32)
    std_dev_filter = np.zeros((height, width), dtype=np.float32)
    sc_filter = np.zeros((height, width), dtype=np.float32)

    process_pixels(image, (mean_filter, std_dev_filter, sc_filter), kernel_size, pixels_to_process, 
                   height, width, first_x, first_y, speckle_calculation)

    return (mean_filter, std_dev_filter, sc_filter, first_x, first_y, 
            std_dev_filter[first_y, first_x], mean_filter[first_y, first_x], sc_filter[first_y, first_x])


# ---------------------------- Utility Functions ---------------------------- #

def calculate_valid_pixel_range(height: int, width: int, kernel_size: int, max_pixels: int) -> Tuple[int, int, int]:
    half_kernel = kernel_size // 2
    first_x = first_y = half_kernel
    valid_height = height - kernel_size + 1
    valid_width = width - kernel_size + 1
    total_valid_pixels = valid_height * valid_width
    pixels_to_process = min(max_pixels, total_valid_pixels)
    return first_x, first_y, pixels_to_process

def draw_kernel_overlay(ax: plt.Axes, x: int, y: int, kernel_size: int):
    kx, ky = int(x - kernel_size // 2), int(y - kernel_size // 2)
    ax.add_patch(plt.Rectangle((kx - 0.5, ky - 0.5), kernel_size, kernel_size, 
                               edgecolor="r", linewidth=1, facecolor="none"))
    lines = ([[(kx + i - 0.5, ky - 0.5), (kx + i - 0.5, ky + kernel_size - 0.5)] for i in range(1, kernel_size)] +
             [[(kx - 0.5, ky + i - 0.5), (kx + kernel_size - 0.5, ky + i - 0.5)] for i in range(1, kernel_size)])
    ax.add_collection(LineCollection(lines, colors='red', linestyles=':', linewidths=0.5))

def draw_search_window_overlay(ax: plt.Axes, image: np.ndarray, x: int, y: int, search_window: Optional[Union[str, int]]):
    if search_window == "full":
        rect = plt.Rectangle((-0.5, -0.5), image.shape[1], image.shape[0], 
                             edgecolor="blue", linewidth=2, facecolor="none")
        ax.add_patch(rect)
    elif isinstance(search_window, int):
        half_window = search_window // 2
        rect = plt.Rectangle((x - half_window - 0.5, y - half_window - 0.5), 
                             search_window, search_window, 
                             edgecolor="blue", linewidth=1, facecolor="none")
        ax.add_patch(rect)

def draw_value_annotations(ax: plt.Axes, image: np.ndarray):
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            ax.text(j, i, f"{image[i, j]:.2f}", ha="center", va="center", color="red", fontsize=8)

def display_processing_statistics(analysis_params):
    with st.expander("Processing Statistics", expanded=False):
        st.write(f"Image size: {analysis_params['width']}x{analysis_params['height']}")
        st.write(f"Kernel size: {analysis_params['kernel_size']}x{analysis_params['kernel_size']}")  
        st.write(f"Max processable pixels: {analysis_params['max_pixels']}")
        st.write(f"Actual processed pixels: {min(analysis_params['max_pixels'], (analysis_params['width'] - analysis_params['kernel_size'] + 1) * (analysis_params['height'] - analysis_params['kernel_size'] + 1))}")
        st.write(f"Processing technique: {analysis_params['technique']}")

def get_search_window_size(use_full_image: bool, kernel_size: int, image: Image.Image) -> Optional[int]:
    if not use_full_image:
        return st.number_input("Search Window Size", 
                               min_value=kernel_size + 2, 
                               max_value=min(max(image.width, image.height) // 2, 35),
                               value=kernel_size + 2,
                               step=2,
                               help="Size of the search window for NL-Means denoising")
    return None



def update_current_position():
    time.sleep(0.1)  # Add a small delay
    st.session_state.current_position = st.session_state.get('current_position', 1)


# ---------------------------- Cached Functions ---------------------------- #

@st.cache_data(persist=True)
def create_combined_plot(plot_image: np.ndarray, plot_x: int, plot_y: int, plot_kernel_size: int, 
                         title: str, plot_cmap: str = "viridis", plot_search_window: Optional[Union[str, int]] = None, 
                         zoom: bool = False, vmin: Optional[float] = None, vmax: Optional[float] = None) -> plt.Figure:
    fig, ax = plt.subplots(1, 1)
    
    ax.imshow(plot_image, cmap=plot_cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.axis('off')

    if not zoom:
        ax.add_patch(plt.Rectangle((plot_x - 0.5, plot_y - 0.5), 1, 1, 
                                   edgecolor="r", linewidth=0.5, facecolor="r", alpha=0.2))
        
        if title == "Original Image with Current Kernel":
            draw_kernel_overlay(ax, plot_x, plot_y, plot_kernel_size)
            
        draw_search_window_overlay(ax, plot_image, plot_x, plot_y, plot_search_window)
    else:
        draw_value_annotations(ax, plot_image)

    fig.tight_layout(pad=2)
    return fig

@st.cache_data(persist=True)
def process_image(technique: str, image: np.ndarray, kernel_size: int, max_pixels: int, height: int, width: int, 
                  search_window_size: Optional[int], filter_strength: float) -> Tuple[np.ndarray, ...]:
    try:
        first_x, first_y, pixels_to_process = calculate_valid_pixel_range(height, width, kernel_size, max_pixels)

        if technique == "speckle":
            result = process_speckle(image, kernel_size, pixels_to_process, height, width, first_x, first_y)
            if len(result) != 8:  # Assuming process_speckle should return 8 values
                raise ValueError(f"process_speckle returned {len(result)} values instead of 8")
            return result
        elif technique == "nlm":
            denoised_image = np.zeros((height, width), dtype=np.float32)
            weight_sum_map = np.zeros((height, width), dtype=np.float32)

            @jit(nopython=True)
            def nlm_calculation(center_patch, comparison_patch, filter_strength):
                distance = np.sum((center_patch - comparison_patch)**2)
                return np.exp(-distance / (filter_strength ** 2))

            result = process_nlm(image, denoised_image, weight_sum_map, kernel_size, search_window_size, filter_strength, 
                                 pixels_to_process, height, width, first_x, first_y, nlm_calculation)
            if len(result) != 5:  # Assuming process_nlm should return 5 values
                raise ValueError(f"process_nlm returned {len(result)} values instead of 5")
            return result
        else:
            raise ValueError(f"Unknown technique: {technique}")
    except Exception as e:
        st.error(f"Error in process_image: {str(e)}")
        st.error(f"Technique: {technique}, Kernel Size: {kernel_size}, Max Pixels: {max_pixels}")
        st.error(f"Image Shape: {image.shape}, Search Window Size: {search_window_size}")
        raise  # Re-raise the exception for further debugging

@st.cache_data
def load_image(image_source: str, selected_image: Optional[str] = None, uploaded_file: Any = None) -> Image.Image:
    if image_source == "Preloaded Image" and selected_image:
        return Image.open(PRELOADED_IMAGES[selected_image]).convert('L')
    elif image_source == "Upload Image" and uploaded_file:
        return Image.open(uploaded_file).convert('L')
    st.warning('Please upload or select an image.')
    st.stop()

# ---------------------------- Main Processing Functions ---------------------------- #

@st.cache_data(persist=True)
def calculate_nlm(image: np.ndarray, kernel_size: int, search_size: Optional[int], filter_strength: float, 
                  pixels_to_process: int, height: int, width: int, first_x: int, first_y: int) -> Tuple[np.ndarray, np.ndarray, int, int, float]:
    """
    Calculates Non-Local Means denoising for the given image.
    
    The Non-Local Means algorithm replaces each pixel with a weighted average of all pixels in the image, 
    where the weights are based on the similarity of the local neighborhoods around the pixels.
    """
    denoised_image = np.zeros((height, width), dtype=np.float32)
    weight_sum_map = np.zeros((height, width), dtype=np.float32)

    @jit(nopython=True)
    def nlm_calculation(center_patch, comparison_patch, filter_strength):
        distance = np.sum((center_patch - comparison_patch)**2)
        return np.exp(-distance / (filter_strength ** 2))

    process_nlm(image, denoised_image, weight_sum_map, kernel_size, search_size, filter_strength, pixels_to_process, 
                height, width, first_x, first_y, nlm_calculation)

    max_weight = np.max(weight_sum_map)
    normalized_weight_map = weight_sum_map / max_weight if max_weight > 0 else weight_sum_map

    return denoised_image, normalized_weight_map, first_x, first_y, weight_sum_map[first_y, first_x]

@st.cache_data(persist=True)
def calculate_speckle(image: np.ndarray, kernel_size: int, pixels_to_process: int, height: int, width: int, 
                      first_x: int, first_y: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, int, float, float, float]:
    """
    Calculates speckle statistics for the given image.
    
    For each pixel, it calculates the local mean, standard deviation, and speckle contrast (ratio of std to mean)
    within a neighborhood defined by the kernel size.
    """
    mean_filter = np.zeros((height, width), dtype=np.float32)
    std_dev_filter = np.zeros((height, width), dtype=np.float32)
    sc_filter = np.zeros((height, width), dtype=np.float32)

    @jit(nopython=True)
    def speckle_calculation(local_window):
        local_mean = np.mean(local_window)
        local_std = np.std(local_window)
        speckle_contrast = local_std / local_mean if local_mean != 0 else 0
        return local_mean, local_std, speckle_contrast

    process_speckle(image, mean_filter, std_dev_filter, sc_filter, kernel_size, pixels_to_process, height, width, 
                    first_x, first_y, speckle_calculation)

    return mean_filter, std_dev_filter, sc_filter, first_x, first_y, mean_filter[first_y, first_x], std_dev_filter[first_y, first_x], sc_filter[first_y, first_x]

@jit(nopython=True)
def speckle_calculation(local_window):
    local_mean = np.mean(local_window)
    local_std = np.std(local_window)
    speckle_contrast = local_std / local_mean if local_mean != 0 else 0
    return local_mean, local_std, speckle_contrast

def create_analysis_params(sidebar_params):
    return {
        "image_np": sidebar_params['image_np'],
        "kernel_size": sidebar_params['kernel_size'],
        "search_window_size": sidebar_params['search_window_size'],
        "filter_strength": sidebar_params['filter_strength'],
        "cmap": sidebar_params['cmap'],
        "max_pixels": sidebar_params['max_pixels'],
        "height": sidebar_params['image_np'].shape[0],
        "width": sidebar_params['image_np'].shape[1],
        "show_full_processed": sidebar_params['show_full_processed'],
        "technique": sidebar_params['technique']
    }

# ---------------------------- Image Processing Functions ---------------------------- #

def handle_animation(animation_params: Dict[str, Any], analysis_params: Dict[str, Any]):
    if animation_params['play_pause']:
        st.session_state.animate = not st.session_state.get('animate', False)
    
    if animation_params['reset']:
        st.session_state.current_position = 1
        st.session_state.animate = False

    if st.session_state.get('animate', False):
        for i in range(st.session_state.current_position, analysis_params['max_pixels'] + 1):
            st.session_state.current_position = i
            update_images_analyze_and_visualize(
                image_np=analysis_params['image_np'],
                kernel_size=analysis_params['kernel_size'],
                cmap=analysis_params['cmap'],
                technique=analysis_params['technique'],
                search_window_size=analysis_params['search_window_size'],
                filter_strength=analysis_params['filter_strength'],
                show_full_processed=analysis_params['show_full_processed'],
                update_state=True,
                handle_visualization=True
            )
            
            time.sleep(0.01)
            if not st.session_state.get('animate', False):
                break

def handle_animation_if_needed(sidebar_params, analysis_params):
    if not sidebar_params['show_full_processed']:
        handle_animation(sidebar_params['animation_params'], analysis_params)

def update_pixels(key: str) -> None:
    if key == 'slider':
        st.session_state.pixels_input = st.session_state.pixels_slider
    else:
        st.session_state.pixels_slider = st.session_state.pixels_input
    st.session_state.current_position = st.session_state.pixels_slider
    update_images_analyze_and_visualize(
        image_np=st.session_state.get('image_np'),
        kernel_size=st.session_state.get('kernel_size'),
        cmap=st.session_state.get('cmap'),
        technique=st.session_state.get('technique'),
        search_window_size=st.session_state.get('search_window_size'),
        filter_strength=st.session_state.get('filter_strength'),
        show_full_processed=st.session_state.get('show_full_processed', False),
        update_state=True,
        handle_visualization=True
    )

def prepare_comparison_images():
    speckle_results = st.session_state.get("speckle_results")
    nlm_results = st.session_state.get("nlm_results")
    analysis_params = st.session_state.analysis_params

    if speckle_results is not None and nlm_results is not None:
        return {
            'Unprocessed Image': analysis_params['image_np'],
            'Standard Deviation': speckle_results[1],
            'Speckle Contrast': speckle_results[2],
            'Mean Filter': speckle_results[0],
            'NL-Means Image': nlm_results[0]
        }
    else:
        return None

def update_images_analyze_and_visualize(
    image_np: np.ndarray,
    kernel_size: int,
    cmap: str,
    technique: str,
    search_window_size: Optional[int],
    filter_strength: float,
    show_full_processed: bool,
    update_state: bool = True,
    handle_visualization: bool = True,
    height: Optional[int] = None,
    width: Optional[int] = None
) -> Tuple[Dict[str, Any], Optional[Tuple[np.ndarray, ...]]]:
    
    if update_state:
        update_current_position()
    
    params = prepare_params(image_np, kernel_size, cmap, search_window_size, filter_strength, show_full_processed)
    placeholders = get_placeholders(technique)
    
    results = None
    
    if handle_visualization:
        try:
            height, width = get_image_dimensions(image_np, height, width)
            pixels_to_process, last_x, last_y = calculate_processing_details(height, width, kernel_size, params['analysis_params']['max_pixels'])
            
            results = process_image(technique, image_np, kernel_size, pixels_to_process, height, width, 
                                    search_window_size, filter_strength)
            
            kernel_matrix, original_value = extract_kernel_info(image_np, last_x, last_y, kernel_size)
            
            visualize_results(image_np, results, technique, placeholders, params, last_x, last_y, kernel_size, kernel_matrix, original_value)
            
        except (ValueError, IndexError) as e:
            st.error(f"Error during image analysis and visualization: {str(e)}")
            return None, None
    
    if update_state:
        update_results_in_session_state(technique, params, results)
    
    return params, results

def prepare_params(image_np, kernel_size, cmap, search_window_size, filter_strength, show_full_processed):
    return {
        "tabs": st.session_state.get('tabs', []),
        "analysis_params": {
            "image_np": image_np,
            "kernel_size": kernel_size,
            "max_pixels": st.session_state.current_position,
            "cmap": cmap,
            "search_window_size": search_window_size,
            "filter_strength": filter_strength
        },
        "show_full_processed": show_full_processed
    }

def get_placeholders(technique):
    return {
        "speckle": st.session_state.get('speckle_placeholders', {}),
        "nlm": st.session_state.get('nlm_placeholders', {})
    }[technique]

def get_image_dimensions(image_np, height, width):
    return height or image_np.shape[0], width or image_np.shape[1]

def calculate_processing_details(height, width, kernel_size, max_pixels):
    valid_height = height - kernel_size + 1
    valid_width = width - kernel_size + 1
    total_valid_pixels = valid_height * valid_width
    pixels_to_process = min(max_pixels, total_valid_pixels)
    
    last_pixel = pixels_to_process - 1
    last_y = (last_pixel // valid_width) + kernel_size // 2
    last_x = (last_pixel % valid_width) + kernel_size // 2
    
    return pixels_to_process, int(last_x), int(last_y)

def extract_kernel_info(image_np, last_x, last_y, kernel_size):
    half_kernel = kernel_size // 2
    height, width = image_np.shape

    y_start = max(0, last_y - half_kernel)
    y_end = min(height, last_y + half_kernel + 1)
    x_start = max(0, last_x - half_kernel)
    x_end = min(width, last_x + half_kernel + 1)

    kernel_values = image_np[y_start:y_end, x_start:x_end]
    
    if kernel_values.size == 0:
        raise ValueError(f"Extracted kernel at ({last_x}, {last_y}) is empty. Image shape: {image_np.shape}, Kernel size: {kernel_size}")

    kernel_values = pad_kernel_if_necessary(kernel_values, kernel_size, last_x, last_y, height, width)
    kernel_matrix = [[float(kernel_values[i, j]) for j in range(kernel_size)] for i in range(kernel_size)]
    original_value = float(image_np[last_y, last_x])

    return kernel_matrix, original_value

def pad_kernel_if_necessary(kernel_values, kernel_size, last_x, last_y, height, width):
    if kernel_values.shape != (kernel_size, kernel_size):
        half_kernel = kernel_size // 2
        pad_top = max(0, half_kernel - last_y)
        pad_bottom = max(0, last_y + half_kernel + 1 - height)
        pad_left = max(0, half_kernel - last_x)
        pad_right = max(0, last_x + half_kernel + 1 - width)
        return np.pad(kernel_values, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='edge')
    return kernel_values

def visualize_results(image_np, results, technique, placeholders, params, last_x, last_y, kernel_size, kernel_matrix, original_value):
    vmin, vmax = np.min(image_np), np.max(image_np)
    show_full_processed = params['show_full_processed']
    cmap = params['analysis_params']['cmap']
    search_window_size = params['analysis_params']['search_window_size']

    visualize_original_image(image_np, placeholders, last_x, last_y, kernel_size, cmap, show_full_processed, vmin, vmax, technique, search_window_size)
    
    filter_options, specific_params = prepare_filter_options_and_params(technique, results, last_x, last_y, params['analysis_params']['filter_strength'], search_window_size)
    
    visualize_filter_results(filter_options, placeholders, last_x, last_y, kernel_size, cmap, show_full_processed)
    
    # Update the specific_params dictionary with additional parameters
    specific_params.update({
        'x': last_x,
        'y': last_y,
        'input_x': last_x,
        'input_y': last_y,
        'kernel_size': kernel_size,
        'kernel_matrix': kernel_matrix,
        'original_value': original_value
    })

    # Call display_formula with the updated specific_params
    display_formula(placeholders['formula'], technique, **specific_params)

    plt.close('all')
    
def visualize_original_image(image_np, placeholders, last_x, last_y, kernel_size, cmap, show_full_processed, vmin, vmax, technique, search_window_size):
    if show_full_processed:
        fig_original = plt.figure()
        plt.imshow(image_np, cmap=cmap, vmin=vmin, vmax=vmax)
        plt.axis('off')
        plt.title("Original Image")
    else:
        fig_original = create_combined_plot(image_np, last_x, last_y, kernel_size, "Original Image with Current Kernel", cmap, 
                                            search_window_size if technique == "nlm" else None, vmin=vmin, vmax=vmax)
    
    placeholders['original_image'].pyplot(fig_original)

    if not show_full_processed:
        visualize_zoomed_original(image_np, placeholders, last_x, last_y, kernel_size, cmap, vmin, vmax)

def visualize_zoomed_original(image_np, placeholders, last_x, last_y, kernel_size, cmap, vmin, vmax):
    zoom_size = kernel_size
    ky = int(max(0, last_y - zoom_size // 2))
    kx = int(max(0, last_x - zoom_size // 2))
    zoomed_original = image_np[ky:min(image_np.shape[0], ky + zoom_size),
                               kx:min(image_np.shape[1], kx + zoom_size)]
    fig_zoom_original = create_combined_plot(zoomed_original, zoom_size // 2, zoom_size // 2, zoom_size, 
                                           "Zoomed-In Original Image", cmap, zoom=True, vmin=vmin, vmax=vmax)
    if 'zoomed_original_image' in placeholders and placeholders['zoomed_original_image'] is not None:
        placeholders['zoomed_original_image'].pyplot(fig_zoom_original)

def prepare_filter_options_and_params(technique, results, last_x, last_y, filter_strength, search_window_size):
    if technique == "speckle":
        return {
            "Mean Filter": results[0],
            "Std Dev Filter": results[1],
            "Speckle Contrast": results[2]
        }, {
            "std": results[5],
            "mean": results[6],
            "sc": results[7],
            "total_pixels": results[3] * results[4]
        }
    elif technique == "nlm":
        denoised_image, weight_sum_map = results[:2]
        return {
            "NL-Means Image": denoised_image,
            "Weight Map": weight_sum_map,
            "Difference Map": np.abs(denoised_image - results[2])
        }, {
            "filter_strength": filter_strength,
            "search_size": search_window_size,
            "total_pixels": results[3] * results[4],
            "nlm_value": denoised_image[last_y, last_x]
        }
    else:
        return {}, {}

def visualize_filter_results(filter_options, placeholders, last_x, last_y, kernel_size, cmap, show_full_processed):
    for filter_name, filter_data in filter_options.items():
        key = filter_name.lower().replace(" ", "_")
        if key in placeholders and placeholders[key] is not None:
            filter_vmin, filter_vmax = np.min(filter_data), np.max(filter_data)
            
            if show_full_processed:
                fig_full = plt.figure()
                plt.imshow(filter_data, cmap=cmap, vmin=filter_vmin, vmax=filter_vmax)
                plt.axis('off')
                plt.title(filter_name)
            else:
                fig_full = create_combined_plot(filter_data, last_x, last_y, kernel_size, filter_name, cmap, vmin=filter_vmin, vmax=filter_vmax)
            placeholders[key].pyplot(fig_full)

            if not show_full_processed:
                visualize_zoomed_filter(filter_data, placeholders, last_x, last_y, kernel_size, cmap, filter_name, filter_vmin, filter_vmax)

def visualize_zoomed_filter(filter_data, placeholders, last_x, last_y, kernel_size, cmap, filter_name, filter_vmin, filter_vmax):
    zoom_size = kernel_size
    ky = int(max(0, last_y - zoom_size // 2))
    kx = int(max(0, last_x - zoom_size // 2))
    zoomed_data = filter_data[ky:min(filter_data.shape[0], ky + zoom_size),
                              kx:min(filter_data.shape[1], kx + zoom_size)]
    fig_zoom = create_combined_plot(zoomed_data, zoom_size // 2, zoom_size // 2, zoom_size, 
                                    f"Zoomed-In {filter_name}", cmap, zoom=True, vmin=filter_vmin, vmax=filter_vmax)
    
    zoomed_key = f'zoomed_{filter_name.lower().replace(" ", "_")}'
    if zoomed_key in placeholders and placeholders[zoomed_key] is not None:
        placeholders[zoomed_key].pyplot(fig_zoom)

def update_results_in_session_state(technique, params, results):
    st.session_state.processed_pixels = params['analysis_params']['max_pixels']
    if technique == "speckle":
        st.session_state.speckle_results = results
    elif technique == "nlm":
        st.session_state.nlm_results = results

# ---------------------------- Formula Display ---------------------------- #
# Define formulas and explanations for each technique
FORMULA_CONFIG = {
    "speckle": {
        "main_formula": r"I_{{{input_x},{input_y}}} = {original_value:.3f} \quad \rightarrow \quad SC_{{{output_x},{output_y}}} = \frac{{\sigma}}{{\mu}} = \frac{{{std:.3f}}}{{{mean:.3f}}} = {sc:.3f}",
        "explanation": "This formula shows the transition from the original pixel intensity I({input_x},{input_y}) in the input image to the Speckle Contrast (SC) for pixel ({output_x},{output_y}) in the output image.",
        "variables": {},  # To be filled dynamically
        "additional_formulas": [
            {
                "title": "Kernel Details",
                "formula": r"\text{{Kernel Size: }} {kernel_size} \times {kernel_size}"
                           r"\qquad\quad\text{{Centered at pixel: }}({x}, {y})"
                           r"\\\\"
                           "{kernel_matrix}",
                "explanation": "The speckle contrast is calculated using a {kernel_size}x{kernel_size} kernel centered around the pixel ({x},{y}). This matrix shows the pixel values in the kernel. The central value (in bold) corresponds to the pixel ({x},{y})."
            },
            {
                "title": "Mean Calculation",
                "formula": r"\mu = \frac{{1}}{{N}} \sum_{{i,j \in K}} I_{{i,j}} \quad \rightarrow \quad \mu = \frac{{1}}{{{total_pixels}}} \sum_{{i,j \in K}} I_{{i,j}} = {mean:.3f}",
                "explanation": "The mean (μ) is calculated as the average intensity of all pixels in the kernel K, where N is the total number of pixels in the kernel (N = {kernel_size}^2 = {total_pixels})."
            },
            {
                "title": "Standard Deviation Calculation",
                "formula": r"\sigma = \sqrt{{\frac{{1}}{{N}} \sum_{{i,j \in K}} (I_{{i,j}} - \mu)^2}} \quad \rightarrow \quad \sigma = \sqrt{{\frac{{1}}{{{total_pixels}}} \sum_{{i,j \in K}} (I_{{i,j}} - {mean:.3f})^2}} = {std:.3f}",
                "explanation": "The standard deviation (σ) is calculated using all pixels in the kernel K, measuring the spread of intensities around the mean."
            },
            {
                "title": "Speckle Contrast Calculation",
                "formula": r"SC = \frac{{\sigma}}{{\mu}} \quad \rightarrow \quad SC = \frac{{{std:.3f}}}{{{mean:.3f}}} = {sc:.3f}",
                "explanation": "The Speckle Contrast (SC) is the ratio of the standard deviation to the mean intensity within the kernel."
            }
        ]
    },
    "nlm": {
        "main_formula": r"I_{{{x},{y}}} = {original_value:.3f} \quad \rightarrow \quad NLM_{{{x},{y}}} = \frac{{1}}{{W(x,y)}} \sum_{{(i,j) \in \Omega}} I_{{i,j}} \cdot w\Bigg((x,y), (i,j)\Bigg) = {nlm_value:.3f}",
        "explanation": """
        This formula shows the transition from the original pixel intensity I({x},{y}) to the denoised value NLM({x},{y}) using the Non-Local Means (NLM) algorithm:
        
        1. For each pixel (x,y), we look at a small patch around it.
        2. We search for similar patches in a larger window Ω (size: {search_size}x{search_size}).
        3. We calculate a weighted average of pixels with similar surrounding patches.
        4. Pixels with more similar patches get higher weights in the average.
        
        Key components:
        - I(i,j): Intensity of pixel (i,j) in the original image
        - w((x,y), (i,j)): Weight assigned to pixel (i,j) when denoising (x,y)
        - W(x,y): Normalization factor (sum of all weights)
        - Ω: Search window where we look for similar patches
        """,
        "variables": {},  # To be filled dynamically
        "additional_formulas": [
            {
                "title": "Patch Comparison",
                "formula": r"\text{{Patch Size: }} {kernel_size} \times {kernel_size}"
                           r"\quad\quad\text{{Centered at pixel: }}({x}, {y})"
                           r"\\\\"
                           "{kernel_matrix}",
                "explanation": "We compare {kernel_size}x{kernel_size} patches centered around each pixel. This matrix shows the pixel values in the patch centered at ({x},{y}). The central value (in bold) is the pixel being denoised."
            },
            {
                "title": "Weight Calculation",
                "formula": r"\textbf{{w}}_{{\textbf{{x}},\textbf{{y}}}}\big((x,y), (i,j)\big) = \exp\left(-\frac{{\Bigg\|\Big(P(x,y) - P(i,j)\Big)\Bigg\|^2}}{{h^2}}\right)",
                "explanation": r"""
                This formula determines how much each pixel (i,j) contributes to denoising pixel (x,y):
                
                - \textbf{{w}}_{{\textbf{{x}},\textbf{{y}}}}\big((x,y), (i,j)\big): Weight assigned to pixel (i,j) when denoising (x,y)
                - P(x,y) and P(i,j) are patches centered at (x,y) and (i,j)
                - \Bigg\|\Big(P(x,y) - P(i,j)\Big)\Bigg\|^2 measures how different the patches are
                - h = {filter_strength} controls the smoothing strength
                - More similar patches result in higher weights
                """
            },
            {
                "title": "Normalization Factor",
                "formula": r"W(x,y) = \sum_{{(i,j) \in \Omega}} w((x,y), (i,j))",
                "explanation": "We sum all weights for pixel (x,y). This ensures the final weighted average preserves the overall image brightness."
            },
            {
                "title": "Search Window",
                "formula": r"\Omega = \begin{{cases}} \text{{Full Image}} & \text{{if search_size = 'full'}} \\ {search_size} \times {search_size} \text{{ window}} & \text{{otherwise}} \end{{cases}}",
                "explanation": "The search window Ω is where we look for similar patches. {search_window_description}"
            },
            {
                "title": "Visual Representation",
                "formula": r"\text{{[Insert diagram here]}}", # Replace with actual diagram
                "explanation": """
                This diagram illustrates the NLM process:
                1. The central pixel being denoised (x,y)
                2. Its surrounding patch P(x,y)
                3. The larger search window Ω
                4. Example similar patches within Ω
                """
            }
        ]
    }
}

def update_variables(config: Dict[str, Any], **kwargs):
    variables = kwargs.copy()
    original_x = kwargs['x'] - kwargs['kernel_size'] // 2
    original_y = kwargs['y'] - kwargs['kernel_size'] // 2
    variables.update({
        'input_x': original_x,
        'input_y': original_y,
        'output_x': kwargs['x'],
        'output_y': kwargs['y']
    })
    return variables

def generate_kernel_matrix(kernel_size: int, kernel_matrix: List[List[float]]) -> str:
    center = kernel_size // 2
    center_value = kernel_matrix[center][center]
    
    matrix_rows = []
    for i in range(kernel_size):
        row = [r"\mathbf{{{:.3f}}}".format(center_value) if i == center and j == center 
               else r"{:.3f}".format(kernel_matrix[i][j]) for j in range(kernel_size)]
        matrix_rows.append(" & ".join(row))

    return (r"\def\arraystretch{1.5}\begin{array}{|" + ":".join(["c"] * kernel_size) + "|}" +
            r"\hline" + r"\\ \hdashline".join(matrix_rows) + r"\\ \hline\end{array}")

def add_search_window_description(variables: Dict[str, Any], search_size: str):
    variables['search_window_description'] = (
        "We search the entire image for similar pixels." if search_size == "full" 
        else f"A search window of size {search_size}x{search_size} centered around the target pixel."
    )

def display_formula(formula_placeholder: Any, technique: str, **kwargs):
    with formula_placeholder.container():
        if technique not in FORMULA_CONFIG:
            st.error(f"Unknown technique: {technique}")
            return

        config = FORMULA_CONFIG[technique]
        variables = update_variables(config, **kwargs)

        if 'kernel_size' in kwargs and 'kernel_matrix' in kwargs:
            variables['kernel_matrix'] = generate_kernel_matrix(kwargs['kernel_size'], kwargs['kernel_matrix'])

        if technique == "nlm":
            add_search_window_description(variables, kwargs.get('search_size', 'full'))

        display_main_formula(config, variables)
        display_additional_formulas(config, variables)

def display_main_formula(config: Dict[str, Any], variables: Dict[str, Any]):
    try:
        st.latex(config['main_formula'].format(**variables))
        st.markdown(config['explanation'].format(**variables))
    except KeyError as e:
        st.error(f"Missing key in main formula or explanation: {e}")

def display_additional_formulas(config: Dict[str, Any], variables: Dict[str, Any]):
    with st.expander("Additional Formulas", expanded=False):
        for additional_formula in config['additional_formulas']:
            with st.expander(additional_formula['title'], expanded=False):
                try:
                    st.latex(additional_formula['formula'].format(**variables))
                    st.markdown(additional_formula['explanation'].format(**variables))
                except KeyError as e:
                    st.error(f"Missing key in additional formula: {e}")
