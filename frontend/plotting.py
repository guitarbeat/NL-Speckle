import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from analysis.speckle import process_speckle, SpeckleResult
from analysis.nlm import process_nlm, NLMResult
from frontend.formula import display_analysis_formula
from shared_types import VisualizationConfig, FilterResult, VisualizationParams, ProcessParams, calculate_processing_details, ImageArray, PixelCoordinates
import logging

# Constants 
KERNEL_OUTLINE_COLOR, SEARCH_WINDOW_OUTLINE_COLOR, PIXEL_VALUE_TEXT_COLOR = 'red', 'blue', 'red'
ZOOMED_IMAGE_DIMENSIONS = (8, 8) 

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Image Processing Functions
def normalize_image(image: ImageArray) -> ImageArray:
    p_low, p_high = np.percentile(image, [2, 98])
    return (np.clip(image, p_low, p_high) - p_low) / (p_high - p_low)

def process_image(params: ProcessParams):
    technique, analysis_params = params.technique, params.analysis_params
    kernel_size = st.session_state.get('kernel_size', 3)
    analysis_params.update({'kernel_size': kernel_size, 'pixels_to_process': analysis_params.get('pixels_to_process', 0)})
    
    normalized_image = normalize_image(params.image_array) if analysis_params.get('normalization_option') == 'Percentile' else params.image_array
    
    results = (process_nlm(normalized_image, kernel_size, analysis_params['pixels_to_process'], 
                           analysis_params.get('search_window_size', 0), analysis_params.get('filter_strength', 0.1)) 
               if technique == "nlm" else 
               process_speckle(normalized_image, kernel_size, analysis_params['pixels_to_process']) 
               if technique == "speckle" else 
               None)
    
    if results is None:
        raise ValueError(f"Unknown technique: {technique}")
    
    params.handle_visualization and visualize_results(normalized_image, technique, analysis_params, results, params.show_per_pixel_processing)
    
    if params.update_state:
        st.session_state.update({
            'processed_pixels': analysis_params['pixels_to_process'],
            f"{technique}_results": results
        })
    
    return params, results

def extract_kernel_from_image(image_array: ImageArray, end_x: int, end_y: int, kernel_size: int):
    half_kernel = kernel_size // 2
    height, width = image_array.shape
    y_start, y_end = max(0, end_y - half_kernel), min(height, end_y + half_kernel + 1)
    x_start, x_end = max(0, end_x - half_kernel), min(width, end_x + half_kernel + 1)
    kernel_values = image_array[y_start:y_end, x_start:x_end]
    
    if kernel_values.size == 0:
        raise ValueError(f"Extracted kernel at ({end_x}, {end_y}) is empty. Image shape: {image_array.shape}, Kernel size: {kernel_size}")
    
    if kernel_values.shape != (kernel_size, kernel_size):
        pad_width = [
            (max(0, half_kernel - end_y), max(0, end_y + half_kernel + 1 - height)),
            (max(0, half_kernel - end_x), max(0, end_x + half_kernel + 1 - width))
        ]
        kernel_values = np.pad(kernel_values, pad_width, mode='edge')
    
    return kernel_values.astype(float), float(image_array[end_y, end_x]), kernel_size

# UI Functions
def create_technique_ui_elements(technique: str, tab, show_per_pixel_processing: bool):
    with tab:
        ui_placeholders = {'formula': st.empty(), 'original_image': st.empty()}
        all_filter_options = ['Original Image'] + (SpeckleResult.get_filter_options() if technique == "speckle" else NLMResult.get_filter_options() if technique == "nlm" else [])
        selected_filters = st.multiselect(
            "Select views to display",
            all_filter_options,
            default=all_filter_options,
            key=f"{technique}_filter_selection"
        )
        st.session_state[f'{technique}_selected_filters'] = selected_filters
        if selected_filters:
            columns = st.columns(len(selected_filters))
            ui_placeholders |= {
                filter_name.lower().replace(" ", "_"): columns[i].empty()
                for i, filter_name in enumerate(selected_filters)
            }
            if show_per_pixel_processing:
                ui_placeholders |= {
                    f'zoomed_{filter_name.lower().replace(" ", "_")}': columns[
                        i
                    ]
                    .expander(f"Zoomed-in {filter_name}", expanded=False)
                    .empty()
                    for i, filter_name in enumerate(selected_filters)
                }
        else:
            st.warning("No views selected. Please select at least one view to display.")
        if show_per_pixel_processing:
            ui_placeholders['zoomed_kernel'] = st.empty()

    return ui_placeholders

def create_process_params(analysis_params: dict, technique: str, technique_params: dict):
    return ProcessParams(
        image_array=analysis_params.get('image_array', np.array([])),
        technique=technique,
        analysis_params={
            **technique_params,
            'kernel_size': st.session_state.get('kernel_size', 3),
            'pixels_to_process': analysis_params.get('pixels_to_process', 0),
            'total_pixels': analysis_params.get('total_pixels', 0),
            'show_per_pixel_processing': analysis_params.get('show_per_pixel_processing', False),
            'search_window_size': technique_params.get('search_window_size'),
            'filter_strength': technique_params.get('filter_strength'),
        },
        update_state=True,
        handle_visualization=True,
        show_per_pixel_processing=analysis_params.get('show_per_pixel_processing', False)
    )

# Visualization Functions
def visualize_results(image_array: ImageArray, technique: str, analysis_params: dict, results: FilterResult, show_per_pixel_processing: bool):
    processing_details = calculate_processing_details(
        image_array,
        analysis_params.get('kernel_size', 3),
        analysis_params.get('total_pixels', 0)  
    )
    try:
        last_processed_x, last_processed_y = results.processing_end_coord if isinstance(results, (NLMResult, SpeckleResult)) else (processing_details.end_x, processing_details.end_y)
        kernel_matrix, original_pixel_value, kernel_size = extract_kernel_from_image(image_array, last_processed_x, last_processed_y, analysis_params.get('kernel_size', 3))

        viz_params = VisualizationParams(
            image_array=image_array,
            results=results,
            ui_placeholders=st.session_state.get(f'{technique}_placeholders', {}),
            analysis_params=analysis_params,  
            last_processed_pixel=(last_processed_x, last_processed_y),
            kernel_size=kernel_size,
            kernel_matrix=kernel_matrix,
            original_pixel_value=original_pixel_value,
            analysis_type=technique,
            search_window_size=analysis_params.get('search_window_size'),
            color_map=st.session_state.get('color_map', 'gray'),
            show_per_pixel_processing=show_per_pixel_processing
        )
        visualize_analysis_results(viz_params)
    except Exception as e:
        logger.error(f"Error while visualizing results: {e}", exc_info=True)
        st.error("An error occurred while visualizing the results. Please check the logs.")

def visualize_analysis_results(viz_params: VisualizationParams):
    try:
        last_processed_x, last_processed_y = viz_params.last_processed_pixel
        
        if viz_params.results is None:
            logger.warning("Results are None. Skipping visualization.")
            return

        filter_options, specific_params = prepare_filter_options_and_parameters(viz_params.results, (last_processed_x, last_processed_y))
        filter_options['Original Image'] = viz_params.image_array

        for filter_name, filter_data in filter_options.items():
            visualize_filter_and_zoomed(filter_name, filter_data, viz_params)

        if viz_params.show_per_pixel_processing:
            display_analysis_formula(specific_params, viz_params.ui_placeholders, viz_params.analysis_type, last_processed_x, last_processed_y, viz_params.kernel_size, viz_params.kernel_matrix, viz_params.original_pixel_value)
    except Exception as e:
        logger.error(f"Error while visualizing analysis results: {e}", exc_info=True)
        raise

def visualize_filter_and_zoomed(filter_name: str, filter_data: ImageArray, viz_params: VisualizationParams):
    placeholder_key = filter_name.lower().replace(" ", "_")
    
    for plot_type in ['main', 'zoomed']:
        plot_key = f'zoomed_{placeholder_key}' if plot_type == 'zoomed' else placeholder_key
        
        if plot_key not in viz_params.ui_placeholders or (plot_type == 'zoomed' and not viz_params.show_per_pixel_processing):
            continue
        
        config = VisualizationConfig(
            vmin=None if filter_name == 'Original Image' else np.min(filter_data),
            vmax=None if filter_name == 'Original Image' else np.max(filter_data),
            zoom=plot_type == 'zoomed',
            show_kernel=viz_params.show_per_pixel_processing if plot_type == 'main' else True,
            show_per_pixel_processing=plot_type == 'zoomed',
            search_window_size=viz_params.search_window_size if viz_params.analysis_type == "nlm" else None
        )
        
        title = f"Zoomed-In {filter_name}" if plot_type == 'zoomed' else filter_name
        
        try:
            visualize_image(
                filter_data,
                viz_params.ui_placeholders[plot_key],
                *viz_params.last_processed_pixel,
                viz_params.kernel_size,
                title=title,
                technique=viz_params.analysis_type,
                config=config
            )
        except Exception as e:
            logger.error(f"Error while visualizing {filter_name} filter: {e}", exc_info=True)
            raise

def visualize_image(image: ImageArray, placeholder, pixel_x: int, pixel_y: int, kernel_size: int, title: str, technique: str, config: VisualizationConfig):
    try:
        if config.zoom:
            image, pixel_x, pixel_y = get_zoomed_image_section(image, pixel_x, pixel_y, kernel_size)
        fig = create_image_plot(
            image, pixel_x, pixel_y, kernel_size, title,
            technique, 
            config=config
        )
        placeholder.pyplot(fig)  
        plt.close(fig)  # Close the figure after displaying
    except Exception as e:
        logger.error(f"Error while visualizing image: {e}", exc_info=True)
        raise

def create_image_plot(plot_image: ImageArray, center_x: int, center_y: int, plot_kernel_size: int, title: str, technique: str, config: VisualizationConfig):
    try:
        fig, ax = plt.subplots(1, 1, figsize=ZOOMED_IMAGE_DIMENSIONS)
        ax.imshow(plot_image, vmin=config.vmin, vmax=config.vmax, cmap=st.session_state.get('color_map', 'gray'))
        ax.set_title(title)
        ax.axis('off')
        add_overlays(ax, plot_image, center_x, center_y, plot_kernel_size, technique, config)
        fig.tight_layout(pad=2)
        return fig
    except Exception as e:
        logger.error(f"Error while creating image plot: {e}", exc_info=True)
        raise

def add_overlays(ax, plot_image: ImageArray, center_x: int, center_y: int, plot_kernel_size: int, technique: str, config: VisualizationConfig):
    if config.show_kernel:
        draw_kernel_overlay(ax, center_x, center_y, plot_kernel_size)
    if technique == "nlm" and config.search_window_size is not None and config.show_kernel:
        draw_search_window_overlay(ax, plot_image, center_x, center_y, config.search_window_size)
    if config.zoom and config.show_per_pixel_processing:  
        draw_pixel_value_annotations(ax, plot_image)
        
def prepare_filter_options_and_parameters(results: FilterResult, last_processed_pixel: PixelCoordinates):
    end_x, end_y = last_processed_pixel
    filter_options = results.get_filter_data()
    specific_params = {
        'kernel_size': results.kernel_size,
'pixels_processed': results.pixels_processed,
        'total_pixels': results.kernel_size ** 2
    }
    for filter_name, filter_data in filter_options.items():
        if isinstance(filter_data, np.ndarray) and filter_data.size > 0:
            specific_params[filter_name.lower().replace(" ", "_")] = filter_data[end_y, end_x]
    
    if isinstance(results, NLMResult):
        specific_params |= {
            'filter_strength': results.filter_strength,
            'search_window_size': results.search_window_size,
        }
    elif isinstance(results, SpeckleResult):
        specific_params |= {
            'start_pixel_mean': results.start_pixel_mean,
            'start_pixel_std_dev': results.start_pixel_std_dev,
            'start_pixel_speckle_contrast': results.start_pixel_speckle_contrast,
        }
    
    return filter_options, specific_params

def prepare_comparison_images():
    speckle_results = st.session_state.get("speckle_results")
    nlm_results = st.session_state.get("nlm_results")
    analysis_params = st.session_state.get('analysis_params', {})

    comparison_images = {
        'Unprocessed Image': analysis_params.get('image_array', np.array([]))
    }

    if speckle_results is not None:
        comparison_images |= speckle_results.get_filter_data()

    if nlm_results is not None:
        comparison_images.update(nlm_results.get_filter_data())

    return comparison_images if len(comparison_images) > 1 else None

# Utility Functions
def draw_kernel_overlay(ax, center_x: int, center_y: int, kernel_size: int):
    kernel_left = center_x - kernel_size // 2
    kernel_top = center_y - kernel_size // 2
    ax.add_patch(plt.Rectangle((kernel_left - 0.5, kernel_top - 0.5), kernel_size, kernel_size,
                               edgecolor=KERNEL_OUTLINE_COLOR, linewidth=1, facecolor="none"))
    lines = ([[(kernel_left + i - 0.5, kernel_top - 0.5), (kernel_left + i - 0.5, kernel_top + kernel_size - 0.5)] for i in range(1, kernel_size)] +
             [[(kernel_left - 0.5, kernel_top + i - 0.5), (kernel_left + kernel_size - 0.5, kernel_top + i - 0.5)] for i in range(1, kernel_size)])
    ax.add_collection(LineCollection(lines, colors=KERNEL_OUTLINE_COLOR, linestyles=':', linewidths=0.5))
    ax.add_patch(plt.Rectangle((center_x - 0.5, center_y - 0.5), 1, 1,
                               edgecolor=SEARCH_WINDOW_OUTLINE_COLOR, linewidth=0.5, facecolor=SEARCH_WINDOW_OUTLINE_COLOR, alpha=0.5))

def draw_search_window_overlay(ax, image: ImageArray, center_x: int, center_y: int, search_window: int):
    height, width = image.shape[:2]
    if search_window >= max(height, width):
        rect = plt.Rectangle((-0.5, -0.5), width, height, 
                             edgecolor=SEARCH_WINDOW_OUTLINE_COLOR, linewidth=2, facecolor="none")
    else:
        half_window = search_window // 2
        sw_left = max(-0.5, center_x - half_window - 0.5)
        sw_top = max(-0.5, center_y - half_window - 0.5)
        sw_width = min(width - sw_left, search_window)
        sw_height = min(height - sw_top, search_window)
        rect = plt.Rectangle((sw_left, sw_top), sw_width, sw_height,
                             edgecolor=SEARCH_WINDOW_OUTLINE_COLOR, linewidth=2, facecolor="none")
    ax.add_patch(rect)

def draw_pixel_value_annotations(ax, image: ImageArray):
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            ax.text(j, i, f"{image[i, j]:.2f}", ha="center", va="center", color=PIXEL_VALUE_TEXT_COLOR, fontsize=15)

def get_zoomed_image_section(image: ImageArray, center_x: int, center_y: int, zoom_size: int):
    top = max(0, center_y - zoom_size // 2)
    left = max(0, center_x - zoom_size // 2)
    zoomed_image = image[top:min(image.shape[0], top + zoom_size),
                         left:min(image.shape[1], left + zoom_size)]
    return zoomed_image, zoom_size // 2, zoom_size // 2

# Main UI Setup
def setup_and_run_analysis_techniques(analysis_params):
    techniques = st.session_state.get('techniques', [])
    tabs = st.session_state.get('tabs', [])

    for technique, tab in zip(techniques, tabs):
        if tab is None:
            continue
        
        with tab:
            technique_params = st.session_state.get(f"{technique}_params", {})
            ui_placeholders = create_technique_ui_elements(technique, tab, analysis_params.get('show_per_pixel_processing', False))
            st.session_state[f"{technique}_placeholders"] = ui_placeholders
            
            process_params = create_process_params(analysis_params, technique, technique_params)
            _, results = process_image(process_params)
            st.session_state[f"{technique}_results"] = results
            
            visualize_results(
                process_params.image_array, 
                technique, 
                analysis_params, 
                results, 
                process_params.show_per_pixel_processing
            )

    