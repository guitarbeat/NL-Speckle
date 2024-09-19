import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from src.speckle import process_speckle, SpeckleResult
from src.nlm import process_nlm, NLMResult
from src.formula import display_analysis_formula
from src.utils import (calculate_processing_details)
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, List
from logging import getLogger
from pydantic import BaseModel, field_validator


# Constants for Annotations and Overlays
KERNEL_OUTLINE_COLOR = 'red'
SEARCH_WINDOW_OUTLINE_COLOR = 'blue'
PIXEL_VALUE_TEXT_COLOR = 'red'
GRID_LINE_COLOR = 'red'
CENTER_PIXEL_COLOR = 'green'
KERNEL_OUTLINE_WIDTH = 1
SEARCH_WINDOW_OUTLINE_WIDTH = 2.0
GRID_LINE_WIDTH = 1
CENTER_PIXEL_OUTLINE_WIDTH = 1
PIXEL_VALUE_FONT_SIZE = 15
GRID_LINE_STYLE = ':'
DEFAULT_COLOR_MAP = 'gray'

# Constants for Image Visualization
ZOOMED_IMAGE_DIMENSIONS = (8, 8)
DEFAULT_SPECKLE_VIEW = ['Speckle Contrast', 'Original Image']
DEFAULT_NLM_VIEW = ['Non-Local Means', 'Original Image']
AVAILABLE_COLOR_MAPS = ["gray", "viridis", "plasma", "inferno", "magma", "cividis", "pink"]
PRELOADED_IMAGE_PATHS = {
    "image50.png": "media/image50.png", 
    "spatial.tif": "media/spatial.tif",
    "logo.jpg": "media/logo.jpg"
}
DEFAULT_SEARCH_WINDOW_SIZE = 21
DEFAULT_FILTER_STRENGTH = 10.0
DEFAULT_KERNEL_SIZE = 3



# Set up logging
logger = getLogger(__name__)


# ---------- Dataclasses ----------
@dataclass
class PixelCoordinates:
    x: int
    y: int

@dataclass
class ImageArray:
    data: np.ndarray

class VisualizationConfig(BaseModel):
    vmin: float | None
    vmax: float | None
    zoom: bool
    show_kernel: bool
    show_per_pixel_processing: bool 
    search_window_size: int | None

    @field_validator('vmin', 'vmax')
    @classmethod
    def validate_vmin_vmax(cls, v, info):
        if info.data.get('vmin') is not None and info.data.get('vmax') is not None and info.data['vmin'] > info.data['vmax']:
            raise ValueError("vmin cannot be greater than vmax.")
        return v
    
# --------- Classes ----------#
@dataclass
class VisualizationParams:
    image_array: ImageArray
    analysis_params: Dict[str, Any]
    results: Any
    ui_placeholders: Dict[str, Any]
    last_processed_pixel: PixelCoordinates
    kernel_size: int
    kernel_matrix: ImageArray
    original_pixel_value: float
    analysis_type: str
    show_per_pixel_processing: bool
    search_window_size: Optional[int]
    color_map: str

@dataclass
class ProcessParams:
    image_array: ImageArray
    analysis_params: Dict[str, Any]
    show_per_pixel_processing: bool
    technique: str
    update_state: bool
    handle_visualization: bool


# --------- Visualization Functions ----------#

def visualize_image(image: np.ndarray, placeholder, pixel_x: int, pixel_y: int, 
                    kernel_size: int, title: str, technique: str, config: VisualizationConfig) -> None:
    """
    Visualize an image with optional zooming and overlays.
    
    Args:
        image (np.ndarray): The image to visualize.
        placeholder (st.empty): Streamlit placeholder for rendering.
        pixel_x (int): X-coordinate of the center pixel.
        pixel_y (int): Y-coordinate of the center pixel.
        kernel_size (int): Size of the kernel.
        title (str): Title of the plot.
        technique (str): Analysis technique (e.g., "nlm" or "speckle").
        config (VisualizationConfig): Configuration parameters.
    """
    try:
        if config.zoom:
            image, pixel_x, pixel_y = get_zoomed_image_section(image, pixel_x, pixel_y, kernel_size)
        fig = create_image_plot(image, pixel_x, pixel_y, kernel_size, title, technique, config)

        placeholder.pyplot(fig)
        plt.close(fig)  # Close the figure after displaying
    except Exception as e:
        logger.error(f"Error while visualizing image: {e}", exc_info=True)
        raise

def visualize_analysis_results(viz_params: Dict[str, Any]) -> None:
    """
    Visualize analysis results based on the provided parameters.
    
    Args:
        viz_params (Dict[str, Any]): Visualization parameters including results, image array, etc.
    """
    try:
        last_processed_x, last_processed_y = viz_params.last_processed_pixel
        
        if viz_params.results is None:
            logger.warning("Results are None. Skipping visualization.")
            return

        filter_options, specific_params = prepare_filter_options_and_parameters(viz_params.results, viz_params.last_processed_pixel)
        filter_options['Original Image'] = viz_params.image_array.data

        selected_filters = st.session_state.get(f'{viz_params.analysis_type}_selected_filters', [])
        for filter_name in selected_filters:
            if filter_name in filter_options:
                filter_data = filter_options[filter_name]
                visualize_filter_and_zoomed(filter_name, filter_data, viz_params)

        if viz_params.show_per_pixel_processing:
            display_analysis_formula(specific_params, viz_params.ui_placeholders, 
                                     viz_params.analysis_type, last_processed_x, last_processed_y, 
                                     viz_params.kernel_size, viz_params.kernel_matrix.data, 
                                     viz_params.original_pixel_value)
    except Exception as e:
        logger.error(f"Error while visualizing analysis results: {e}", exc_info=True)
        raise

def create_image_plot(plot_image: np.ndarray, center_x: int, center_y: int, 
                      plot_kernel_size: int, title: str, technique: str, 
                      config: VisualizationConfig) -> plt.Figure:
    """
    Create a plot of the image with optional overlays.
    
    Args:
        plot_image (np.ndarray): The image to plot.
        center_x (int): X-coordinate of the center pixel.
        center_y (int): Y-coordinate of the center pixel.
        plot_kernel_size (int): Size of the kernel to plot.
        title (str): Title of the plot.
        technique (str): Analysis technique.
        config (VisualizationConfig): Configuration parameters.
    
    Returns:
        plt.Figure: The created figure.
    """
    try:
        fig, ax = plt.subplots(1, 1, figsize=ZOOMED_IMAGE_DIMENSIONS)
        ax.imshow(plot_image, vmin=config.vmin, vmax=config.vmax, 
                cmap=st.session_state.get('color_map', 'gray'))
        ax.set_title(title)
        ax.axis('off')
        add_overlays(ax, plot_image, center_x, center_y, plot_kernel_size, technique, config)
        fig.tight_layout(pad=2)
        return fig
    except Exception as e:
        logger.error(f"Error while creating image plot: {e}", exc_info=True)
        raise

def prepare_filter_options_and_parameters(results: Any, last_processed_pixel: Tuple[int, int]) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """
    Prepare filter options and specific parameters based on the analysis results.
    
    Args:
        results (Any): The analysis results object.
        last_processed_pixel (Tuple[int, int]): Coordinates of the last processed pixel.
    
    Returns:
        Tuple[Dict[str, np.ndarray], Dict[str, Any]]: Filter options and specific parameters.
    """
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

    if hasattr(results, 'filter_strength'):
        specific_params['filter_strength'] = results.filter_strength
        specific_params['search_window_size'] = results.search_window_size
    elif hasattr(results, 'start_pixel_mean'):
        specific_params |= {
            'start_pixel_mean': results.start_pixel_mean,
            'start_pixel_std_dev': results.start_pixel_std_dev,
            'start_pixel_speckle_contrast': results.start_pixel_speckle_contrast,
        }

    return filter_options, specific_params

def prepare_comparison_images() -> Dict[str, np.ndarray]:
    """
    Prepare images for comparison from different analysis results.
    
    Returns:
        Dict[str, np.ndarray]: A dictionary of image names and their corresponding arrays.
    """
    comparison_images = {
        'Unprocessed Image': st.session_state.get('analysis_params', {}).get('image_array', np.array([]))
    }

    for result_key in ['speckle_results', 'nlm_results']:
        results = st.session_state.get(result_key)
        if results is not None:
            comparison_images |= results.get_filter_data()

    return comparison_images if len(comparison_images) > 1 else None

def get_zoomed_image_section(image: np.ndarray, center_x: int, center_y: int, zoom_size: int) -> Tuple[np.ndarray, int, int]:
    """
    Extract a zoomed section of the image.
    
    Args:
        image (np.ndarray): The original image.
        center_x (int): X-coordinate of the zoom center.
        center_y (int): Y-coordinate of the zoom center.
        zoom_size (int): Size of the zoomed section.
    
    Returns:
        Tuple[np.ndarray, int, int]: Zoomed image section and new center coordinates.
    """
    half_zoom = zoom_size // 2
    top = max(0, center_y - half_zoom)
    bottom = min(image.shape[0], top + zoom_size)
    left = max(0, center_x - half_zoom)
    right = min(image.shape[1], left + zoom_size)
    
    row_indices = np.arange(top, bottom)
    col_indices = np.arange(left, right)
    
    zoomed_image = np.take(np.take(image, row_indices, axis=0), col_indices, axis=1)
    
    return zoomed_image, center_x - left, center_y - top

def visualize_filter_and_zoomed(filter_name: str, filter_data: np.ndarray, viz_params: VisualizationParams):
    for plot_type in ['main', 'zoomed']:
        plot_key = f'zoomed_{filter_name.lower().replace(" ", "_")}' if plot_type == 'zoomed' else filter_name.lower().replace(" ", "_")
        
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

def extract_kernel_from_image(image_array, end_x, end_y, kernel_size):
    """Extract a kernel from an image centered at the given coordinates."""
    half_kernel = kernel_size // 2
    height, width = image_array.shape
    y_start, y_end = max(0, end_y - half_kernel), min(height, end_y + half_kernel + 1)
    x_start, x_end = max(0, end_x - half_kernel), min(width, end_x + half_kernel + 1)
    kernel_values = image_array[y_start:y_end, x_start:x_end]
    
    if kernel_values.size == 0:
        raise ValueError(f"Extracted kernel at ({end_x}, {end_y}) is empty. Image shape: {image_array.shape}, Kernel size: {kernel_size}")
    
    if kernel_values.shape != (kernel_size, kernel_size):
        pad_width_y = (max(0, half_kernel - end_y), max(0, end_y + half_kernel + 1 - height))
        pad_width_x = (max(0, half_kernel - end_x), max(0, end_x + half_kernel + 1 - width))
        kernel_values = np.pad(kernel_values, (pad_width_y, pad_width_x), mode='edge')
    
    return kernel_values.astype(float), float(image_array[end_y, end_x]), kernel_size





# --------- Utility Functions ----------#

def add_overlays(ax: plt.Axes, plot_image: np.ndarray, center_x: int, center_y: int, 
                 plot_kernel_size: int, technique: str, config: VisualizationConfig) -> None:
    """
    Add overlays to the plot based on the technique and configuration.
    
    Args:
        ax (plt.Axes): The axes to add overlays to.
        plot_image (np.ndarray): The image being plotted.
        center_x (int): X-coordinate of the center pixel.
        center_y (int): Y-coordinate of the center pixel.
        plot_kernel_size (int): Size of the kernel to plot.
        technique (str): Analysis technique.
        config (VisualizationConfig): Configuration parameters.
    """
    if config.show_kernel:
        draw_kernel_overlay(ax, center_x, center_y, plot_kernel_size)
    if technique == "nlm" and config.search_window_size is not None and config.show_kernel:
        draw_search_window_overlay(ax, plot_image, center_x, center_y, config.search_window_size)
    if config.zoom and config.show_per_pixel_processing:  
        draw_pixel_value_annotations(ax, plot_image)

def draw_kernel_overlay(ax: plt.Axes, center_x: int, center_y: int, kernel_size: int) -> None:
    """
    Draw a kernel overlay on the given axes.
    
    Args:
        ax (plt.Axes): The axes to draw on.
        center_x (int): X-coordinate of the kernel center.
        center_y (int): Y-coordinate of the kernel center.
        kernel_size (int): Size of the kernel.
    """
    half_kernel = kernel_size // 2
    kernel_left = center_x - half_kernel
    kernel_top = center_y - half_kernel
    
    # Draw main rectangle
    ax.add_patch(plt.Rectangle((kernel_left - 0.5, kernel_top - 0.5), kernel_size, kernel_size,
                               edgecolor=KERNEL_OUTLINE_COLOR, linewidth=KERNEL_OUTLINE_WIDTH, facecolor="none"))
    
    # Draw grid lines
    grid_lines = (
        [[(kernel_left + i - 0.5, kernel_top - 0.5), (kernel_left + i - 0.5, kernel_top + kernel_size - 0.5)] for i in range(1, kernel_size)] +
        [[(kernel_left - 0.5, kernel_top + i - 0.5), (kernel_left + kernel_size - 0.5, kernel_top + i - 0.5)] for i in range(1, kernel_size)]
    )
    ax.add_collection(LineCollection(grid_lines, colors=GRID_LINE_COLOR, linestyles=GRID_LINE_STYLE, linewidths=GRID_LINE_WIDTH))
    
    # Highlight center pixel
    ax.add_patch(plt.Rectangle((center_x - 0.5, center_y - 0.5), 1, 1,
                               edgecolor=CENTER_PIXEL_COLOR, linewidth=CENTER_PIXEL_OUTLINE_WIDTH, 
                               facecolor=CENTER_PIXEL_COLOR, alpha=0.5))

def draw_search_window_overlay(ax: plt.Axes, image: np.ndarray, center_x: int, center_y: int, search_window: int) -> None:
    """
    Draw a search window overlay on the given axes.
    
    Args:
        ax (plt.Axes): The axes to draw on.
        image (np.ndarray): The image being processed.
        center_x (int): X-coordinate of the window center.
        center_y (int): Y-coordinate of the window center.
        search_window (int): Size of the search window.
    """
    height, width = image.shape[:2]
    half_window = search_window // 2
    
    sw_left = max(-0.5, center_x - half_window - 0.5)
    sw_top = max(-0.5, center_y - half_window - 0.5)
    sw_width = min(width - sw_left, search_window)
    sw_height = min(height - sw_top, search_window)
    
    rect = plt.Rectangle((sw_left, sw_top), sw_width, sw_height,
                         edgecolor=SEARCH_WINDOW_OUTLINE_COLOR, linewidth=SEARCH_WINDOW_OUTLINE_WIDTH, facecolor="none")
    ax.add_patch(rect)

def draw_pixel_value_annotations(ax: plt.Axes, image: np.ndarray) -> None:
    """
    Annotate pixel values on the given axes.
    
    Args:
        ax (plt.Axes): The axes to annotate.
        image (np.ndarray): The image containing pixel values.
    """
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            ax.text(j, i, f"{image[i, j]:.2f}", ha="center", va="center", 
                    color=PIXEL_VALUE_TEXT_COLOR, fontsize=PIXEL_VALUE_FONT_SIZE)


#---- Recently Refactoed Code Below- ----#

# --------- Image Processing Functions ----------#

def process_image(params: ProcessParams):
    """Process an image based on the provided parameters."""
    technique = params.technique
    analysis_params = params.analysis_params
    
    # Extract common parameters
    kernel_size = st.session_state.get('kernel_size', 3)
    pixels_to_process = analysis_params.get('pixels_to_process', 0)
    analysis_params.update({'kernel_size': kernel_size, 'pixels_to_process': pixels_to_process})
    
    # Normalize image if needed
    normalized_image = normalize_image(params.image_array) if analysis_params.get('normalization_option') == 'Percentile' else params.image_array
    
    # Process image based on technique
    if technique == "nlm":
        nlm_params = analysis_params.get('nlm_params', {})
        search_window_size = nlm_params.get('search_window_size')
        filter_strength = nlm_params.get('filter_strength')
        results = process_nlm(normalized_image, kernel_size, pixels_to_process, search_window_size, filter_strength)
    elif technique == "speckle":
        results = process_speckle(normalized_image, kernel_size, pixels_to_process)
    else:
        raise ValueError(f"Unknown technique: {technique}")
    
    # Handle visualization and state updates
    if params.handle_visualization:
        visualize_results(normalized_image, technique, analysis_params, results, params.show_per_pixel_processing)
    
    if params.update_state:
        update_session_state(technique, pixels_to_process, results)
    
    return params, results

def normalize_image(image, low_percentile=2, high_percentile=98):
    """Normalize an image using percentile-based scaling."""
    p_low, p_high = np.percentile(image, [low_percentile, high_percentile])
    return (np.clip(image, p_low, p_high) - p_low) / (p_high - p_low)

# --------- Visualization Functions ----------#

def visualize_results(image_array: ImageArray, technique: str, analysis_params: Dict[str, Any], results: Any, show_per_pixel_processing: bool):
    """Visualize the results of image processing."""
    try:
        viz_params = create_visualization_params(image_array, technique, analysis_params, results, show_per_pixel_processing)
        visualize_analysis_results(viz_params)
    except Exception as e:
        logger.error(f"Error while visualizing results: {e}", exc_info=True)
        st.error("An error occurred while visualizing the results. Please check the logs.")

def create_visualization_params(image_array: ImageArray, technique: str, analysis_params: Dict[str, Any], results: Any, show_per_pixel_processing: bool) -> VisualizationParams:
    """Create VisualizationParams object for result visualization."""
    processing_details = calculate_processing_details(image_array, analysis_params.get('kernel_size', 3), analysis_params.get('total_pixels', 0))
    last_processed_x, last_processed_y = get_last_processed_coordinates(results, processing_details)
    kernel_matrix, original_pixel_value, kernel_size = extract_kernel_from_image(image_array, last_processed_x, last_processed_y, analysis_params.get('kernel_size', 3))
    
    return VisualizationParams(
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

# --------- UI Setup Functions ----------#

def setup_and_run_analysis_techniques(analysis_params: Dict[str, Any]) -> None:
    """Set up and run analysis techniques based on the provided parameters."""
    techniques: List[str] = st.session_state.get('techniques', [])
    tabs: List[Any] = st.session_state.get('tabs', [])

    for technique, tab in zip(techniques, tabs):
        if tab is not None:
            with tab:
                run_technique(technique, tab, analysis_params)

def run_technique(technique: str, tab: Any, analysis_params: Dict[str, Any]) -> None:
    """Run a specific analysis technique."""
    technique_params = st.session_state.get(f"{technique}_params", {})
    show_per_pixel_processing = analysis_params.get('show_per_pixel_processing', False)
    
    ui_placeholders = create_technique_ui_elements(technique, tab, show_per_pixel_processing)
    st.session_state[f"{technique}_placeholders"] = ui_placeholders
    
    process_params = create_process_params(analysis_params, technique, technique_params)
    _, results = process_image(process_params)
    st.session_state[f"{technique}_results"] = results

def create_technique_ui_elements(technique: str, tab: Any, show_per_pixel_processing: bool) -> Dict[str, Any]:
    """Create UI elements for a specific image processing technique."""
    with tab:
        ui_placeholders = {'formula': st.empty(), 'original_image': st.empty()}

        filter_options = get_filter_options(technique)
        if selected_filters := create_filter_selection(
            technique, filter_options
        ):
            create_filter_views(selected_filters, ui_placeholders, show_per_pixel_processing)
        else:
            st.warning("No views selected. Please select at least one view to display.")

        if show_per_pixel_processing:
            ui_placeholders['zoomed_kernel'] = st.empty()

    return ui_placeholders

# --------- Utility Functions ----------#

def get_filter_options(technique: str) -> List[str]:
    """Get filter options based on the technique."""
    if technique == "speckle":
        return ['Original Image'] + SpeckleResult.get_filter_options()
    elif technique == "nlm":
        return ['Original Image'] + NLMResult.get_filter_options()
    else:
        return []

def create_filter_selection(technique: str, filter_options: List[str]) -> List[str]:
    """Create and return filter selection UI element."""
    default_selection = DEFAULT_SPECKLE_VIEW if technique == "speckle" else DEFAULT_NLM_VIEW
    selected_filters = st.multiselect(
        "Select views to display",
        filter_options,
        default=default_selection,
        key=f"{technique}_filter_selection"
    )
    st.session_state[f'{technique}_selected_filters'] = selected_filters
    return selected_filters

def create_filter_views(selected_filters: List[str], ui_placeholders: Dict[str, Any], show_per_pixel_processing: bool) -> None:
    """Create views for selected filters."""
    columns = st.columns(len(selected_filters))
    for i, filter_name in enumerate(selected_filters):
        ui_placeholders[filter_name.lower().replace(" ", "_")] = columns[i].empty()
        if show_per_pixel_processing:
            ui_placeholders[f'zoomed_{filter_name.lower().replace(" ", "_")}'] = columns[i].expander(f"Zoomed-in {filter_name}", expanded=False).empty()

def update_session_state(technique: str, pixels_to_process: int, results: Any) -> None:
    """Update session state with processing results."""
    st.session_state.update({
        'processed_pixels': pixels_to_process,
        f"{technique}_results": results
    })

def get_last_processed_coordinates(results: Any, processing_details: Any) -> tuple:
    """Get the coordinates of the last processed pixel."""
    if isinstance(results, (NLMResult, SpeckleResult)):
        return results.processing_end_coord
    else:
        return processing_details.end_x, processing_details.end_y
    
# --------- Helpers ----------#

def create_process_params(analysis_params: Dict[str, Any], technique: str, technique_params: Dict[str, Any]) -> ProcessParams:
    return ProcessParams(
        image_array=analysis_params.get('image_array', ImageArray(np.array([]))),
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
