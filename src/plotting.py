import itertools
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from src.speckle import process_speckle, SpeckleResult
from src.nlm import process_nlm, NLMResult
from src.formula import display_analysis_formula
from src.utils import calculate_processing_details
from dataclasses import dataclass, field
from typing import Any, Dict, Tuple, List, Optional
import logging

# Constants for Image Visualization
ZOOMED_IMAGE_DIMENSIONS = (8, 8)
DEFAULT_SPECKLE_VIEW = ['Speckle Contrast', 'Original Image']
DEFAULT_NLM_VIEW = ['Non-Local Means', 'Original Image']

# Constants for plot type
PLOT_MAIN = 'main'
PLOT_ZOOMED = 'zoomed'
DEFAULT_SEARCH_WINDOW_SIZE = 21
DEFAULT_FILTER_STRENGTH = 10.0
DEFAULT_KERNEL_SIZE = 3

@dataclass
class PixelCoordinates:
    """Represents the pixel's (x, y) coordinates."""
    x: int
    y: int

@dataclass
class ImageArray:
    """Container for the image data as a numpy array."""
    data: np.ndarray


@dataclass
class VisualizationConfig:
    """Holds configuration for image visualization and analysis settings."""
    vmin: Optional[float] = None
    vmax: Optional[float] = None
    zoom: bool = False
    show_kernel: bool = False
    show_per_pixel_processing: bool = False
    search_window_size: Optional[int] = None
    use_full_image: bool = False
    image_array: Optional[ImageArray] = None
    analysis_params: Dict[str, Any] = field(default_factory=dict)
    results: Optional[Any] = None
    ui_placeholders: Dict[str, Any] = field(default_factory=dict)
    last_processed_pixel: Optional[PixelCoordinates] = None
    kernel_size: int = 3
    kernel_matrix: Optional[np.ndarray] = None
    original_pixel_value: float = 0.0
    analysis_type: str = ""
    color_map: str = 'gray'
    kernel_outline_color: str = 'red'
    search_window_outline_color: str = 'blue'
    pixel_value_text_color: str = 'red'
    grid_line_color: str = 'red'
    center_pixel_color: str = 'green'
    kernel_outline_width: int = 1
    search_window_outline_width: float = 2.0
    grid_line_width: int = 1
    center_pixel_outline_width: int = 1
    pixel_value_font_size: int = 15
    grid_line_style: str = ':'
    title: str = ""
    figure_size: Tuple[int, int] = (8, 8)

    def __post_init__(self):
        """Post-initialization validation."""
        self._validate_vmin_vmax()

    def _validate_vmin_vmax(self):
        """Ensure vmin is not greater than vmax."""
        if self.vmin is not None and self.vmax is not None and self.vmin > self.vmax:
            raise ValueError("vmin cannot be greater than vmax.")

    @property
    def zoom_dimensions(self) -> Tuple[int, int]:
        """Return zoomed dimensions if zoom is enabled."""
        return self.figure_size if self.zoom else (self.image_array.data.shape[:2])

    def set_kernel_matrix(self, matrix: np.ndarray):
        """Set the kernel matrix with validation."""
        if matrix.shape != (self.kernel_size, self.kernel_size):
            raise ValueError(f"Kernel matrix must be of shape ({self.kernel_size}, {self.kernel_size})")
        self.kernel_matrix = matrix

def create_visualization_config(image_array: ImageArray, technique: str, analysis_params: Dict[str, Any], results: Any, 
                                last_processed_pixel: Tuple[int, int], kernel_matrix: np.ndarray, kernel_size: int, 
                                original_pixel_value: float, show_per_pixel_processing: bool) -> VisualizationConfig:
    """Utility to create a VisualizationConfig object."""
    return VisualizationConfig(
        vmin=None,
        vmax=None,
        zoom=False,
        show_kernel=show_per_pixel_processing,
        show_per_pixel_processing=show_per_pixel_processing,
        search_window_size=analysis_params.get('search_window_size'),
        use_full_image=analysis_params.get('use_whole_image', False),
        image_array=image_array,
        analysis_params=analysis_params,
        results=results,
        ui_placeholders=st.session_state.get(f'{technique}_placeholders', {}),
        last_processed_pixel=PixelCoordinates(x=last_processed_pixel[0], y=last_processed_pixel[1]),
        kernel_size=kernel_size,
        kernel_matrix=kernel_matrix,
        original_pixel_value=original_pixel_value,
        analysis_type=technique,
        color_map=st.session_state.get('color_map', 'gray'),
        title=f"{technique.upper()} Analysis Result",
        figure_size=(8, 8)
    )
@dataclass
class ProcessParams:
    """Holds parameters for image processing."""
    image_array: ImageArray
    analysis_params: Dict[str, Any]
    show_per_pixel_processing: bool
    technique: str
    update_state: bool
    handle_visualization: bool


def create_process_params(analysis_params: Dict[str, Any], technique: str, technique_params: Dict[str, Any]) -> ProcessParams:
    common_params = {
        'kernel_size': st.session_state.get('kernel_size', 3),
        'pixels_to_process': analysis_params.get('pixels_to_process', 0),
        'total_pixels': analysis_params.get('total_pixels', 0),
        'show_per_pixel_processing': analysis_params.get('show_per_pixel_processing', False),
    }

    if technique == "nlm":
        common_params |= {
            'search_window_size': analysis_params.get('search_window_size'),
            'filter_strength': analysis_params.get('filter_strength'),
        }

    return ProcessParams(
        image_array=analysis_params.get(
            'image_array', ImageArray(np.array([]))
        ),
        technique=technique,
        analysis_params=technique_params | common_params,
        update_state=True,
        handle_visualization=True,
        show_per_pixel_processing=analysis_params.get(
            'show_per_pixel_processing', False
        ),
    )
# --------- Updated Functions ----------#



def visualize_results(image_array: ImageArray, technique: str, analysis_params: Dict[str, Any], results: Any, show_per_pixel_processing: bool):
    """Visualize the results of image processing."""
    try:
        # Extract processing details and kernel information
        processing_details = calculate_processing_details(image_array, analysis_params.get('kernel_size', 3), analysis_params.get('total_pixels', 0))
        pixel_coords = get_last_processed_coordinates(results, processing_details)

        # Check if pixel_coords is valid
        if pixel_coords is None:
            raise ValueError("No pixel coordinates returned.")

        last_processed_x, last_processed_y = pixel_coords.x, pixel_coords.y
        kernel_matrix, original_pixel_value, kernel_size = extract_kernel_from_image(image_array, last_processed_x, last_processed_y, analysis_params.get('kernel_size', 3))

        # Create VisualizationConfig object for result visualization
        viz_config = create_visualization_config(
            image_array=image_array,
            technique=technique,
            analysis_params=analysis_params,
            results=results,
            last_processed_pixel=(last_processed_x, last_processed_y),
            kernel_matrix=kernel_matrix,
            kernel_size=kernel_size,
            original_pixel_value=original_pixel_value,
            show_per_pixel_processing=show_per_pixel_processing
        )

        visualize_analysis_results(viz_config)
    except Exception as e:
        logging.error(f"Error while visualizing results for {technique}: {e}")
        st.error("An error occurred while visualizing the results. Please check the logs.")

def visualize_filter_and_zoomed(filter_name: str, filter_data: np.ndarray, viz_config: VisualizationConfig):
    """Visualize the main and zoomed versions of a filter."""
    for plot_type in [PLOT_MAIN, PLOT_ZOOMED]:
        plot_key = generate_plot_key(filter_name, plot_type)

        # Skip unnecessary visualizations
        if plot_key not in viz_config.ui_placeholders or (plot_type == PLOT_ZOOMED and not viz_config.show_per_pixel_processing):
            continue

        # Create updated config for zoomed view
        config = update_visualization_config(viz_config, filter_data, filter_name, plot_type)
        title = f"Zoomed-In {filter_name}" if plot_type == PLOT_ZOOMED else filter_name

        try:
            visualize_image(
                filter_data,
                viz_config.ui_placeholders[plot_key],
                *viz_config.last_processed_pixel,
                viz_config.kernel_size,
                title=title,
                technique=viz_config.analysis_type,
                config=config
            )
        except Exception as e:
            logging.error(f"Error while visualizing {filter_name} filter: {e}")
            raise

def generate_plot_key(filter_name: str, plot_type: str) -> str:
    """Generate a key for identifying plots based on filter name and plot type."""
    base_key = filter_name.lower().replace(" ", "_")
    return f'zoomed_{base_key}' if plot_type == PLOT_ZOOMED else base_key

def update_visualization_config(viz_config: VisualizationConfig, filter_data: np.ndarray, filter_name: str, plot_type: str) -> VisualizationConfig:
    """Update the VisualizationConfig object for zoomed or main view."""
    return VisualizationConfig(
        vmin=None if filter_name == 'Original Image' else np.min(filter_data),
        vmax=None if filter_name == 'Original Image' else np.max(filter_data),
        zoom=(plot_type == PLOT_ZOOMED),
        show_kernel=(viz_config.show_per_pixel_processing if plot_type == PLOT_MAIN else True),
        show_per_pixel_processing=(plot_type == PLOT_ZOOMED),
        search_window_size=viz_config.search_window_size if viz_config.analysis_type == "nlm" else None,
        use_full_image=viz_config.analysis_params.get('use_whole_image', False),
        image_array=viz_config.image_array,
        analysis_params=viz_config.analysis_params,
        results=viz_config.results,
        ui_placeholders=viz_config.ui_placeholders,
        last_processed_pixel=viz_config.last_processed_pixel,
        kernel_size=viz_config.kernel_size,
        kernel_matrix=viz_config.kernel_matrix,
        original_pixel_value=viz_config.original_pixel_value,
        analysis_type=viz_config.analysis_type,
        color_map=viz_config.color_map
    )

# --------- Visualization Functions ----------#

def visualize_analysis_results(viz_params: VisualizationConfig) -> None:
    """
    Visualize analysis results based on the provided parameters.
    
    Args:
        viz_params (VisualizationConfig): Visualization parameters including results, image array, etc.
    """
    try:
        last_processed_x, last_processed_y = viz_params.last_processed_pixel
        
        if viz_params.results is None:
            logging.warning("Results are None. Skipping visualization.")
            return

        filter_options, specific_params = prepare_filter_options_and_parameters(viz_params.results, viz_params.last_processed_pixel)
        filter_options['Original Image'] = viz_params.image_array.data

        selected_filters = st.session_state.get(f'{viz_params.analysis_type}_selected_filters', [])
        for filter_name in selected_filters:
            if filter_name in filter_options:
                filter_data = filter_options[filter_name]
                visualize_filter_and_zoomed(filter_name, filter_data, viz_params)

        if viz_params.show_per_pixel_processing:
            display_analysis_formula(
                specific_params, viz_params.ui_placeholders,
                viz_params.analysis_type, last_processed_x, last_processed_y,
                viz_params.kernel_size, viz_params.kernel_matrix,
                viz_params.original_pixel_value
            )
    except Exception as e:
        logging.error(f"Error while visualizing analysis results: {e}")
        st.error("An error occurred during visualization. Please check the logs.")
        raise

def create_image_plot(plot_image: np.ndarray, config: VisualizationConfig) -> plt.Figure:
    try:
        fig, ax = plt.subplots(1, 1, figsize=config.figure_size)
        ax.imshow(plot_image, vmin=config.vmin, vmax=config.vmax, cmap=config.color_map)
        ax.set_title(config.title)
        ax.axis('off')

        add_overlays(ax, plot_image, config)
        fig.tight_layout(pad=2)
        return fig
    except Exception as e:
        logging.error(f"Error while creating image plot for {config.title}: {e}")
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

    # Include any additional attributes if available
    if hasattr(results, 'filter_strength'):
        specific_params |= {
            'filter_strength': results.filter_strength,
            'search_window_size': results.search_window_size,
        }
    elif hasattr(results, 'start_pixel_mean'):
        specific_params |= {
            'start_pixel_mean': results.start_pixel_mean,
            'start_pixel_std_dev': results.start_pixel_std_dev,
            'start_pixel_speckle_contrast': results.start_pixel_speckle_contrast,
        }

    return filter_options, specific_params

def prepare_comparison_images() -> Optional[Dict[str, np.ndarray]]:
    """
    Prepare images for comparison from different analysis results.
    
    Returns:
        Optional[Dict[str, np.ndarray]]: A dictionary of image names and their corresponding arrays, or None.
    """
    comparison_images = {
        'Unprocessed Image': st.session_state.get('analysis_params', {}).get('image_array', np.array([]))
    }

    for result_key in ['speckle_results', 'nlm_results']:
        results = st.session_state.get(result_key)
        if results is not None:
            comparison_images |= results.get_filter_data()

    return comparison_images if len(comparison_images) > 1 else None

def get_zoomed_image_section(image: np.ndarray, center_x: int, center_y: int, kernel_size: int) -> Tuple[np.ndarray, int, int]:
    """
    Extract a zoomed section of the image.
    
    Args:
        image (np.ndarray): The original image.
        center_x (int): X-coordinate of the zoom center.
        center_y (int): Y-coordinate of the zoom center.
        kernel_size (int): Size of the zoomed section.
    
    Returns:
        Tuple[np.ndarray, int, int]: Zoomed image section and new center coordinates.
    """
    half_zoom = kernel_size // 2
    top = max(0, center_y - half_zoom)
    bottom = min(image.shape[0], top + kernel_size)
    left = max(0, center_x - half_zoom)
    right = min(image.shape[1], left + kernel_size)
    
    zoomed_image = image[top:bottom, left:right]
    
    new_center_x = center_x - left
    new_center_y = center_y - top
    
    return zoomed_image, new_center_x, new_center_y

def extract_kernel_from_image(image_array: np.ndarray, end_x: int, end_y: int, kernel_size: int) -> Tuple[np.ndarray, float, int]:
    """
    Extract a kernel from an image centered at the given coordinates.
    
    Args:
        image_array (np.ndarray): The input image.
        end_x (int): X-coordinate for the kernel center.
        end_y (int): Y-coordinate for the kernel center.
        kernel_size (int): Size of the kernel.
    
    Returns:
        Tuple[np.ndarray, float, int]: The extracted kernel, the original pixel value, and the kernel size.
    """
    half_kernel = kernel_size // 2
    height, width = image_array.shape

    # Determine kernel bounds
    y_start, y_end = max(0, end_y - half_kernel), min(height, end_y + half_kernel + 1)
    x_start, x_end = max(0, end_x - half_kernel), min(width, end_x + half_kernel + 1)
    kernel_values = image_array[y_start:y_end, x_start:x_end]

    # Handle edge cases by padding the kernel if needed
    if kernel_values.size == 0:
        raise ValueError(f"Extracted kernel at ({end_x}, {end_y}) is empty. Image shape: {image_array.shape}, Kernel size: {kernel_size}")

    if kernel_values.shape != (kernel_size, kernel_size):
        kernel_values = np.pad(kernel_values, ((max(0, half_kernel - end_y), max(0, end_y + half_kernel + 1 - height)),
                                               (max(0, half_kernel - end_x), max(0, end_x + half_kernel + 1 - width))), 
                                               mode='edge')

    return kernel_values.astype(float), float(image_array[end_y, end_x]), kernel_size

# ---- Annotation Functions---=--#

def add_overlays(ax: plt.Axes, plot_image: np.ndarray, config: VisualizationConfig) -> None:
    """
    Add overlays to the plot based on the technique and configuration.
    
    Args:
        ax (plt.Axes): The axes to add overlays to.
        plot_image (np.ndarray): The image being plotted.
        config (VisualizationConfig): Configuration parameters.
    """
    logging.debug(f"[add_overlays] center: {config.last_processed_pixel}, kernel_size: {config.kernel_size}, technique: {config.analysis_type}, config: {config}")
    
    if not validate_inputs(config):
        raise ValueError("Invalid inputs for center, kernel size or config")

    if config.show_kernel:
        draw_kernel_overlay(ax, config)
        
    if config.analysis_type == "nlm" and config.search_window_size is not None and config.show_kernel:
        draw_search_window_overlay(ax, plot_image, config)
        
    if config.zoom and config.show_per_pixel_processing:
        draw_pixel_value_annotations(ax, plot_image, config)

def validate_inputs(config: VisualizationConfig) -> bool:
    """
    Validate the input parameters.
    
    Args:
        config (VisualizationConfig): Configuration parameters.
    
    Returns:
        bool: Whether the inputs are valid.
    """
    if config.last_processed_pixel.x < 0 or config.last_processed_pixel.y < 0:
        logging.error("Center coordinates must be non-negative.")
        return False
    if config.kernel_size <= 0:
        logging.error("Kernel size must be positive.")
        return False
    return True

def draw_kernel_overlay(ax: plt.Axes, config: VisualizationConfig) -> None:
    """
    Draw a kernel overlay on the given axes.
    
    Args:
        ax (plt.Axes): The axes to draw on.
        config (VisualizationConfig): Configuration parameters for colors, line width, etc.
    """
    center_x, center_y = config.last_processed_pixel.x, config.last_processed_pixel.y
    half_kernel = config.kernel_size // 2
    kernel_left = center_x - half_kernel
    kernel_top = center_y - half_kernel
    
    # Draw main rectangle
    ax.add_patch(plt.Rectangle((kernel_left - 0.5, kernel_top - 0.5), config.kernel_size, config.kernel_size,
                               edgecolor=config.kernel_outline_color, linewidth=config.kernel_outline_width, facecolor="none"))
    
    # Draw grid lines
    grid_lines = (
        [[(kernel_left + i - 0.5, kernel_top - 0.5), (kernel_left + i - 0.5, kernel_top + config.kernel_size - 0.5)] for i in range(1, config.kernel_size)] +
        [[(kernel_left - 0.5, kernel_top + i - 0.5), (kernel_left + config.kernel_size - 0.5, kernel_top + i - 0.5)] for i in range(1, config.kernel_size)]
    )
    ax.add_collection(LineCollection(grid_lines, colors=config.grid_line_color, linestyles=config.grid_line_style, linewidths=config.grid_line_width))
    
    # Highlight center pixel
    ax.add_patch(plt.Rectangle((center_x - 0.5, center_y - 0.5), 1, 1,
                               edgecolor=config.center_pixel_color, linewidth=config.center_pixel_outline_width, 
                               facecolor=config.center_pixel_color, alpha=0.5))

def draw_search_window_overlay(axes: plt.Axes, image: np.ndarray, config: VisualizationConfig) -> None:
    """
    Draw a search window overlay on the given axes.

    Args:
        axes (plt.Axes): The axes to draw on.
        image (np.ndarray): The image being processed.
        config (VisualizationConfig): Configuration parameters.
    """
    image_height, image_width = image.shape[:2]
    half_window_size = config.search_window_size // 2
    center_x, center_y = config.last_processed_pixel.x, config.last_processed_pixel.y

    if config.use_full_image:
        window_left, window_top = -0.5, -0.5
        window_width, window_height = image_width, image_height
        logging.debug("Using full image as search window")
    else:
        window_left = max(0, center_x - half_window_size) - 0.5
        window_top = max(0, center_y - half_window_size) - 0.5
        window_width = min(image_width - (center_x - half_window_size), config.search_window_size)
        window_height = min(image_height - (center_y - half_window_size), config.search_window_size)

    axes.add_patch(plt.Rectangle((window_left, window_top), window_width, window_height,
                                 edgecolor=config.search_window_outline_color, linewidth=config.search_window_outline_width, facecolor="none"))

def draw_pixel_value_annotations(ax: plt.Axes, image: np.ndarray, config: VisualizationConfig) -> None:
    """
    Annotate pixel values on the given axes.
    
    Args:
        ax (plt.Axes): The axes to annotate.
        image (np.ndarray): The image containing pixel values.
        config (VisualizationConfig): Configuration parameters for text formatting.
    """
    image_height, image_width = image.shape[:2]

    for i, j in itertools.product(range(image_height), range(image_width)):
        ax.text(j, i, f"{image[i, j]:.2f}", ha="center", va="center", 
                color=config.pixel_value_text_color, fontsize=config.pixel_value_font_size)
# --------- Image Processing Functions ----------#

def process_image(params: ProcessParams):
    """
    Process an image based on the provided parameters.
    
    Args:
        params (ProcessParams): The processing parameters including image array and techniques.
    
    Returns:
        tuple: The modified parameters and results.
    """
    logging.debug("Starting image processing...")
    
    technique = params.technique
    analysis_params = params.analysis_params

    # Extract or assign default parameters
    kernel_size = extract_or_default(st.session_state, 'kernel_size', 3)
    pixels_to_process = extract_or_default(analysis_params, 'pixels_to_process', 0)
    search_window_size = extract_or_default(analysis_params, 'search_window_size', DEFAULT_SEARCH_WINDOW_SIZE)
    filter_strength = extract_or_default(analysis_params, 'filter_strength', DEFAULT_FILTER_STRENGTH)

    # Update analysis_params with the resolved values
    analysis_params.update({
        'kernel_size': kernel_size,
        'pixels_to_process': pixels_to_process,
        'search_window_size': search_window_size,
        'filter_strength': filter_strength
    })

    # Normalize image if necessary
    normalized_image = normalize_image(params.image_array) if analysis_params.get('normalization_option') == 'Percentile' else params.image_array
    
    # Process image based on the specified technique
    results = apply_technique(technique, normalized_image, kernel_size, pixels_to_process, search_window_size, filter_strength)

    # Handle visualization and state updates
    if params.handle_visualization:
        visualize_results(normalized_image, technique, analysis_params, results, params.show_per_pixel_processing)
    
    if params.update_state:
        update_session_state(technique, pixels_to_process, results)

    return params, results

def extract_or_default(source: dict, key: str, default_value):
    """
    Extract a value from the dictionary or return a default if key is not present.
    
    Args:
        source (dict): The dictionary to extract from.
        key (str): The key to look for.
        default_value: The value to return if key is not found.
    
    Returns:
        The extracted value or the default value.
    """
    value = source.get(key, default_value)
    logging.debug(f"Parameter '{key}': {value}")
    return value

def apply_technique(technique: str, image: np.ndarray, kernel_size: int, pixels_to_process: int, search_window_size: int, filter_strength: int):
    """
    Apply the specified technique to the image.
    
    Args:
        technique (str): The processing technique to apply.
        image (np.ndarray): The image array to process.
        kernel_size (int): The size of the kernel.
        pixels_to_process (int): Number of pixels to process.
        search_window_size (int): The size of the search window (for NLM).
        filter_strength (int): The strength of the filter.
    
    Returns:
        The results of the image processing.
    """
    logging.debug(f"Applying technique: {technique}")
    
    if technique == "nlm":
        return process_nlm(
            image=image, 
            kernel_size=kernel_size, 
            pixels_to_process=pixels_to_process, 
            search_window_size=search_window_size, 
            filter_strength=filter_strength
        )
    elif technique == "speckle":
        return process_speckle(image, kernel_size, pixels_to_process)
    else:
        logging.error(f"Unknown technique: {technique}")
        raise ValueError(f"Unknown technique: {technique}")

def normalize_image(image: np.ndarray, low_percentile: int = 2, high_percentile: int = 98) -> np.ndarray:
    """
    Normalize an image using percentile-based scaling.
    
    Args:
        image (np.ndarray): The image array to normalize.
        low_percentile (int): The lower percentile for clipping.
        high_percentile (int): The upper percentile for clipping.
    
    Returns:
        np.ndarray: The normalized image array.
    """
    p_low, p_high = np.percentile(image, [low_percentile, high_percentile])
    logging.debug(f"Normalizing image with percentiles: {low_percentile}, {high_percentile}")
    return np.clip(image, p_low, p_high) - p_low / (p_high - p_low)


# --------- UI Setup Functions ----------#

def create_technique_ui_elements(technique: str, tab: Any, show_per_pixel_processing: bool) -> Dict[str, Any]:
    """
    Create UI elements for a specific image processing technique.

    Args:
        technique (str): The image processing technique (e.g., 'nlm', 'speckle').
        tab (Any): The tab in which to place the UI elements.
        show_per_pixel_processing (bool): Whether to display per-pixel processing views.

    Returns:
        Dict[str, Any]: Dictionary of placeholders for various UI elements.
    """
    # Input validation for technique
    if not technique or not isinstance(technique, str):
        logging.error("Invalid technique input provided.")
        raise ValueError("Technique must be a non-empty string.")

    logging.info(f"Creating UI elements for technique: {technique}")

    with tab:
        # Create empty placeholders for dynamic UI elements
        ui_placeholders = {
            'formula': st.empty(),
            'original_image': st.empty()
        }

        # Fetch available filters for the given technique
        filter_options = get_filter_options(technique)
        logging.debug(f"Filter options for {technique}: {filter_options}")

        if selected_filters := create_filter_selection(
            technique, filter_options
        ):
            # Render the selected filter views
            create_filter_views(selected_filters, ui_placeholders, show_per_pixel_processing)
        else:
            st.warning("No views selected. Please select at least one view to display.")
            logging.warning(f"No filters selected for {technique}")

        # Handle per-pixel processing view if enabled
        if show_per_pixel_processing:
            ui_placeholders['zoomed_kernel'] = st.empty()
            logging.info("Per-pixel processing view enabled.")

    return ui_placeholders

# --------- Utility Functions ----------#
def visualize_image(image: np.ndarray, placeholder, config: VisualizationConfig) -> None:
    """
    Visualize an image with optional zooming and overlays.
    
    Args:
        image (np.ndarray): The image array to visualize.
        placeholder: Streamlit placeholder to display the plot.
        config (VisualizationConfig): Configuration for visualization options.
    """
    try:
        # Optional zooming on the image
        if config.zoom:
            image, pixel_x, pixel_y = get_zoomed_image_section(
                image, 
                config.last_processed_pixel.x, 
                config.last_processed_pixel.y, 
                config.kernel_size
            )
        else:
            pixel_x, pixel_y = config.last_processed_pixel.x, config.last_processed_pixel.y

        fig = create_image_plot(image, pixel_x, pixel_y, config)
        placeholder.pyplot(fig)  # Display plot
        plt.close(fig)  # Ensure figure is closed after rendering to free up memory
    except Exception as e:
        logging.error(f"Error while visualizing image: {e}")
        placeholder.error("An error occurred while visualizing the image. Please check the logs for details.")






def get_filter_options(technique: str) -> List[str]:
    """
    Get filter options based on the image processing technique.
    
    Args:
        technique (str): Image processing technique (e.g., 'nlm', 'speckle').
    
    Returns:
        List[str]: A list of filter options for the technique.
    """
    if technique == "speckle":
        return ['Original Image'] + SpeckleResult.get_filter_options()
    elif technique == "nlm":
        return ['Original Image'] + NLMResult.get_filter_options()
    else:
        logging.warning(f"Unknown technique: {technique}")
        return []

def create_filter_selection(technique: str, filter_options: List[str]) -> List[str]:
    """
    Create and return a filter selection UI element.
    
    Args:
        technique (str): Image processing technique.
        filter_options (List[str]): List of available filter options.
    
    Returns:
        List[str]: List of selected filters from the UI.
    """
    # Default views for different techniques
    default_selection = DEFAULT_SPECKLE_VIEW if technique == "speckle" else DEFAULT_NLM_VIEW
    
    # Create multi-select widget for filter views
    selected_filters = st.multiselect(
        "Select views to display",
        filter_options,
        default=default_selection,
        key=f"{technique}_filter_selection"
    )
    
    st.session_state[f'{technique}_selected_filters'] = selected_filters
    logging.info(f"Selected filters for {technique}: {selected_filters}")
    return selected_filters

def create_filter_views(selected_filters: List[str], ui_placeholders: Dict[str, Any], show_per_pixel_processing: bool) -> None:
    """
    Create views for the selected filters in the UI.
    
    Args:
        selected_filters (List[str]): List of selected filters to display.
        ui_placeholders (Dict[str, Any]): Dictionary of placeholders to update.
        show_per_pixel_processing (bool): Whether to display per-pixel processing views.
    """
    columns = st.columns(len(selected_filters))
    
    for i, filter_name in enumerate(selected_filters):
        filter_key = filter_name.lower().replace(" ", "_")
        ui_placeholders[filter_key] = columns[i].empty()  # Create a placeholder for the filter
        
        # If per-pixel processing is enabled, create zoomed-in views
        if show_per_pixel_processing:
            ui_placeholders[f'zoomed_{filter_key}'] = columns[i].expander(f"Zoomed-in {filter_name}", expanded=False).empty()
        logging.debug(f"Created view for {filter_name} with zoom: {show_per_pixel_processing}")

def update_session_state(technique: str, pixels_to_process: int, results: Any) -> None:
    """
    Update session state with processing results.
    
    Args:
        technique (str): Image processing technique.
        pixels_to_process (int): Number of processed pixels.
        results (Any): The result of the image processing.
    """
    st.session_state.update({
        'processed_pixels': pixels_to_process,
        f"{technique}_results": results
    })
    logging.info(f"Session state updated for {technique} with {pixels_to_process} pixels processed.")

def get_last_processed_coordinates(results: Any, processing_details: Any) -> PixelCoordinates:
    """Get the coordinates of the last processed pixel."""
    if isinstance(results, (NLMResult, SpeckleResult)):
        return results.processing_end_coord  # Ensure this returns a PixelCoordinates object
    logging.warning("Unknown result type for getting processed coordinates.")
    return PixelCoordinates(x=processing_details.end_x, y=processing_details.end_y)  # Ensure this returns a PixelCoordinates object
# --------- Helpers ----------#

def run_technique(technique: str, tab: Any, analysis_params: Dict[str, Any]) -> None:
    technique_params = st.session_state.get(f"{technique}_params", {})
    show_per_pixel_processing = analysis_params.get('show_per_pixel_processing', False)
    
    ui_placeholders = create_technique_ui_elements(technique, tab, show_per_pixel_processing)
    st.session_state[f"{technique}_placeholders"] = ui_placeholders
    
    process_params = create_process_params(analysis_params, technique, technique_params)
    
    try:
        if technique == "nlm":
            process_params.analysis_params['use_whole_image'] = technique_params.get('use_whole_image', False)
        
        _, results = process_image(process_params)
        st.session_state[f"{technique}_results"] = results
    except Exception as e:
        print(f"Error processing image for {technique}: {str(e)}")
        st.error(f"An error occurred while processing the image for {technique}. Please check the logs for details.")

def setup_and_run_analysis_techniques(analysis_params: Dict[str, Any]) -> None:
    """Set up and run analysis techniques based on the provided parameters."""
    techniques: List[str] = st.session_state.get('techniques', [])
    tabs: List[Any] = st.session_state.get('tabs', [])

    for technique, tab in zip(techniques, tabs):
        if tab is not None:
            with tab:
                run_technique(technique, tab, analysis_params)
