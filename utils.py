# Library imports
import numpy as np
import streamlit as st
from analysis.speckle import process_speckle, SPECKLE_FORMULA_CONFIG
from analysis.nlm import process_nlm, NLM_FORMULA_CONFIG
from formula import display_formula
from viz import visualize_image
from image_processing import calculate_processing_details


# Constants
FILTER_OPTIONS = {
    "speckle": ["Mean Filter", "Std Dev Filter", "Speckle Contrast"],
    "nlm": ["Weight Map", "NL-Means Image", "Difference Map"]
}

# ----------------------------- Shared Utilities ----------------------------- #


# Prepares filter options and specific parameters based on the analysis type
def prepare_filter_options_and_params(results, first_pixel):
    """
    Prepares filter options and specific parameters for both NLM and Speckle analysis.
    
    Args:
        results: The analysis results (NLMResult or SpeckleResult).
        first_pixel: The coordinates of the first pixel.
    
    Returns:
        A tuple containing the filter options and specific parameters.
    """
    first_x, first_y = first_pixel
    
    filter_options = {
        "Mean Filter": getattr(results, 'mean_filter', None),
        "Std Dev Filter": getattr(results, 'std_dev_filter', None),
        "Speckle Contrast": getattr(results, 'speckle_contrast_filter', None),
        "NL-Means Image": getattr(results, 'processed_image', None),
        "Weight Map": getattr(results, 'normalized_weight_map', None),
        "Difference Map": getattr(results, 'difference_map', None)
    }
    
    # Remove None values from filter_options
    filter_options = {k: v for k, v in filter_options.items() if v is not None}
    
    specific_params = {
        "filter_strength": getattr(results, 'filter_strength', None),
        "search_window_size": getattr(results, 'search_window_size', None),
        "total_pixels": getattr(results, 'pixels_processed', None),
        "std": getattr(results, 'first_pixel_std_dev', None),
        "mean": getattr(results, 'first_pixel_mean', None),
        "sc": getattr(results, 'first_pixel_speckle_contrast', None),
    }
    
    # Safely get nlm_value
    nl_means_image = filter_options.get("NL-Means Image")
    if nl_means_image is not None and first_y < nl_means_image.shape[0] and first_x < nl_means_image.shape[1]:
        specific_params["nlm_value"] = nl_means_image[first_y, first_x]
    
    # Remove None values from specific_params
    specific_params = {k: v for k, v in specific_params.items() if v is not None}
    
    return filter_options, specific_params


# Processes and visualizes an image based on the given parameters
# TODO: Split this function into separate processing and visualization functions
# TODO: Consider using a configuration object instead of a large params dictionary
def process_and_visualize_image(params):
    """
    Process and visualize an image based on the given parameters.
    
    Args:
    params: A dictionary containing all necessary parameters for processing and visualization.
    
    Returns:
    The input parameters and the processing results (if successful).
    """
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
                'original_value': original_value,
                'analysis_type': technique
            }
            
            visualize_results(**visualization_params)
        
        # Update state
        if params['update_state']:
            st.session_state.processed_pixels = params['pixels_to_process']
            st.session_state[f"{technique}_results"] = results
        
        return params, results
    except Exception as e:
        st.error(f"Error during image processing and visualization: {str(e)}")
        st.exception(e)
        return params, None

# Processes images using both speckle and NLM techniques
# TODO: Split this function into separate processing and visualization functions
# TODO: Consider using a configuration object instead of a large params dictionary
def process_techniques(analysis_params):
    """
    Process images using both speckle and NLM techniques.
    
    Args:
    analysis_params: Parameters for the analysis, including image data and processing settings.
    """
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
        except Exception:
            st.error(f"Failed to process {technique} technique. Please check your inputs and try again.")

# Creates UI elements for the given technique and tab
# TODO: Separate UI creation logic from data processing logic
# TODO: Consider using a state management solution for complex UI interactions
def create_ui_elements(technique, tab, show_full_processed):
    """
    Create UI elements for the given technique and tab.
    
    Args:
    technique: The image processing technique ('speckle' or 'nlm').
    tab: The Streamlit tab to render elements in.
    show_full_processed: Whether to show the full processed image or not.
    
    Returns:
    A dictionary of placeholders for various UI elements.
    """
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

# Extracts and prepares kernel information from the image
# TODO: Consider using numpy operations for better performance in kernel extraction
def extract_kernel_info(image_np, last_x, last_y, kernel_size):
    """
    Extract and prepare kernel information from the image.

    Args:
    image_np: The input image as a NumPy array.
    last_x: The x-coordinate of the center pixel.
    last_y: The y-coordinate of the center pixel.
    kernel_size: The size of the kernel.

    Returns:
    A tuple containing the kernel matrix and the original center pixel value.
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

# Visualizes the results of the analysis, including original and processed images
# TODO: Implement a more modular visualization system, possibly using a plugin architecture
# TODO: Separate the visualization logic from the data preparation logic
# TODO: Consider using asynchronous rendering for large images or complex visualizations
# TODO: Implement caching for visualizations to improve performance on repeated views
def visualize_results(image_np, results, placeholders, params, first_pixel, kernel_size, kernel_matrix, original_value, analysis_type):
    """
    Visualizes the results of the analysis, including original and processed images.
    
    Args:
        image_np: The original image.
        results: The analysis results.
        placeholders: The Streamlit placeholders for the image displays.
        params: The analysis parameters.
        first_pixel: The coordinates of the first pixel.
        kernel_size: Size of the kernel.
        kernel_matrix: The kernel matrix.
        original_value: The original value of the first pixel.
        analysis_type: The type of analysis ('nlm' or 'speckle').
    """
    first_x, first_y = first_pixel
    vmin, vmax = np.min(image_np), np.max(image_np)
    show_full_processed = params['show_full_processed']
    cmap = params['analysis_params']['cmap']
    search_window_size = params['analysis_params'].get('search_window_size') if analysis_type == 'nlm' else None

    visualize_image(image_np, placeholders['original_image'], first_x, first_y, kernel_size, cmap, 
                    show_full_processed, vmin, vmax, "Original Image", analysis_type, search_window_size)
    
    if not show_full_processed:
        visualize_image(image_np, placeholders['zoomed_original_image'], first_x, first_y, kernel_size, 
                        cmap, show_full_processed, vmin, vmax, "Zoomed-In Original Image", zoom=True)

    filter_options, specific_params = prepare_filter_options_and_params(results, (first_x, first_y))
    
    for filter_name, filter_data in filter_options.items():
        if filter_data is not None:
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

    formula_config = NLM_FORMULA_CONFIG if analysis_type == 'nlm' else SPECKLE_FORMULA_CONFIG
    display_formula(formula_config, specific_params, placeholders['formula'])


    