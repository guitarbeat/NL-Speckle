import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, Tuple, Dict, Any
from matplotlib.collections import LineCollection

from analysis.speckle import process_speckle, SpeckleResult
from analysis.nlm import process_nlm, NLMResult

from streamlit_image_comparison import image_comparison
from PIL import Image


# ===== Constants =====
KERNEL_COLOR = 'red'
SEARCH_WINDOW_COLOR = 'blue'
PIXEL_VALUE_COLOR = 'red'
ZOOMED_IMAGE_SIZE = (8, 8)
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

# ===== Main UI Setup =====
def setup_and_run_analysis_techniques(analysis_params: Dict[str, Any]) -> None:
    for technique in ["speckle", "nlm"]:
        tab_index = 0 if technique == "speckle" else 1
        tab = st.session_state.tabs[tab_index]
        with tab:
            technique_params = st.session_state.get(f"{technique}_params", {})
            placeholders = create_technique_ui_elements(technique, tab, analysis_params['show_per_pixel'])
            st.session_state[f"{technique}_placeholders"] = placeholders

            params = create_process_params(analysis_params, technique, technique_params)
            _, results = process_image(params)
            st.session_state[f"{technique}_results"] = results

# ===== Visualization Functions =====
def visualize_results(image_np: np.ndarray, technique: str, analysis_params: Dict[str, Any], results: Any, show_per_pixel: bool) -> None:
    from utils import calculate_processing_details

    details = calculate_processing_details(image_np, analysis_params['kernel_size'], analysis_params['max_pixels'])

    end_x, end_y = get_end_processed_pixel(results, details)

    kernel_matrix, original_value = extract_kernel_from_image(
        image_np, end_x, end_y, analysis_params['kernel_size']
    )

    placeholders = st.session_state.get(f'{technique}_placeholders', {})

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

    visualize_analysis_results(**visualization_params)

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

    visualize_image(
        image_np, placeholders['original_image'], end_x, end_y, kernel_size,
        vmin=vmin, vmax=vmax, title="Original Image",
        technique=analysis_type, search_window_size=search_window_size,
        show_kernel=show_per_pixel, show_per_pixel=show_per_pixel
    )

    filter_options, specific_params = prepare_filter_options_and_parameters(results, (end_x, end_y))

    visualize_filter_results(
        filter_options, placeholders, params, (end_x, end_y),
        kernel_size, analysis_type, search_window_size
    )

    if show_per_pixel:
        for key, placeholder in placeholders.items():
            if key.startswith('zoomed_'):
                filter_name = key[7:].replace('_', ' ').title()
                filter_data = image_np if filter_name == 'Original Image' else filter_options.get(filter_name)
                if filter_data is not None:
                    visualize_image(
                        filter_data, placeholder, end_x, end_y, kernel_size,
                        vmin=np.min(filter_data), vmax=np.max(filter_data),
                        title=f"Zoomed-In {filter_name}", technique=analysis_type,
                        search_window_size=search_window_size, zoom=True, show_kernel=True,
                        show_per_pixel=show_per_pixel
                    )

        specific_params.update({
            'x': end_x, 'y': end_y, 'input_x': end_x, 'input_y': end_y,
            'kernel_size': kernel_size, 'kernel_matrix': kernel_matrix, 'original_value': original_value
        })
        display_analysis_formula(specific_params, placeholders, analysis_type)

def visualize_filter_results(
    filter_options: Dict[str, Any],
    placeholders: Dict[str, Any],
    params: Dict[str, Any],
    end_processed_pixel: Tuple[int, int],
    kernel_size: int,
    analysis_type: str,
    search_window_size: Optional[int]
) -> None:
    show_per_pixel = params['show_per_pixel']
    end_x, end_y = end_processed_pixel

    for filter_name, filter_data in filter_options.items():
        if filter_data is not None:
            key = filter_name.lower().replace(" ", "_")
            if key in placeholders:
                visualize_image(
                    filter_data, placeholders[key], end_x, end_y, kernel_size,
                    vmin=np.min(filter_data), vmax=np.max(filter_data),
                    title=filter_name, technique=analysis_type, search_window_size=search_window_size,
                    show_kernel=show_per_pixel, show_per_pixel=show_per_pixel
                )

def visualize_image(
    image: np.ndarray,
    placeholder,
    pixel_x: Optional[int],
    pixel_y: Optional[int],
    kernel_size: int,
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
        image, pixel_x, pixel_y = get_zoomed_image_section(image, pixel_x, pixel_y, kernel_size)

    fig = create_image_plot(
        image, pixel_x, pixel_y, kernel_size, title,
        plot_search_window=search_window_size if show_search_window else None,
        zoom=zoom, vmin=vmin, vmax=vmax, show_kernel=show_kernel
    )

    placeholder.pyplot(fig)

def create_image_plot(
    plot_image: np.ndarray,
    center_x: int,
    center_y: int,
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
        draw_kernel_overlay(ax, center_x, center_y, plot_kernel_size)

    if plot_search_window is not None:
        draw_search_window_overlay(ax, plot_image, center_x, center_y, plot_search_window)

    if zoom:
        draw_pixel_value_annotations(ax, plot_image)

    fig.tight_layout(pad=2)
    return fig

# ===== Sidebar UI Functions =====
def setup_sidebar() -> Dict[str, Any]:
    """Set up the sidebar with image processing settings."""
    try:
        st.sidebar.title("Image Processing Settings")
        
        st.sidebar.markdown("### 📷 Image Source")
        image = create_image_source_ui()
        if image is None:
            return None

        st.sidebar.markdown("### 🎨 Color Map")
        cmap = create_color_map_ui()
        
        display_options = create_display_options_ui(image, 7)  # Use default kernel_size of 7
        advanced_options = create_advanced_options_ui(image)

        return {
            "image": image,
            "cmap": cmap,
            **display_options,
            **advanced_options
        }
    except Exception as e:
        st.sidebar.error(f"An error occurred while setting up the sidebar: {str(e)}")
        return None

def create_image_source_ui() -> Image.Image:
    """Create the image source selection UI in the sidebar."""
    image_source = st.radio("Select Image Source", ("Preloaded Images", "Upload Image"))
    
    if image_source == "Preloaded Images":
        selected_image = st.selectbox("Select Image", list(PRELOADED_IMAGES))
        image = load_image(PRELOADED_IMAGES[selected_image])
    else:
        uploaded_file = st.file_uploader("Upload Image")
        image = load_image(uploaded_file) if uploaded_file else None

    if image is None:
        st.warning('Please select or upload an image.')
        return None

    st.image(image, "Input Image", use_column_width=True)
    return image

def create_color_map_ui() -> str:
    """Create the color map selection UI in the sidebar."""
    cmap = st.session_state.get('cmap', COLOR_MAPS[0])
    cmap = st.selectbox("Select Color Map", COLOR_MAPS, index=COLOR_MAPS.index(cmap))
    st.session_state.cmap = cmap
    return cmap

def create_display_options_ui(image: Image.Image, kernel_size: int) -> Dict[str, Any]:
    """Create the display options UI in the sidebar."""
    col1, col2 = st.columns(2)
    show_per_pixel = col1.toggle("Show Per-Pixel Processing Steps", value=False)
    max_pixels = (image.width - kernel_size + 1) * (image.height - kernel_size + 1)
    
    pixels_to_process = max_pixels
    if show_per_pixel:
        percentage = col2.slider("Percentage of Pixels to Process", 
                                 min_value=1, max_value=100, value=100, step=1,
                                 key=f"percentage_slider_{kernel_size}")
        st.session_state.percentage = percentage
        pixels_to_process = int(max_pixels * percentage / 100)

    return {"show_per_pixel": show_per_pixel, "max_pixels": max_pixels, "pixels_to_process": pixels_to_process}

def create_advanced_options_ui(image: Image.Image) -> Dict[str, Any]:
    """Create the advanced options UI in the sidebar."""
    with st.expander("🔬 Advanced Options"):
        add_noise = st.checkbox("Add Gaussian Noise", value=False,
                                help="Add Gaussian noise to the image")
        if add_noise:
            noise_mean = st.number_input("Noise Mean", min_value=0.0, max_value=1.0, value=0.0, step=0.01, format="%.2f")
            noise_std = st.number_input("Noise Std", min_value=0.0, max_value=1.0, value=0.1, step=0.01, format="%.2f")
            image_np = np.clip(np.array(image) / 255.0 + np.random.normal(noise_mean, noise_std, np.array(image).shape), 0, 1)
        else:
            image_np = np.array(image) / 255.0
    return {"image_np": image_np, "add_noise": add_noise}

# ===== Image Comparison Functions =====
def handle_image_comparison(tab, cmap_name: str, images: Dict[str, np.ndarray]):
    """Handle the image comparison functionality."""
    with tab:
        st.header("Image Comparison")
        if not images:
            st.warning("No images available for comparison.")
            return

        available_images = list(images.keys())
        
        # Render UI for selecting images to compare
        col1, col2 = st.columns(2)
        image_choice_1 = col1.selectbox('Select first image to compare:', [''] + available_images, index=0)
        image_choice_2 = col2.selectbox('Select second image to compare:', [''] + available_images, index=0)

        if image_choice_1 and image_choice_2:
            img1, img2 = images[image_choice_1], images[image_choice_2]
            display_comparison(img1, img2, image_choice_1, image_choice_2, cmap_name)
        else:
            st.info("Select two images to compare.")

def prepare_comparison_images() -> Dict[str, np.ndarray]:
    """Prepare the images for comparison."""
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


def display_comparison(img1: np.ndarray, img2: np.ndarray, label1: str, label2: str, cmap_name: str):
    """Display the image comparison using streamlit-image-comparison."""
    if label1 != label2:
        # Normalize and colorize the images
        def normalize_and_colorize(img):
            normalized = (img - np.min(img)) / (np.max(img) - np.min(img))
            colored = plt.get_cmap(cmap_name)(normalized)[:, :, :3]
            return (colored * 255).astype(np.uint8)
        
        img1_uint8, img2_uint8 = map(normalize_and_colorize, [img1, img2])
        
        image_comparison(img1=img1_uint8, img2=img2_uint8, label1=label1, label2=label2, make_responsive=True)
        st.subheader("Selected Images")
        st.image([img1_uint8, img2_uint8], caption=[label1, label2])
    else:
        st.error("Please select two different images for comparison.")
        st.image(np.abs(img1 - img2), caption="Difference Map", use_column_width=True)

# ===== Image Processing Functions =====
def process_image(params: Dict[str, Any]) -> Tuple[Dict[str, Any], Any]:
    try:
        image_np = params['image_np']
        technique = params['technique']
        analysis_params = params['analysis_params']

        results = run_analysis_technique(image_np, technique, analysis_params)
        if params.get('return_processed_only', False):
            return params, results

        if params['handle_visualization']:
            visualize_results(image_np, technique, analysis_params, results, params.get('show_per_pixel', False))

        if params['update_state']:
            update_session_state(technique, analysis_params['pixels_to_process'], results)

        return params, results
    except Exception as e:
        error_message = f"An error occurred during image processing: {str(e)}"
        st.error(error_message)
        return params, None

def run_analysis_technique(image_np: np.ndarray, technique: str, analysis_params: Dict[str, Any]) -> Any:
    if technique == "speckle":
        return process_speckle(image_np, analysis_params['kernel_size'], analysis_params['max_pixels'])
    elif technique == "nlm":
        return process_nlm(
            image_np, analysis_params['kernel_size'], analysis_params['max_pixels'],
            analysis_params['search_window_size'], analysis_params['filter_strength']
        )
    else:
        raise ValueError(f"Unknown technique: {technique}")

def extract_kernel_from_image(image_np: np.ndarray, end_x: int, end_y: int, kernel_size: int) -> tuple[np.ndarray, float]:
    """Extract the kernel from the image around the specified pixel."""
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
    """Update the session state with the processed pixels and results."""
    st.session_state.processed_pixels = pixels_to_process
    st.session_state[f"{technique}_results"] = results
 

# ===== Filter Options and Parameters Functions =====
def prepare_filter_options_and_parameters(results: Any, end_processed_pixel: Tuple[int, int]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    end_x, end_y = end_processed_pixel

    filter_options, specific_params = get_filter_options_and_params(results, end_x, end_y)

    filter_options = {k: v for k, v in filter_options.items() if v is not None}

    if isinstance(results, SpeckleResult) or hasattr(results, 'kernel_size'):
        specific_params['total_pixels'] = results.kernel_size ** 2

    return filter_options, {k: v for k, v in specific_params.items() if v is not None}

def get_filter_options_and_params(results: Any, end_x: int, end_y: int) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    if isinstance(results, NLMResult):
        return get_nlm_filter_options_and_params(results, end_x, end_y)
    elif isinstance(results, SpeckleResult):
        return get_speckle_filter_options_and_params(results, end_x, end_y)
    else:
        return {}, {}

def get_nlm_filter_options_and_params(results: NLMResult, end_x: int, end_y: int) -> Tuple[Dict[str, Any], Dict[str, Any]]:
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
    return filter_options, specific_params

def get_speckle_filter_options_and_params(results: SpeckleResult, end_x: int, end_y: int) -> Tuple[Dict[str, Any], Dict[str, Any]]:
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
    return filter_options, specific_params

# ===== Technique Parameter Functions =====
def get_technique_params(technique: str, analysis_params: Dict[str, Any]) -> Dict[str, Any]:
    """Get the technique-specific parameters based on the selected technique."""
    kernel_size = st.slider('Kernel Size', min_value=3, max_value=21, value=7, step=2, key=f'kernel_size_{technique}')
    
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
    """Get NLM-specific parameters."""
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
    """Create the parameters for the image processing function."""
    return {
        'image_np': analysis_params['image_np'],
        'technique': technique,
        'analysis_params': technique_params,
        'pixels_to_process': technique_params['max_pixels'],
        'update_state': True,
        'handle_visualization': True,
        'show_per_pixel': analysis_params['show_per_pixel']
    }

# ===== UI Element Creation Functions =====
def create_technique_ui_elements(technique, tab, show_per_pixel: bool) -> Dict[str,Any]:
    """Create UI elements for a specific technique."""
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
                if show_per_pixel:
                    placeholders[f'zoomed_{key}'] = st.expander(f"Zoomed-in {filter_name}", expanded=False).empty()
        
        if show_per_pixel:
            placeholders['zoomed_kernel'] = st.empty()
        
    return placeholders

# ===== Utility Functions =====
def load_image(image_path: str) -> Image.Image:
    """Load an image from the given path and convert it to grayscale."""
    try:
        image = Image.open(image_path).convert('L')
        return image
    except Exception as e:
        st.error(f"Failed to load the image: {str(e)}")
        return None
    
def get_end_processed_pixel(results: Any, details: Any) -> Tuple[int, int]:
    if isinstance(results, NLMResult):
        return results.processing_end_coord
    elif isinstance(results, SpeckleResult):
        return results.processing_start_coord
    else:
        return details.end_x, details.end_y

def draw_kernel_overlay(ax: plt.Axes, center_x: int, center_y: int, kernel_size: int) -> None:
    kernel_left = center_x - kernel_size // 2
    kernel_top = center_y - kernel_size // 2
    ax.add_patch(plt.Rectangle((kernel_left - 0.5, kernel_top - 0.5), kernel_size, kernel_size,
                               edgecolor=KERNEL_COLOR, linewidth=1, facecolor="none"))
    lines = ([[(kernel_left + i - 0.5, kernel_top - 0.5), (kernel_left + i - 0.5, kernel_top + kernel_size - 0.5)] for i in range(1, kernel_size)] +
             [[(kernel_left - 0.5, kernel_top + i - 0.5), (kernel_left + kernel_size - 0.5, kernel_top + i - 0.5)] for i in range(1, kernel_size)])
    ax.add_collection(LineCollection(lines, colors=KERNEL_COLOR, linestyles=':', linewidths=0.5))

    ax.add_patch(plt.Rectangle((center_x - 0.5, center_y - 0.5), 1, 1,
                               edgecolor=SEARCH_WINDOW_COLOR, linewidth=0.5, facecolor=SEARCH_WINDOW_COLOR, alpha=0.5))

def draw_search_window_overlay(ax: plt.Axes, image: np.ndarray, center_x: int, center_y: int, search_window: int) -> None:
    height, width = image.shape[:2]
    if search_window >= max(height, width):
        rect = plt.Rectangle((-0.5, -0.5), width, height, 
                             edgecolor=SEARCH_WINDOW_COLOR, linewidth=2, facecolor="none")
    else:
        half_window = search_window // 2
        sw_left = max(-0.5, center_x - half_window - 0.5)
        sw_top = max(-0.5, center_y - half_window - 0.5)
        sw_width = min(width - sw_left, search_window)
        sw_height = min(height - sw_top, search_window)
        rect = plt.Rectangle((sw_left, sw_top), sw_width, sw_height,
                             edgecolor=SEARCH_WINDOW_COLOR, linewidth=2, facecolor="none")
    ax.add_patch(rect)

def draw_pixel_value_annotations(ax: plt.Axes, image: np.ndarray) -> None:
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            ax.text(j, i, f"{image[i, j]:.2f}", ha="center", va="center", color=PIXEL_VALUE_COLOR, fontsize=8)

def get_zoomed_image_section(image: np.ndarray, center_x: int, center_y: int, zoom_size: int) -> Tuple[np.ndarray, int, int]:
    top = max(0, center_y - zoom_size // 2)
    left = max(0, center_x - zoom_size // 2)
    zoomed_image = image[top:min(image.shape[0], top + zoom_size),
                         left:min(image.shape[1], left + zoom_size)]
    return zoomed_image, zoom_size // 2, zoom_size // 2
