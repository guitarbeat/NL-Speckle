import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from typing import Tuple, List, Optional, Union, Any, Dict
from streamlit_image_comparison import image_comparison
from numba import jit

# ---------------------------- Cached Functions ---------------------------- #

@st.cache_data(persist=True)
def create_combined_plot(plot_image: np.ndarray, plot_x: int, plot_y: int, plot_kernel_size: int, 
                         title: str, plot_cmap: str = "viridis", plot_search_window: Optional[Union[str, int]] = None, 
                         zoom: bool = False, vmin: Optional[float] = None, vmax: Optional[float] = None) -> plt.Figure:
    fig, ax = plt.subplots(1, 1)
    
    # Use the provided vmin and vmax for consistent color mapping
    ax.imshow(plot_image, cmap=plot_cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.axis('off')

    if not zoom:
        # Mark the center of the kernel
        ax.add_patch(plt.Rectangle((plot_x - 0.5, plot_y - 0.5), 1, 1, 
                                   edgecolor="r", linewidth=0.5, facecolor="r", alpha=0.2))
        
        # Kernel overlay for the original image
        if title == "Original Image with Current Kernel":
            kx, ky = int(plot_x - plot_kernel_size // 2), int(plot_y - plot_kernel_size // 2)
            ax.add_patch(plt.Rectangle((kx - 0.5, ky - 0.5), plot_kernel_size, plot_kernel_size, 
                                       edgecolor="r", linewidth=1, facecolor="none"))
            lines = ([[(kx + i - 0.5, ky - 0.5), (kx + i - 0.5, ky + plot_kernel_size - 0.5)] for i in range(1, plot_kernel_size)] +
                     [[(kx - 0.5, ky + i - 0.5), (kx + plot_kernel_size - 0.5, ky + i - 0.5)] for i in range(1, plot_kernel_size)])
            ax.add_collection(LineCollection(lines, colors='red', linestyles=':', linewidths=0.5))
            
        # Search window overlay (for NLM)
        if plot_search_window == "full":
            rect = plt.Rectangle((-0.5, -0.5), plot_image.shape[1], plot_image.shape[0], 
                                 edgecolor="blue", linewidth=2, facecolor="none")
            ax.add_patch(rect)
        elif isinstance(plot_search_window, int):
            half_window = plot_search_window // 2
            rect = plt.Rectangle((plot_x - half_window - 0.5, plot_y - half_window - 0.5), 
                                 plot_search_window, plot_search_window, 
                                 edgecolor="blue", linewidth=1, facecolor="none")
            ax.add_patch(rect)
    else:
        # Value annotations for zoomed view
        for i in range(plot_image.shape[0]):
            for j in range(plot_image.shape[1]):
                ax.text(j, i, f"{plot_image[i, j]:.2f}", ha="center", va="center", color="red", fontsize=8)

    fig.tight_layout(pad=2)
    return fig

@st.cache_data(persist=True)
def process_image(technique: str, image: np.ndarray, kernel_size: int, max_pixels: int, height: int, width: int, 
                  search_window_size: Optional[int], filter_strength: float) -> Tuple[np.ndarray, ...]:
    half_kernel = kernel_size // 2
    
    # Calculate the first valid pixel coordinates
    first_x = first_y = half_kernel
    
    # Calculate the number of valid pixels
    valid_height = height - kernel_size + 1
    valid_width = width - kernel_size + 1
    total_valid_pixels = valid_height * valid_width
    
    # Use the minimum of max_pixels and total_valid_pixels
    pixels_to_process = min(max_pixels, total_valid_pixels)
    
    if technique == "speckle":
        return calculate_speckle(image, kernel_size, pixels_to_process, height, width, half_kernel, first_x, first_y)
    elif technique == "nlm":
        return calculate_nlm(image, kernel_size, search_window_size, filter_strength, pixels_to_process, height, width, half_kernel, first_x, first_y)
    else:
        raise ValueError(f"Unknown technique: {technique}")

@st.cache_data(persist=True)
def calculate_nlm(image: np.ndarray, kernel_size: int, search_size: Optional[int], filter_strength: float, pixels_to_process: int, height: int, width: int, half_kernel: int, first_x: int, first_y: int) -> Tuple[np.ndarray, np.ndarray, int, int, float]:
    
    # Initialize output arrays for NLM(x, y) and W(x, y)
    denoised_image = np.zeros((height, width), dtype=np.float32)
    weight_sum_map = np.zeros((height, width), dtype=np.float32)

    @jit(nopython=True)
    def process_nlm(image, denoised_image, weight_sum_map, kernel_size, filter_strength, pixels_to_process, search_size, height, width, half_kernel, first_x, first_y):
        for pixel in range(pixels_to_process):
            row = first_y + pixel // (width - kernel_size + 1)
            col = first_x + pixel % (width - kernel_size + 1)
            
            # Define P(x, y): the neighborhood around the current pixel
            center_patch = image[row-half_kernel:row+half_kernel+1, col-half_kernel:col+half_kernel+1]
            
            denoised_value = 0.0
            weight_sum = 0.0  # This represents W(x, y) in the formula

            # Define Ω: the search window
            if search_size is None:
                search_y_start, search_y_end = 0, height
                search_x_start, search_x_end = 0, width
            else:
                search_y_start = max(0, row - search_size // 2)
                search_y_end = min(height, row + search_size // 2 + 1)
                search_x_start = max(0, col - search_size // 2)
                search_x_end = min(width, col + search_size // 2 + 1)

            # Iterate over Ω
            for i in range(search_y_start, search_y_end):
                for j in range(search_x_start, search_x_end):
                    if j < half_kernel or j >= width - half_kernel or i < half_kernel or i >= height - half_kernel:
                        continue
                    
                    # Define P(i, j): the neighborhood around the comparison pixel
                    patch = image[i-half_kernel:i+half_kernel+1, j-half_kernel:j+half_kernel+1]
                    
                    # Calculate ||P(x, y) - P(i, j)||^2
                    distance = np.sum((center_patch - patch)**2)
                    
                    # Calculate w((x, y), (i, j)) = exp(-||P(x, y) - P(i, j)||^2 / h^2)
                    weight = np.exp(-distance / (filter_strength ** 2))
                    
                    # Accumulate w((x, y), (i, j)) * I(i, j)
                    denoised_value += image[i, j] * weight
                    # Accumulate W(x, y)
                    weight_sum += weight
                    # Track total weight for each pixel (not in original formula, used for visualization)
                    weight_sum_map[i, j] += weight

            # Calculate NLM(x, y) = (1 / W(x, y)) * Σ w((x, y), (i, j)) * I(i, j)
            denoised_image[row, col] = denoised_value / weight_sum if weight_sum > 0 else image[row, col]

        return denoised_image, weight_sum_map

    denoised_image, weight_sum_map = process_nlm(image, denoised_image, weight_sum_map, kernel_size, filter_strength, pixels_to_process, search_size, height, width, half_kernel, first_x, first_y)

    # Normalize weight_sum_map for visualization (not part of the original NLM formula)
    max_weight = np.max(weight_sum_map)
    normalized_weight_map = weight_sum_map / max_weight if max_weight > 0 else weight_sum_map

    return denoised_image, normalized_weight_map, first_x, first_y, weight_sum_map[first_y, first_x]

@st.cache_data(persist=True)
def calculate_speckle(image: np.ndarray, kernel_size: int, pixels_to_process: int, height: int, width: int, half_kernel: int, first_x: int, first_y: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, int, float, float, float]:
    mean_filter = np.zeros((height, width), dtype=np.float32)
    std_dev_filter = np.zeros((height, width), dtype=np.float32)
    sc_filter = np.zeros((height, width), dtype=np.float32)

    @jit(nopython=True, parallel=True)
    def process_pixels(image, mean_filter, std_dev_filter, sc_filter, kernel_size, width, pixels_to_process, half_kernel, first_x, first_y):
        for pixel in range(pixels_to_process):
            row = first_y + pixel // (width - kernel_size + 1)
            col = first_x + pixel % (width - kernel_size + 1)

            local_window = image[row-half_kernel:row+half_kernel+1, col-half_kernel:col+half_kernel+1]
            
            local_mean = np.mean(local_window)
            local_std = np.std(local_window)
            
            if local_mean != 0 and local_std != 0:
                speckle_contrast = local_std / local_mean
            else:
                speckle_contrast = 0

            mean_filter[row, col] = local_mean
            std_dev_filter[row, col] = local_std
            sc_filter[row, col] = speckle_contrast

        return mean_filter, std_dev_filter, sc_filter

    mean_filter, std_dev_filter, sc_filter = process_pixels(image, mean_filter, std_dev_filter, sc_filter, kernel_size, width, pixels_to_process, half_kernel, first_x, first_y)

    return mean_filter, std_dev_filter, sc_filter, first_x, first_y, mean_filter[first_y, first_x], std_dev_filter[first_y, first_x], sc_filter[first_y, first_x]

# ---------------------------- UI Components and Layout ---------------------------- #

def create_placeholders_and_sections(technique: str, tab: st.delta_generator.DeltaGenerator, show_full_processed: bool) -> Dict[str, Any]:
    """
    Create placeholders and sections for the Streamlit UI based on the selected technique.

    Args:
        technique (str): The image processing technique ("speckle" or "nlm").
        tab (st.delta_generator.DeltaGenerator): The Streamlit tab to display the UI components.
        show_full_processed (bool): Whether to show the full processed image or zoomed views.

    Returns:
        Dict[str, Any]: A dictionary of placeholders for the UI components.
    """
    with tab:
        placeholders = {
            'formula': st.empty(),
            'original_image': st.empty()  # Always create a placeholder for the original image
        }

        filter_options = {
            "speckle": ["Mean Filter", "Std Dev Filter", "Speckle Contrast"],
            "nlm": ["Weight Map", "NL-Means Image", "Difference Map"]
        }

        selected_filters = {
            "speckle": ["Speckle Contrast"],
            "nlm": ["NL-Means Image"]
        }

        selected_filters = st.multiselect(
            "Select views to display",
            filter_options[technique],
            default=selected_filters[technique]
        )

        columns = st.columns(len(selected_filters) + 1)  # +1 for the original image

        # Create placeholder for original image
        with columns[0]:
            if not show_full_processed:
                placeholders['original_image'], placeholders['zoomed_original_image'] = create_section("Original Image", expanded_main=True)
            else:
                placeholders['original_image'] = st.empty()

        # Create placeholders for selected filters
        for i, filter_name in enumerate(selected_filters, start=1):
            with columns[i]:
                key = filter_name.lower().replace(" ", "_")
                if show_full_processed:
                    placeholders[key] = st.empty()
                else:
                    placeholders[key], placeholders[f'zoomed_{key}'] = create_section(filter_name, expanded_main=True)

        if not show_full_processed:
            placeholders['zoomed_kernel'] = placeholders.get('zoomed_kernel', st.empty())

        return placeholders

def create_section(title: str, expanded_main: bool = False, expanded_zoomed: bool = False):
    """
    Create a section with main and zoomed views in the Streamlit UI.

    Args:
        title (str): The title of the section.
        expanded_main (bool): Whether the main view should be expanded by default.
        expanded_zoomed (bool): Whether the zoomed view should be expanded by default.

    Returns:
        (st.empty, st.empty): Placeholders for the main and zoomed views.
    """
    with st.expander(title, expanded=expanded_main):
        main_placeholder = st.empty()
        zoomed_placeholder = st.expander(f"Zoomed-in {title.split()[0]}", expanded=expanded_zoomed).empty()
    return main_placeholder, zoomed_placeholder

def handle_image_comparison(tab, cmap_name: str, images: Dict[str, np.ndarray]):
    """
    Display an interactive image comparison tool in the Streamlit UI.

    Args:
        tab: The Streamlit tab to display the image comparison tool.
        cmap_name (str): The name of the colormap to apply to the images.
        images (Dict[str, np.ndarray]): A dictionary of images to compare, with keys as image names and values as image data.
    """
    with tab:
        st.header("Image Comparison")
        
        available_images = list(images.keys())
        col1, col2 = st.columns(2)
        image_choice_1 = col1.selectbox('Select first image to compare:', [''] + available_images, index=0)
        image_choice_2 = col2.selectbox('Select second image to compare:', [''] + available_images, index=0)
        
        if image_choice_1 and image_choice_2:
            if image_choice_1 != image_choice_2:
                cmap = plt.get_cmap(cmap_name)
                img1, img2 = images[image_choice_1], images[image_choice_2]
                
                # Normalize images and apply colormap
                def normalize_and_apply_cmap(img):
                    normalized = (img - np.min(img)) / (np.max(img) - np.min(img))
                    return (cmap(normalized)[:, :, :3] * 255).astype(np.uint8)

                img1_uint8, img2_uint8 = map(normalize_and_apply_cmap, [img1, img2])
                
                # Display images and comparison
                image_comparison(img1=img1_uint8, img2=img2_uint8, label1=image_choice_1, label2=image_choice_2, make_responsive=True)
                st.subheader("Selected Images")
                st.image([img1_uint8, img2_uint8], caption=[image_choice_1, image_choice_2])
            else:
                st.error("Please select two different images for comparison.")
                # Display additional comparison options
                st.subheader("Comparison Options")
                diff_map = np.abs(img1 - img2)
                st.image(diff_map, caption="Difference Map", use_column_width=True)
                
        else:
            st.info("Select two images to compare.")


# ---------------------------- Image Visualization ---------------------------- #

def process_and_visualize_image(image: np.ndarray, kernel_size: int, x: int, y: int, 
                                results: Tuple[np.ndarray, ...], cmap: str, technique: str, 
                                placeholders: Dict[str, Any],
                                search_window_size: Optional[int] = None,
                                show_full_processed: bool = False):
    # Calculate vmin and vmax for consistent color mapping
    vmin, vmax = np.min(image), np.max(image)

    # Always display original image
    if show_full_processed:
        fig_original = plt.figure()
        plt.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax)
        plt.axis('off')
        plt.title("Original Image")
    else:
        # Use the last processed pixel coordinates (x, y) for kernel annotation
        fig_original = create_combined_plot(image, x, y, kernel_size, "Original Image with Current Kernel", cmap, 
                                            search_window_size if technique == "nlm" else None, vmin=vmin, vmax=vmax)
    
    placeholders['original_image'].pyplot(fig_original)

    if not show_full_processed:
        # Create zoomed view of original image
        zoom_size = kernel_size
        ky = int(max(0, y - zoom_size // 2))
        kx = int(max(0, x - zoom_size // 2))
        zoomed_original = image[ky:min(image.shape[0], ky + zoom_size),
                                kx:min(image.shape[1], kx + zoom_size)]
        fig_zoom_original = create_combined_plot(zoomed_original, zoom_size // 2, zoom_size // 2, zoom_size, 
                                               "Zoomed-In Original Image", cmap, zoom=True, vmin=vmin, vmax=vmax)
        if 'zoomed_original_image' in placeholders and placeholders['zoomed_original_image'] is not None:
            placeholders['zoomed_original_image'].pyplot(fig_zoom_original)

    # Process and display results based on technique
    if technique == "speckle":
        filter_options = {
            "Mean Filter": results[0],
            "Std Dev Filter": results[1],
            "Speckle Contrast": results[2]
        }
    elif technique == "nlm":
        denoised_image, weight_sum_map = results[:2]
        filter_options = {
            "NL-Means Image": denoised_image,
            "Weight Map": weight_sum_map,
            "Difference Map": np.abs(image - denoised_image)
        }
    else:
        filter_options = {}

    # Display filter results
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
                # Use the last processed pixel coordinates (x, y) for kernel annotation
                fig_full = create_combined_plot(filter_data, x, y, kernel_size, filter_name, cmap, vmin=filter_vmin, vmax=filter_vmax)
            placeholders[key].pyplot(fig_full)

            if not show_full_processed:
                zoomed_data = filter_data[ky:min(filter_data.shape[0], ky + zoom_size),
                                          kx:min(filter_data.shape[1], kx + zoom_size)]
                fig_zoom = create_combined_plot(zoomed_data, zoom_size // 2, zoom_size // 2, zoom_size, 
                                                f"Zoomed-In {filter_name}", cmap, zoom=True, vmin=filter_vmin, vmax=filter_vmax)
                
                zoomed_key = f'zoomed_{key}'
                if zoomed_key in placeholders and placeholders[zoomed_key] is not None:
                    placeholders[zoomed_key].pyplot(fig_zoom)

    plt.close('all')

# ---------------------------- Main Analysis Handler ---------------------------- #

def handle_image_analysis(
    tab: Any,
    image_np: np.ndarray,
    kernel_size: int,
    max_pixels: int,
    cmap: str,
    technique: str = "speckle",
    search_window_size: Optional[int] = None,
    filter_strength: float = 0.1,
    placeholders: Optional[Dict[str, Any]] = None,
    show_full_processed: bool = False,
    height: int = None,
    width: int = None
) -> Tuple[np.ndarray, ...]:
    
    with tab:
        placeholders = placeholders or create_placeholders_and_sections(technique, tab, show_full_processed)
        
        try:
            height = height or image_np.shape[0]
            width = width or image_np.shape[1]
            
            # Calculate the number of valid pixels
            valid_height = height - kernel_size + 1
            valid_width = width - kernel_size + 1
            total_valid_pixels = valid_height * valid_width
            
            # Use the minimum of max_pixels and total_valid_pixels
            pixels_to_process = min(max_pixels, total_valid_pixels)
            
            results = process_image(technique, image_np, kernel_size, pixels_to_process, height, width, 
                                    search_window_size, filter_strength)
            
            # Calculate the coordinates of the last processed pixel
            last_pixel = pixels_to_process - 1
            last_y = (last_pixel // valid_width) + kernel_size // 2
            last_x = (last_pixel % valid_width) + kernel_size // 2
            
            original_value, kernel_matrix = get_kernel_values(image_np, last_x, last_y, kernel_size)
            
            specific_params = get_technique_specific_params(technique, results, filter_strength, search_window_size, pixels_to_process, image_np, last_x, last_y)
            
            # Remove 'original_value' from specific_params if it exists
            specific_params.pop('original_value', None)
            
            display_formula(placeholders['formula'], technique, FORMULA_CONFIG,
                            x=last_x, y=last_y, 
                            input_x=last_x, input_y=last_y,
                            kernel_size=kernel_size,
                            kernel_matrix=kernel_matrix,
                            original_value=original_value,
                            **specific_params)
            
            process_and_visualize_image(image_np, kernel_size, last_x, last_y, results, cmap, technique, 
                                        placeholders, search_window_size, show_full_processed)
        
        except (ValueError, IndexError) as e:
            st.error(f"Error during image analysis: {str(e)}")
            return None
    
    return results

def get_kernel_values(image_np: np.ndarray, last_x: float, last_y: float, kernel_size: int) -> Tuple[float, List[List[float]]]:
    half_kernel = kernel_size // 2
    last_x, last_y = int(last_x), int(last_y)
    height, width = image_np.shape

    # Calculate valid ranges for kernel extraction
    y_start = max(0, last_y - half_kernel)
    y_end = min(height, last_y + half_kernel + 1)
    x_start = max(0, last_x - half_kernel)
    x_end = min(width, last_x + half_kernel + 1)

    try:
        kernel_values = image_np[y_start:y_end, x_start:x_end]
        
        if kernel_values.size == 0:
            raise ValueError(f"Extracted kernel at ({last_x}, {last_y}) is empty. Image shape: {image_np.shape}, Kernel size: {kernel_size}")

        # Pad the kernel if it's smaller than expected
        if kernel_values.shape != (kernel_size, kernel_size):
            pad_top = max(0, half_kernel - last_y)
            pad_bottom = max(0, last_y + half_kernel + 1 - height)
            pad_left = max(0, half_kernel - last_x)
            pad_right = max(0, last_x + half_kernel + 1 - width)
            kernel_values = np.pad(kernel_values, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='edge')

        kernel_matrix = [[float(kernel_values[i, j]) for j in range(kernel_size)] for i in range(kernel_size)]
        original_value = float(image_np[last_y, last_x])
        
        return original_value, kernel_matrix
    except IndexError as e:
        raise IndexError(f"Index error while extracting kernel. last_x: {last_x}, last_y: {last_y}, kernel_size: {kernel_size}, image shape: {image_np.shape}") from e

# ---------------------------- Formula Display ---------------------------- #


def get_technique_specific_params(technique: str, results: Tuple[np.ndarray, ...], filter_strength: float, search_window_size: Optional[int], pixels_to_process: int, image_np: np.ndarray, x: int, y: int) -> Dict[str, Any]:
    if technique == "speckle":
        return {
            "std": results[5],
            "mean": results[6],
            "sc": results[7],
            "total_pixels": pixels_to_process,
            "original_value": image_np[y, x]
        }
    elif technique == "nlm":
        denoised_image = results[0]
        return {
            "filter_strength": filter_strength,
            "search_size": search_window_size,
            "total_pixels": pixels_to_process,
            "original_value": image_np[y, x],
            "nlm_value": denoised_image[y, x]
        }
    else:
        return {}
    
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
                           r"\quad\quad\text{{Centered at pixel: }}({x}, {y})"
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
        "main_formula": r"I_{{{x},{y}}} = {original_value:.3f} \quad \rightarrow \quad NLM_{{{x},{y}}} = \frac{{1}}{{W(x,y)}} \sum_{{(i,j) \in \Omega}} I_{{i,j}} \cdot w((x,y), (i,j)) = {nlm_value:.3f}",
        "explanation": """
        This formula shows the transition from the original pixel intensity I({x},{y}) to the denoised value NLM({x},{y}) using the Non-Local Means (NLM) algorithm:
        
        1. For each pixel (x,y), we look at a small patch around it.
        2. We search for similar patches in a larger window Ω (size: {search_size}x{search_size}).
        3. We calculate a weighted average of pixels with similar surrounding patches.
        4. Pixels with more similar patches get higher weights in the average.
        
        Key components:
        • I_{{{x},{y}}}: Original intensity at pixel ({x},{y})
        • NLM_{{{x},{y}}}: Denoised value at pixel ({x},{y})
        • W(x,y): Normalization factor (sum of all weights)
        • w((x,y), (i,j)): Weight between pixels (x,y) and (i,j)
        • Ω: Search window (size: {search_size}x{search_size})
        
        Filter strength (h): {filter_strength:.3f}
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
                "formula": r"w((x,y), (i,j)) = \exp\left(-\frac{{||P(x,y) - P(i,j)||^2}}{{h^2}}\right)",
                "explanation": """
                This formula determines how much each pixel (i,j) contributes to denoising pixel (x,y):
                
                - P(x,y) and P(i,j) are patches centered at (x,y) and (i,j)
                - ||P(x,y) - P(i,j)||^2 measures how different the patches are
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


def display_formula(formula_placeholder: Any, technique: str, formula_config: Dict[str, Any], kernel_matrix: Optional[List[List[float]]] = None, **kwargs):
    with formula_placeholder.container():
        if technique not in formula_config:
            st.error(f"Unknown technique: {technique}")
            return

        config = formula_config[technique]
        
        # Update variables with provided kwargs
        config['variables'].update(kwargs)
        
        # Calculate original (input) image coordinates
        original_x = kwargs['x'] - kwargs['kernel_size'] // 2
        original_y = kwargs['y'] - kwargs['kernel_size'] // 2
        
        # Add input and output coordinates to variables
        config['variables']['input_x'] = original_x
        config['variables']['input_y'] = original_y
        config['variables']['output_x'] = kwargs['x']
        config['variables']['output_y'] = kwargs['y']

        
        
        # Generate kernel matrix formula if kernel_size and kernel_matrix are provided
        if 'kernel_size' in kwargs and kernel_matrix is not None:
            center = kwargs['kernel_size'] // 2
            center_value = kernel_matrix[center][center]
            kernel_matrix_str = generate_kernel_matrix_formula(kwargs['kernel_size'], center_value, kernel_matrix)
            config['variables']['kernel_matrix'] = kernel_matrix_str

        # Add search window description for NLM
        if technique == "nlm":
            config['variables']['search_window_description'] = get_search_window_description(kwargs.get('search_size', 'full'))
        
        # Display the main formula
        try:
            st.latex(config['main_formula'].format(**config['variables']))
        except KeyError as e:
            st.error(f"Missing key in main formula: {e}")
        
        # Display explanation if available
        if 'explanation' in config:
            try:
                st.markdown(config['explanation'].format(**config['variables']))
            except KeyError as e:
                st.error(f"Missing key in explanation: {e}")
        
        # Main expander to hold all additional formulas
        with st.expander("Additional Formulas", expanded=False):
            # Display additional formulas and explanations
            for additional_formula in config.get('additional_formulas', []):
                with st.expander(additional_formula['title'], expanded=False):
                    try:
                        st.latex(additional_formula['formula'].format(**config['variables']))
                        st.markdown(additional_formula['explanation'].format(**config['variables']))
                    except KeyError as e:
                        st.error(f"Missing key in additional formula: {e}")

def get_search_window_description(search_size):
    return "We search the entire image for similar pixels." if search_size == "full" else f"A search window of size {search_size}x{search_size} centered around the target pixel."

def generate_kernel_matrix_formula(kernel_size, center_value, kernel_matrix):
    matrix_rows = []
    for i in range(kernel_size):
        row = []
        for j in range(kernel_size):
            if i == kernel_size // 2 and j == kernel_size // 2:
                row.append(r"\mathbf{{{:.3f}}}".format(center_value))
            else:
                row.append(r"{:.3f}".format(kernel_matrix[i][j]))
        matrix_rows.append(" & ".join(row))

    return (r"\def\arraystretch{1.5}\begin{array}{|" + ":".join(["c"] * kernel_size) + "|}" +
            r"\hline" + r"\\ \hdashline".join(matrix_rows) + r"\\ \hline\end{array}")
