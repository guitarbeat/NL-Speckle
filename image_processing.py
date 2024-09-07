import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from typing import Tuple, List, Optional, Union, Any, Dict, Callable
from streamlit_image_comparison import image_comparison
from numba import jit, prange
import logging
import time
from functools import wraps
import multiprocessing as mp

# ---------------------------- Logging and Timing ---------------------------- #

logging.basicConfig(format='%(message)s', level=logging.INFO)

def timing_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logging.info(f"{func.__name__} took {end_time - start_time:.2f} seconds")
        return result
    return wrapper

# ---------------------------- Image Analysis Algorithms ---------------------------- #

@timing_decorator
@st.cache_data(persist=True)
def calculate_speckle(image: np.ndarray, kernel_size: int, stride: int, max_pixels: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, int, float, float, float]:
    pad_size = kernel_size // 2
    output_height = (image.shape[0] - kernel_size + 1)
    output_width = (image.shape[1] - kernel_size + 1)
    total_pixels = min(max_pixels, output_height * output_width)

    # Use float32 for better memory efficiency
    mean_filter = np.zeros(image.shape, dtype=np.float32)
    std_dev_filter = np.zeros(image.shape, dtype=np.float32)
    sc_filter = np.zeros(image.shape, dtype=np.float32)

    # Use Numba to speed up the loop
    @jit(nopython=True, parallel=True)
    def process_pixels(image, mean_filter, std_dev_filter, sc_filter, kernel_size, output_width, total_pixels):
        for pixel in range(total_pixels):
            row, col = divmod(pixel, output_width)
            center_y, center_x = row + pad_size, col + pad_size

            local_window = image[row:row+kernel_size, col:col+kernel_size]
            
            local_mean = np.mean(local_window)
            local_std = np.std(local_window)
            speckle_contrast = local_std / local_mean if local_mean != 0 else 0

            mean_filter[center_y, center_x] = local_mean
            std_dev_filter[center_y, center_x] = local_std
            sc_filter[center_y, center_x] = speckle_contrast

        return mean_filter, std_dev_filter, sc_filter

    mean_filter, std_dev_filter, sc_filter = process_pixels(image, mean_filter, std_dev_filter, sc_filter, kernel_size, output_width, total_pixels)

    last_x, last_y = (total_pixels - 1) % output_width + pad_size, (total_pixels - 1) // output_width + pad_size
    return mean_filter, std_dev_filter, sc_filter, last_x, last_y, mean_filter[last_y, last_x], std_dev_filter[last_y, last_x], sc_filter[last_y, last_x]

@timing_decorator
@st.cache_data(persist=True)
def calculate_nlm(image: np.ndarray, kernel_size: int, search_size: Optional[int], filter_strength: float, stride: int, max_pixels: int) -> Tuple[np.ndarray, np.ndarray, int, int, float]:
    
    # Initialize output arrays for NLM(x, y) and W(x, y)
    denoised_image = np.zeros_like(image, dtype=np.float32)
    weight_sum_map = np.zeros_like(image, dtype=np.float32)

    height, width = image.shape
    total_pixels = min(max_pixels, height * width)

    @jit(nopython=True)
    def process_nlm(image, denoised_image, weight_sum_map, kernel_size, filter_strength, total_pixels, search_size):
        height, width = image.shape
        pad_size = kernel_size // 2
        for pixel in range(total_pixels):
            y, x = divmod(pixel, width)

            # Define P(x, y): the neighborhood around the current pixel
            y_start, y_end = max(0, y - pad_size), min(height, y + pad_size + 1)
            x_start, x_end = max(0, x - pad_size), min(width, x + pad_size + 1)
            center_patch = image[y_start:y_end, x_start:x_end]
            
            denoised_value = 0.0
            weight_sum = 0.0  # This represents W(x, y) in the formula

            # Define Ω: the search window
            if search_size is None:
                search_y_start, search_y_end = 0, height
                search_x_start, search_x_end = 0, width
            else:
                search_y_start = max(0, y - search_size // 2)
                search_y_end = min(height, y + search_size // 2 + 1)
                search_x_start = max(0, x - search_size // 2)
                search_x_end = min(width, x + search_size // 2 + 1)

            # Iterate over Ω
            for i in range(search_y_start, search_y_end):
                for j in range(search_x_start, search_x_end):
                    # Define P(i, j): the neighborhood around the comparison pixel
                    i_start, i_end = max(0, i - pad_size), min(height, i + pad_size + 1)
                    j_start, j_end = max(0, j - pad_size), min(width, j + pad_size + 1)
                    patch = image[i_start:i_end, j_start:j_end]
                    
                    # Ensure patches are the same size for comparison
                    min_height = min(center_patch.shape[0], patch.shape[0])
                    min_width = min(center_patch.shape[1], patch.shape[1])
                    center_patch_crop = center_patch[:min_height, :min_width]
                    patch_crop = patch[:min_height, :min_width]
                    
                    # Calculate ||P(x, y) - P(i, j)||^2
                    distance = np.sum((center_patch_crop - patch_crop)**2)
                    
                    # Calculate w((x, y), (i, j)) = exp(-||P(x, y) - P(i, j)||^2 / h^2)
                    weight = np.exp(-distance / (filter_strength ** 2))
                    
                    # Accumulate w((x, y), (i, j)) * I(i, j)
                    denoised_value += image[i, j] * weight
                    # Accumulate W(x, y)
                    weight_sum += weight
                    # Track total weight for each pixel (not in original formula, used for visualization)
                    weight_sum_map[i, j] += weight

            # Calculate NLM(x, y) = (1 / W(x, y)) * Σ w((x, y), (i, j)) * I(i, j)
            denoised_image[y, x] = denoised_value / weight_sum if weight_sum > 0 else image[y, x]

        return denoised_image, weight_sum_map

    denoised_image, weight_sum_map = process_nlm(image, denoised_image, weight_sum_map, kernel_size, filter_strength, total_pixels, search_size)

    last_x, last_y = (total_pixels - 1) % width, (total_pixels - 1) // width
    last_weight_sum = weight_sum_map[last_y, last_x]

    # Normalize weight_sum_map for visualization (not part of the original NLM formula)
    max_weight = np.max(weight_sum_map)
    normalized_weight_map = weight_sum_map / max_weight if max_weight > 0 else weight_sum_map

    return denoised_image, normalized_weight_map, last_x, last_y, last_weight_sum

# ---------------------------- UI Components and Layout ---------------------------- #

def create_placeholders_and_sections(technique: str, tab: st.delta_generator.DeltaGenerator, show_full_processed: bool) -> Dict[str, Any]:
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
    with st.expander(title, expanded=expanded_main):
        main_placeholder = st.empty()
        zoomed_placeholder = st.expander(f"Zoomed-in {title.split()[0]}", expanded=expanded_zoomed).empty()
    return main_placeholder, zoomed_placeholder

def handle_image_comparison(tab, cmap_name: str, images: Dict[str, np.ndarray]):
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
        else:
            st.info("Select two images to compare.")

# ---------------------------- Formula Display ---------------------------- #

# Define formulas and explanations for each technique
FORMULA_CONFIG = {
    "speckle": {
        "main_formula": r"I_{{{x},{y}}} = {original_value:.3f} \quad \rightarrow \quad SC_{{{x},{y}}} = \frac{{\sigma}}{{\mu}} = \frac{{{std:.3f}}}{{{mean:.3f}}} = {sc:.3f}",
        "variables": {},  # To be filled dynamically
        "explanation": "This formula shows the transition from the original pixel intensity I({x},{y}) to the Speckle Contrast (SC) for pixel ({x},{y}).",
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
        "main_formula": r"\text{{NLM}}(x_{{{x}}}, y_{{{y}}}) = \frac{{1}}{{W(x_{{{x}}}, y_{{{y}}})}} \sum_{{(i,j) \in \Omega}} I(i,j) \cdot w((x_{{{x}}}, y_{{{y}}}), (i,j))",
        "variables": {},  # To be filled dynamically
        "explanation": "This formula represents the Non-Local Means (NLM) denoising algorithm for pixel ({x}, {y}).",
        "additional_formulas": [
            {
                "title": "Normalization Factor",
                "formula": r"W(x_{{{x}}}, y_{{{y}}}) = \sum_{{(i,j) \in \Omega}} w((x_{{{x}}}, y_{{{y}}}), (i,j))",
                "explanation": "This formula calculates the normalization factor to ensure all weights sum to 1."
            },
            {
                "title": "Weight Calculation",
                "formula": r"w((x_{{{x}}}, y_{{{y}}}), (i,j)) = \exp\left(-\frac{{|P(i,j) - P(x_{{{x}}}, y_{{{y}}})|^2}}{{h^2}}\right)",
                "explanation": "This formula determines the weight of each pixel based on neighborhood similarity. h={filter_strength} is the smoothing strength."
            },
            {
                "title": "Neighborhood Average",
                "formula": r"P(x_{{{x}}}, y_{{{y}}}) = \frac{{1}}{{|N(x_{{{x}}}, y_{{{y}}})|}} \sum_{{(k,l) \in N(x_{{{x}}}, y_{{{y}}})}} I(k,l)",
                "explanation": "This formula calculates the average intensity of the {kernel_size}x{kernel_size} neighborhood around a pixel."
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
        
        # Generate kernel matrix formula if kernel_size and kernel_matrix are provided
        if 'kernel_size' in kwargs and kernel_matrix is not None:
            center = kwargs['kernel_size'] // 2
            center_value = kernel_matrix[center][center]
            kernel_matrix_str = generate_kernel_matrix_formula(kwargs['kernel_size'], center_value, kernel_matrix)
            config['variables']['kernel_matrix'] = kernel_matrix_str
        
        # Debug: Print out all variables
        # st.write("Debug: Variables received:", config['variables'])
        
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

    return (r"\def\arraystretch{1.5}\begin{array}{" + ":".join(["c"] * kernel_size) + "}" +
            r"\\" + r"\\".join(matrix_rows) + r"\end{array}")

# ---------------------------- Image Visualization ---------------------------- #

def create_combined_plot(plot_image: np.ndarray, plot_x: int, plot_y: int, plot_kernel_size: int, 
                         title: str, plot_cmap: str = "viridis", plot_search_window: Optional[Union[str, int]] = None, 
                         zoom: bool = False, vmin: Optional[float] = None, vmax: Optional[float] = None) -> plt.Figure:
    fig, ax = plt.subplots(1, 1)
    
    # Use the provided vmin and vmax for consistent color mapping
    im = ax.imshow(plot_image, cmap=plot_cmap, vmin=vmin, vmax=vmax)
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

def process_and_visualize_image(image: np.ndarray, kernel_size: int, x: int, y: int, 
                                results: Tuple[np.ndarray, ...], cmap: str, technique: str, 
                                placeholders: Dict[str, Any], stride: int, 
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
        fig_original = create_combined_plot(image, x, y, kernel_size, "Original Image with Current Kernel", cmap, 
                                            search_window_size if technique == "nlm" else None, vmin=vmin, vmax=vmax)
    
    placeholders['original_image'].pyplot(fig_original)

    if not show_full_processed:
        # Create zoomed view of original image
        zoom_size = kernel_size
        ky, kx = max(0, y - zoom_size // 2), max(0, x - zoom_size // 2)
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
            # Calculate vmin and vmax for this specific filter
            filter_vmin, filter_vmax = np.min(filter_data), np.max(filter_data)
            
            # Full image view
            if show_full_processed:
                fig_full = plt.figure()
                plt.imshow(filter_data, cmap=cmap, vmin=filter_vmin, vmax=filter_vmax)
                plt.axis('off')
                plt.title(filter_name)
            else:
                fig_full = create_combined_plot(filter_data, x, y, kernel_size, filter_name, cmap, vmin=filter_vmin, vmax=filter_vmax)
            placeholders[key].pyplot(fig_full)

            if not show_full_processed:
                # Zoomed view
                zoomed_data = filter_data[ky:min(filter_data.shape[0], ky + zoom_size),
                                          kx:min(filter_data.shape[1], kx + zoom_size)]
                fig_zoom = create_combined_plot(zoomed_data, zoom_size // 2, zoom_size // 2, zoom_size, 
                                                f"Zoomed-In {filter_name}", cmap, zoom=True, vmin=filter_vmin, vmax=filter_vmax)
                
                zoomed_key = f'zoomed_{key}'
                if zoomed_key in placeholders and placeholders[zoomed_key] is not None:
                    placeholders[zoomed_key].pyplot(fig_zoom)

    # Close all figures to avoid memory issues
    plt.close('all')

# ---------------------------- Main Analysis Handler ---------------------------- #

def handle_image_analysis(
    tab: Any,
    image_np: np.ndarray,
    kernel_size: int,
    stride: int,
    max_pixels: int,
    cmap: str,
    technique: str = "speckle",
    search_window_size: Optional[int] = None,
    filter_strength: float = 0.1,
    placeholders: Optional[Dict[str, Any]] = None,
    show_full_processed: bool = False
) -> Tuple[np.ndarray, ...]:
    
    with tab:
        if placeholders is None:
            placeholders = create_placeholders_and_sections(technique, tab, show_full_processed)

        if technique == "speckle":
            results = calculate_speckle(image_np, kernel_size, stride, max_pixels)
            last_x, last_y = results[3:5]
            original_value = image_np[last_y, last_x]  # Get the original pixel value
            
            # Extract kernel values
            half_kernel = kernel_size // 2
            kernel_values = image_np[last_y-half_kernel:last_y+half_kernel+1, 
                                     last_x-half_kernel:last_x+half_kernel+1]
            
            # Create a 2D list to store the kernel values
            kernel_matrix = [[kernel_values[i, j] for j in range(kernel_size)] for i in range(kernel_size)]

            display_formula(placeholders['formula'], technique, FORMULA_CONFIG, 
                            x=last_x, y=last_y, std=results[5], mean=results[6], 
                            sc=results[7], original_value=original_value,
                            kernel_size=kernel_size,
                            total_pixels=kernel_size**2,
                            kernel_matrix=kernel_matrix)

        
        elif technique == "nlm":
            results = calculate_nlm(image_np, kernel_size, search_window_size, filter_strength, stride, max_pixels)
            last_x, last_y = results[2:4]
            display_formula(placeholders['formula'], technique, FORMULA_CONFIG, x=last_x, y=last_y, kernel_size=kernel_size, search_size=search_window_size, filter_strength=filter_strength)

        process_and_visualize_image(image_np, kernel_size, last_x, last_y, results, cmap, technique, placeholders, stride, search_window_size, show_full_processed)

    return results