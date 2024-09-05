import numpy as np
import time
import streamlit as st
from helpers import create_plot, display_kernel_view
from typing import Tuple, Dict
import io
from PIL import Image
import matplotlib.pyplot as plt

# ---------------------------- Core Calculations ---------------------------- #

@st.cache_data
def calculate_speckle(image: np.ndarray, kernel_size: int, stride: int, max_pixels: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, int, float, float, float]:
    output_height = (image.shape[0] - kernel_size) // stride + 1
    output_width = (image.shape[1] - kernel_size) // stride + 1
    total_pixels = min(max_pixels, output_height * output_width)

    mean_filter = np.zeros((output_height, output_width))
    std_dev_filter = np.zeros((output_height, output_width))
    sc_filter = np.zeros((output_height, output_width))

    for pixel in range(total_pixels):
        row, col = divmod(pixel, output_width)
        top_left_y, top_left_x = row * stride, col * stride

        local_window = image[top_left_y:top_left_y + kernel_size, top_left_x:top_left_x + kernel_size]
        
        # Calculate local statistics
        local_mean = np.mean(local_window)
        local_std = np.std(local_window)
        speckle_contrast = local_std / local_mean if local_mean != 0 else 0

        mean_filter[row, col] = local_mean
        std_dev_filter[row, col] = local_std
        sc_filter[row, col] = speckle_contrast

    last_x, last_y = top_left_x, top_left_y
    return mean_filter, std_dev_filter, sc_filter, last_x, last_y, local_mean, local_std, speckle_contrast

@st.cache_data
def calculate_nlm(image: np.ndarray, kernel_size: int, search_size: int, filter_strength: float, stride: int, max_pixels: int) -> Tuple[np.ndarray, np.ndarray, int, int, float]:
    output_height = (image.shape[0] - kernel_size) // stride + 1
    output_width = (image.shape[1] - kernel_size) // stride + 1
    total_pixels = min(max_pixels, output_height * output_width)

    denoised_image = np.zeros_like(image, dtype=float)
    weight_map = np.zeros_like(image, dtype=float)

    pad_size = search_size // 2
    padded_image = np.pad(image, pad_size, mode='reflect')

    for pixel in range(total_pixels):
        row, col = divmod(pixel, output_width)
        center_y, center_x = row * stride + kernel_size // 2, col * stride + kernel_size // 2

        search_area = padded_image[center_y:center_y+search_size, center_x:center_x+search_size]
        weights = calculate_weights(search_area, kernel_size, filter_strength)

        denoised_image[center_y, center_x] = np.sum(search_area * weights) / np.sum(weights)
        weight_map[center_y, center_x] = np.max(weights)

    last_x, last_y = center_x, center_y
    last_weight = weight_map[last_y, last_x]
    return denoised_image, weight_map, last_x, last_y, last_weight

def calculate_weights(search_area: np.ndarray, kernel_size: int, filter_strength: float) -> np.ndarray:
    center = search_area.shape[0] // 2
    center_patch = search_area[center:center+kernel_size, center:center+kernel_size]

    weights = np.zeros_like(search_area)
    for i in range(search_area.shape[0] - kernel_size + 1):
        for j in range(search_area.shape[1] - kernel_size + 1):
            patch = search_area[i:i+kernel_size, j:j+kernel_size]
            distance = np.sum((center_patch - patch) ** 2)
            weights[i+kernel_size//2, j+kernel_size//2] = np.exp(-distance / (filter_strength ** 2))

    return weights

# ---------------------------- Placeholder and Section Creation ---------------------------- #

def create_placeholders(technique: str) -> Dict[str, st.empty]:
    placeholders = {
        'formula': st.empty(),
        'original_image': None,
        'zoomed_kernel': None,
    }
    if technique == "speckle":
        placeholders.update({
            'mean_filter': None,
            'zoomed_mean_filter': None,
            'standard_deviation_filter': None,
            'zoomed_standard_deviation_filter': None,
            'speckle_contrast': None,
            'zoomed_speckle_contrast': None
        })
    elif technique == "nlm":
        placeholders.update({
            'denoised_image': None,
            'zoomed_denoised_image': None,
            'weight_map': None,
            'zoomed_weight_map': None,
        })
    return placeholders

def create_sections(placeholders: Dict[str, st.empty], technique: str):
    def create_section(title: str, expanded_main: bool = False, expanded_zoomed: bool = False):
        with st.expander(title, expanded=expanded_main):
            main_placeholder = st.empty()
            with st.expander(f"Zoomed-in {title.split()[0]}", expanded=expanded_zoomed):
                zoomed_placeholder = st.empty()
        return main_placeholder, zoomed_placeholder
    
    if technique == "speckle":
        filter_options = st.multiselect(
            "Select filters to display",
            ["Mean Filter", "Std Dev Filter", "Speckle Contrast"],
            default=["Mean Filter", "Std Dev Filter", "Speckle Contrast"]
        )
        
        columns = st.columns(len(filter_options) + 1)  # +1 for the original image
        
        with columns[0]:
            placeholders['original_image'], placeholders['zoomed_kernel'] = create_section("Original Image with Current Kernel", expanded_main=True, expanded_zoomed=False)
        
        for i, filter_name in enumerate(filter_options, start=1):
            with columns[i]:
                key = filter_name.lower().replace(" ", "_")
                placeholders[key], placeholders[f'zoomed_{key}'] = create_section(filter_name, expanded_main=True, expanded_zoomed=False)
    
    elif technique == "nlm":
        filter_options = st.multiselect(
            "Select views to display",
            ["Original Image", "Denoised Image", "Weight Map"],
            default=["Original Image", "Denoised Image", "Weight Map"]
        )
        
        columns = st.columns(len(filter_options))
        
        for i, filter_name in enumerate(filter_options):
            with columns[i]:
                if filter_name == "Original Image":
                    placeholders['original_image'], placeholders['zoomed_kernel'] = create_section("Original Image with Current Kernel", expanded_main=True, expanded_zoomed=False)
                elif filter_name == "Denoised Image":
                    placeholders['denoised_image'], placeholders['zoomed_denoised_image'] = create_section("Denoised Image", expanded_main=True, expanded_zoomed=False)
                elif filter_name == "Weight Map":
                    placeholders['weight_map'], placeholders['zoomed_weight_map'] = create_section("Weight Map", expanded_main=True, expanded_zoomed=False)
        
        # Add a section for the NLM formula
        placeholders['formula'] = st.empty()

# ---------------------------- Analysis Loop and Visualization ---------------------------- #

def run_analysis_loop(image_np: np.ndarray, kernel_size: int, stride: int, max_pixels: int, animation_speed: float, cmap: str, technique: str, placeholders: Dict[str, st.empty]) -> Tuple[np.ndarray, ...]:
    
    for i in range(1, max_pixels + 1) if st.session_state.is_animating else [max_pixels]:
        st.session_state.max_pixels = max_pixels

        if technique == "speckle":
            results = calculate_speckle(image_np, kernel_size, stride, i)
            mean_filter, std_dev_filter, sc_filter, last_x, last_y, last_mean, last_std, last_sc = results
            display_speckle_contrast_formula(placeholders['formula'], last_x, last_y, last_std, last_mean, last_sc)
        # Add calculations for other techniques here
        
        update_visualizations(image_np, kernel_size, last_x, last_y, results, cmap, technique, placeholders, stride)
          
      
        if not st.session_state.is_animating:
            break
        
        time.sleep(animation_speed)
    
    return results

def update_visualizations(image_np: np.ndarray, kernel_size: int, last_x: int, last_y: int, results: Tuple[np.ndarray, ...], cmap: str, technique: str, placeholders: Dict[str, st.empty], stride: int):
    fig_original = create_plot(
        image_np, [], last_x, last_y, kernel_size,
        ["Original Image with Current Kernel"], cmap=cmap, 
        search_window=None, figsize=(5, 5)
    )
    placeholders['original_image'].pyplot(fig_original)
    plt.close(fig_original)  # Close the figure after plotting

    zoomed_kernel = image_np[last_y : last_y + kernel_size, last_x : last_x + kernel_size]
    display_kernel_view(zoomed_kernel, image_np, "Zoomed-In Kernel", placeholders['zoomed_kernel'], cmap)

    if technique == "speckle":
        mean_filter, std_dev_filter, sc_filter = results[:3]
        filter_options = {
            "Mean Filter": mean_filter,
            "Std Dev Filter": std_dev_filter,
            "Speckle Contrast": sc_filter
        }
        for filter_name, filter_data in filter_options.items():
            display_filter(filter_name, filter_data, last_x, last_y, cmap, placeholders, stride)
    # Add visualizations for other techniques here

def display_filter(filter_name: str, filter_data: np.ndarray, last_x: int, last_y: int, cmap: str, placeholders: Dict[str, st.empty], stride: int):
    key = filter_name.lower().replace(" ", "_")
    
    if key in placeholders and placeholders[key] is not None:
        fig_full, ax_full = plt.subplots()
        ax_full.imshow(filter_data, cmap=cmap)
        ax_full.set_title(filter_name)
        ax_full.axis('off')
        placeholders[key].pyplot(fig_full)
        plt.close(fig_full)  # Close the figure after plotting
    
        zoom_size = 1
        zoomed_data = filter_data[last_y // stride : last_y // stride + zoom_size, last_x // stride : last_x // stride + zoom_size]
        fig_zoom, ax_zoom = plt.subplots()
        ax_zoom.imshow(zoomed_data, cmap=cmap)
        ax_zoom.set_title(f"Zoomed-In {filter_name}")
        ax_zoom.axis('off')
        for i, row in enumerate(zoomed_data):
            for j, val in enumerate(row):
                ax_zoom.text(j, i, f"{val:.2f}", ha="center", va="center", color="red", fontsize=10)
        fig_zoom.tight_layout(pad=2)
        
        zoomed_key = f'zoomed_{key}'
        if zoomed_key in placeholders and placeholders[zoomed_key] is not None:
            placeholders[zoomed_key].pyplot(fig_zoom)
        plt.close(fig_zoom)  # Close the zoomed figure after plotting
    else:
        print(f"Placeholder for {filter_name} not found or is None. Skipping visualization.")

# ---------------------------- Save Results ---------------------------- #

def create_save_section(results: Tuple[np.ndarray, ...], technique: str):
    with st.expander("Save Results"):
        if technique == "speckle":
            mean_filter, std_dev_filter, sc_filter, *_ = results
            if std_dev_filter is not None and sc_filter is not None:
                filter_options = {
                    "std_dev_filter": (std_dev_filter, "std_dev_filter.png", "Download Std Dev Filter"),
                    "speckle_contrast": (sc_filter, "speckle_contrast.png", "Download Speckle Contrast Image"),
                    "mean_filter": (mean_filter, "mean_filter.png", "Download Mean Filter")
                }
                for filter_data, filename, button_text in filter_options.values():
                    create_download_button(filter_data, filename, button_text)
            else:
                st.error("No results to save. Please generate images by running the analysis.")
        # Add save options for other techniques here

def create_download_button(image: np.ndarray, filename: str, button_text: str):
    img_buffer = io.BytesIO()
    Image.fromarray((255 * image).astype(np.uint8)).save(img_buffer, format='PNG')
    img_buffer.seek(0)
    st.download_button(label=button_text, data=img_buffer, file_name=filename, mime="image/png")

# ---------------------------- Information Display ---------------------------- #

def display_speckle_contrast_formula(formula_placeholder: st.empty, x: int, y: int, std: float, mean: float, sc: float):
    """Display the speckle contrast formula."""
    with formula_placeholder.container():
        st.latex(f'SC_{{{x}, {y}}} = \\frac{{\\sigma}}{{\\mu}} = \\frac{{{std:.3f}}}{{{mean:.3f}}} = {sc:.3f}')


# Display the formula for Non-Local Means denoising for a specific pixel.
def display_nlm_formula(formula_placeholder, x, y, window_size, search_size, filter_strength):
    """Display the formula for Non-Local Means denoising for a specific pixel."""
    
    with formula_placeholder.container():
        with st.expander("Non-Local Means (NLM) Denoising Formula", expanded=False):
            st.markdown(f"""
            Let's define our variables first:
            - $(x_{{{x}}}, y_{{{y}}})$: Coordinates of the target pixel we're denoising
            - $I(i,j)$: Original image value at pixel $(i,j)$
            - $\Omega$: {get_search_window_description(search_size)}
            - $N(x,y)$: Neighborhood of size {window_size}x{window_size} around pixel $(x,y)$
            - $h$: Filtering parameter (controls smoothing strength), set to {filter_strength:.2f}
            """)

            st.markdown("Now, let's break down the NLM formula:")

            st.latex(r'''
            \text{NLM}(x_{%d}, y_{%d}) = \frac{1}{W(x_{%d}, y_{%d})} \sum_{(i,j) \in \Omega} I(i,j) \cdot w((x_{%d}, y_{%d}), (i,j))
            ''' % (x, y, x, y, x, y))
            st.markdown("This is the main NLM formula. It calculates the denoised value as a weighted average of pixels in the search window.")

            st.latex(r'''
            W(x_{%d}, y_{%d}) = \sum_{(i,j) \in \Omega} w((x_{%d}, y_{%d}), (i,j))
            ''' % (x, y, x, y))
            st.markdown("$W(x,y)$ is a normalization factor, ensuring weights sum to 1.")

            st.latex(r'''
            w((x_{%d}, y_{%d}), (i,j)) = \exp\left(-\frac{|P(i,j) - P(x_{%d}, y_{%d})|^2}{h^2}\right)
            ''' % (x, y, x, y))
            st.markdown("This calculates the weight based on similarity between neighborhoods. More similar neighborhoods get higher weights.")

            st.latex(r'''
            P(x_{%d}, y_{%d}) = \frac{1}{|N(x_{%d}, y_{%d})|} \sum_{(k,l) \in N(x_{%d}, y_{%d})} I(k,l)
            ''' % (x, y, x, y, x, y))
            st.markdown("$P(x,y)$ is the average value of the neighborhood around pixel $(x,y)$. This is used in weight calculation.")

            st.markdown("""
            Additional notes:
            - The search window $\Omega$ determines which pixels are considered for denoising.
            - The neighborhood size affects how similarity is calculated between different parts of the image.
            - The filtering parameter $h$ controls the strength of denoising. Higher values lead to more smoothing.
            """)

# Helper function to get a description of the search window
def get_search_window_description(search_size):
    if search_size is None:
        return "Search window covering the entire image"
    else:
        return f"Search window of size {search_size}x{search_size} centered at $(x, y)$"

# ---------------------------- Main Entry Point ---------------------------- #


def handle_image_analysis(tab: st.tabs, image_np: np.ndarray, kernel_size: int, stride: int, max_pixels: int, animation_speed: float, cmap: str, technique: str = "speckle") -> Tuple[np.ndarray, ...]:
    with tab:
        st.header(f"{technique.capitalize()} Analysis", divider="rainbow")
        
        # Create placeholders
        placeholders = create_placeholders(technique)
        
        # Create sections
        create_sections(placeholders, technique)
        
        # Calculation and visualization loop
        results = run_analysis_loop(image_np, kernel_size, stride, max_pixels, animation_speed, cmap, technique, placeholders)
        
        # Save Results Section
        create_save_section(results, technique)

        return results
