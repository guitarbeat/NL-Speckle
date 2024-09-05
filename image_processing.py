import numpy as np
import time
import streamlit as st
from helpers import create_plot, display_kernel_view
from typing import Tuple, Dict, Optional
import io
from PIL import Image
import matplotlib.pyplot as plt

# ---------------------------- Core Calculations ---------------------------- #

@st.cache_resource
def calculate_speckle(image: np.ndarray, kernel_size: int, stride: int, max_pixels: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, int, float, float, float]:
    pad_size = kernel_size // 2
    output_height = (image.shape[0] - kernel_size + 1)
    output_width = (image.shape[1] - kernel_size + 1)
    total_pixels = min(max_pixels, output_height * output_width)

    mean_filter = np.zeros((image.shape[0], image.shape[1]))
    std_dev_filter = np.zeros((image.shape[0], image.shape[1]))
    sc_filter = np.zeros((image.shape[0], image.shape[1]))

    for pixel in range(total_pixels):
        row, col = divmod(pixel, output_width)
        center_y, center_x = row + pad_size, col + pad_size

        local_window = image[row:row+kernel_size, col:col+kernel_size]
        
        # Calculate local statistics
        local_mean = np.mean(local_window)
        local_std = np.std(local_window)
        speckle_contrast = local_std / local_mean if local_mean != 0 else 0

        mean_filter[center_y, center_x] = local_mean
        std_dev_filter[center_y, center_x] = local_std
        sc_filter[center_y, center_x] = speckle_contrast

    last_x, last_y = col + pad_size, row + pad_size  # Center of the last processed kernel
    return mean_filter, std_dev_filter, sc_filter, last_x, last_y, local_mean, local_std, speckle_contrast

@st.cache_resource
def calculate_nlm(image: np.ndarray, kernel_size: int, search_size: Optional[int], filter_strength: float, stride: int, max_pixels: int) -> Tuple[np.ndarray, np.ndarray, int, int, float]:
    pad_size = kernel_size // 2
    
    # Handle the "full image" search window case
    if search_size is None or search_size == "full":
        search_size = max(image.shape)
    
    search_pad = search_size // 2
    output_height = (image.shape[0] - kernel_size + 1)
    output_width = (image.shape[1] - kernel_size + 1)
    total_pixels = min(max_pixels, output_height * output_width)

    denoised_image = np.zeros_like(image, dtype=float)
    weight_sum_map = np.zeros_like(image, dtype=float)

    padded_image = np.pad(image, search_pad, mode='reflect')

    for pixel in range(total_pixels):
        row, col = divmod(pixel, output_width)
        center_y, center_x = row + search_pad, col + search_pad

        search_area = padded_image[center_y-search_pad:center_y+search_pad+1, center_x-search_pad:center_x+search_pad+1]
        _, denoised_value, weight_sum = calculate_nlm_pixel(search_area, kernel_size, filter_strength)

        denoised_image[row+pad_size, col+pad_size] = denoised_value / weight_sum if weight_sum > 0 else search_area[search_pad, search_pad]
        weight_sum_map[row+pad_size, col+pad_size] = weight_sum

    last_x, last_y = col + pad_size, row + pad_size  # Center of the last processed kernel
    last_weight_sum = weight_sum_map[last_y, last_x]

    # Normalize the weight sum map
    max_weight = np.max(weight_sum_map)
    normalized_weight_map = weight_sum_map / max_weight if max_weight > 0 else weight_sum_map

    return denoised_image, normalized_weight_map, last_x, last_y, last_weight_sum

def calculate_nlm_pixel(search_area: np.ndarray, kernel_size: int, filter_strength: float) -> Tuple[np.ndarray, float, float]:
    center = search_area.shape[0] // 2
    center_patch = search_area[center:center+kernel_size, center:center+kernel_size]

    weights = np.zeros_like(search_area)
    denoised_value = 0.0
    weight_sum = 0.0

    for i in range(search_area.shape[0] - kernel_size + 1):
        for j in range(search_area.shape[1] - kernel_size + 1):
            patch = search_area[i:i+kernel_size, j:j+kernel_size]
            
            # Calculate patch difference
            patch_diff = center_patch - patch
            
            # Calculate Euclidean distance
            distance = np.sum(patch_diff ** 2)
            
            # Calculate weight using the NLM formula
            weight = np.exp(-distance / (filter_strength ** 2))
            
            # Update weight map
            weights[i+kernel_size//2, j+kernel_size//2] = weight
            
            # Update denoised value and weight sum
            denoised_value += search_area[i+kernel_size//2, j+kernel_size//2] * weight
            weight_sum += weight

    return weights, denoised_value, weight_sum

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
    if technique == "nlm":
        placeholders.update({
            'denoised_image': None,
            'zoomed_denoised_image': None,
            'weight_map': None,
            'zoomed_weight_map': None,
            'difference_map': None,
            'zoomed_difference_map': None,
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
    
    if technique == "nlm":
        filter_options = st.multiselect(
            "Select views to display",
            ["Original Image", "Denoised Image", "Weight Map", "Difference Map"],
            default=["Original Image", "Weight Map", "Denoised Image", "Difference Map"]
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
                    # st.expander("Weight Map Explanation", expanded=False)(
                    #     "The weight map shows the contribution of each pixel in the search window to the denoised value of the "
                    #     "current pixel. Brighter areas indicate higher weights, meaning those pixels had a stronger influence on "
                    #     "the denoising process. This map gives insight into which parts of the image the algorithm considers similar "
                    #     "to the current patch being processed. High weights typically correspond to areas with similar structures or patterns."
                    # )
                elif filter_name == "Difference Map":
                    placeholders['difference_map'], placeholders['zoomed_difference_map'] = create_section("Difference Map", expanded_main=True, expanded_zoomed=False)
        
        # Add a section for the NLM formula
        placeholders['formula'] = st.empty()

# ---------------------------- Analysis Loop and Visualization ---------------------------- #
@st.cache_data(ttl=60)  # Cache for 60 seconds
def run_analysis_loop(image_np: np.ndarray, kernel_size: int, stride: int, max_pixels: int, animation_speed: float, cmap: str, technique: str, placeholders: Dict[str, st.empty], search_window_size: Optional[int] = None, filter_strength: float = 0.1) -> Tuple[np.ndarray, ...]:
    
    for i in range(1, max_pixels + 1) if st.session_state.is_animating else [max_pixels]:
        st.session_state.max_pixels = max_pixels

        if technique == "speckle":
            results = calculate_speckle(image_np, kernel_size, stride, i)
            mean_filter, std_dev_filter, sc_filter, last_x, last_y, last_mean, last_std, last_sc = results
            display_speckle_contrast_formula(placeholders['formula'], last_x, last_y, last_std, last_mean, last_sc)
        elif technique == "nlm":
            results = calculate_nlm(image_np, kernel_size, search_window_size, filter_strength, stride, i)
            denoised_image, weight_sum_map, last_x, last_y, last_weight_sum = results
            display_nlm_formula(placeholders['formula'], last_x, last_y, kernel_size, search_window_size, filter_strength)
        # Add calculations for other techniques here
        
        update_visualizations(image_np, kernel_size, last_x, last_y, results, cmap, technique, placeholders, stride, search_window_size)
        
        if not st.session_state.is_animating:
            break
        
        time.sleep(animation_speed)
    
    return results

@st.cache_resource
def update_visualizations(image_np: np.ndarray, kernel_size: int, last_x: int, last_y: int, results: Tuple[np.ndarray, ...], cmap: str, technique: str, placeholders: Dict[str, st.empty], stride: int, search_window_size: Optional[int] = None):
    fig_original = create_plot(
        image_np, [], last_x, last_y, kernel_size,
        ["Original Image with Current Kernel"], cmap=cmap, 
        search_window=search_window_size, figsize=(5, 5)
    )
    placeholders['original_image'].pyplot(fig_original)
    plt.close(fig_original)  # Close the figure after plotting

    # Ensure kernel is centered and within image bounds
    kernel_start_y = max(0, last_y - kernel_size // 2)
    kernel_start_x = max(0, last_x - kernel_size // 2)
    kernel_end_y = min(image_np.shape[0], last_y + kernel_size // 2 + 1)
    kernel_end_x = min(image_np.shape[1], last_x + kernel_size // 2 + 1)
    
    zoomed_kernel = image_np[kernel_start_y:kernel_end_y, kernel_start_x:kernel_end_x]
    display_kernel_view(zoomed_kernel, image_np, "Zoomed-In Kernel", placeholders['zoomed_kernel'], cmap)

    if technique == "speckle":
        mean_filter, std_dev_filter, sc_filter = results[:3]
        filter_options = {
            "Mean Filter": mean_filter,
            "Std Dev Filter": std_dev_filter,
            "Speckle Contrast": sc_filter
        }
    if technique == "nlm":
        denoised_image, weight_sum_map = results[:2]
        difference_map = np.abs(image_np - denoised_image)
        filter_options = {
            "Denoised Image": denoised_image,
            "Weight Map": weight_sum_map,
            "Difference Map": difference_map
        }

    for filter_name, filter_data in filter_options.items():
        display_filter(filter_name, filter_data, last_x, last_y, cmap, placeholders, stride)

@st.cache_resource
def display_filter(filter_name: str, filter_data: np.ndarray, last_x: int, last_y: int, cmap: str, placeholders: Dict[str, st.empty], stride: int):
    key = filter_name.lower().replace(" ", "_")
    
    if key in placeholders and placeholders[key] is not None:
        fig_full, ax_full = plt.subplots()
        im = ax_full.imshow(filter_data, cmap=cmap)
        ax_full.set_title(filter_name)
        ax_full.axis('off')


       # Only add colorbar for Weight Map
        if filter_name == "Weight Map":
            cbar = plt.colorbar(im, ax=ax_full)
            cbar.set_label("Weight Value")
        
        placeholders[key].pyplot(fig_full)
        plt.close(fig_full)  # Close the figure after plotting
    
        zoom_size = 5
        zoomed_data = filter_data[max(0, last_y - zoom_size // 2):min(filter_data.shape[0], last_y + zoom_size // 2 + 1),
                                  max(0, last_x - zoom_size // 2):min(filter_data.shape[1], last_x + zoom_size // 2 + 1)]
        fig_zoom, ax_zoom = plt.subplots()
        im_zoom = ax_zoom.imshow(zoomed_data, cmap=cmap)
        ax_zoom.set_title(f"Zoomed-In {filter_name}")
        ax_zoom.axis('off')

        # Only add colorbar for Weight Map in zoomed view
        if filter_name == "Weight Map":
            plt.colorbar(im_zoom, ax=ax_zoom, label="Weight Value")
        
        for i, row in enumerate(zoomed_data):
            for j, val in enumerate(row):
                ax_zoom.text(j, i, f"{val:.2f}", ha="center", va="center", color="red", fontsize=8)
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


# Display the formula for Non-Local Means denoising in simpler terms
def display_nlm_formula(formula_placeholder, x, y, window_size, search_size, filter_strength):
    """Display the formula for Non-Local Means denoising for a specific pixel."""
    
    with formula_placeholder.container():
        with st.expander("Non-Local Means (NLM) Denoising Formula", expanded=False):
            # Simple explanation of the variables
            st.markdown(f"""
            ### Key Variables:
            - **Target Pixel**: Coordinates $(x_{{{x}}}, y_{{{y}}})$ are the pixel we want to clean up. It's the pixel we're focusing on.
            - **$I(i,j)$**: This is the original image value (or intensity) at any pixel $(i,j)$, where $(i,j)$ represents pixel coordinates in the image.
            - **Search Window ($\Omega$)**: {get_search_window_description(search_size)}. This is the area we search around the target pixel to find similar pixels.
            - **Neighborhood ($N(x,y)$)**: A small area ({window_size}x{window_size}) around each pixel $(x,y)$. Think of this as a little box of pixels around each pixel, used to calculate its average color.
            - **Smoothing Strength ($h$)**: This is a parameter that controls how much smoothing is done. A higher value means stronger smoothing.
            """)

            st.markdown("### NLM Formula Breakdown:")

            # Explain the main NLM formula
            st.latex(r'''
            \text{NLM}(x_{%d}, y_{%d}) = \frac{1}{W(x_{%d}, y_{%d})} \sum_{(i,j) \in \Omega} I(i,j) \cdot w((x_{%d}, y_{%d}), (i,j))
            ''' % (x, y, x, y, x, y))
            st.markdown("""
            **What does this formula mean?**
            
            - **NLM**: This is the denoised value for the pixel at $(x_{%d}, y_{%d})$ (the target pixel).
            - **$W(x_{%d}, y_{%d})$**: This is a normalization factor that ensures all the weights (explained next) add up to 1.
            - **$\sum_{(i,j) \in \Omega}$**: This symbol means we are adding up values for all pixels $(i,j)$ inside the search window $\Omega$.
            - **$I(i,j)$**: This is the original image value (intensity) at pixel $(i,j)$.
            - **$w((x_{%d}, y_{%d}), (i,j))$**: This is the weight we assign to pixel $(i,j)$, which depends on how similar its neighborhood is to the target pixel's neighborhood.
            
            In simple terms, this formula calculates the new value for the target pixel by averaging the values of all pixels in the search window. Pixels that are more similar to the target pixel's neighborhood are given more importance (higher weight).
            """ % (x, y, x, y, x, y))

            # Explain the normalization factor formula
            st.latex(r'''
            W(x_{%d}, y_{%d}) = \sum_{(i,j) \in \Omega} w((x_{%d}, y_{%d}), (i,j))
            ''' % (x, y, x, y))
            st.markdown("""
            **What does this formula mean?**
            
            - **$W(x_{%d}, y_{%d})$**: This is the sum of all the weights for the pixels in the search window $\Omega$. 
            - **$\sum_{(i,j) \in \Omega}$**: This symbol means we add up the weights for all pixels $(i,j)$ in the search window.
            - **$w((x_{%d}, y_{%d}), (i,j))$**: These are the weights that tell us how much each pixel contributes to the final value of the target pixel.
            
            This ensures that the contributions (or weights) of all pixels in the search window add up to 1, so the final value is correctly balanced.
            """ % (x, y, x, y))

            # Explain the weight calculation formula
            st.latex(r'''
            w((x_{%d}, y_{%d}), (i,j)) = \exp\left(-\frac{|P(i,j) - P(x_{%d}, y_{%d})|^2}{h^2}\right)
            ''' % (x, y, x, y))
            st.markdown("""
            **What does this formula mean?**
            
            - **$w((x_{%d}, y_{%d}), (i,j))$**: This is the weight given to the pixel $(i,j)$ when deciding the value of the target pixel.
            - **$\exp()$**: This stands for "exponential function," and it's used here to ensure the weight decreases quickly as the difference between neighborhoods increases.
            - **$|P(i,j) - P(x_{%d}, y_{%d})|^2$**: This is the squared difference between the average color (or intensity) of the neighborhood around pixel $(i,j)$ and the neighborhood around the target pixel $(x_{%d}, y_{%d})$.
            - **$h$**: This is the smoothing strength. A smaller $h$ makes the formula more sensitive to differences, while a larger $h$ makes it less sensitive.

            This formula says that the more similar two neighborhoods are, the higher the weight given to that pixel. Similar neighborhoods have a smaller difference, so they get a bigger weight.
            """ % (x, y, x, y, x, y))

            # Explain the neighborhood average formula
            st.latex(r'''
            P(x_{%d}, y_{%d}) = \frac{1}{|N(x_{%d}, y_{%d})|} \sum_{(k,l) \in N(x_{%d}, y_{%d})} I(k,l)
            ''' % (x, y, x, y, x, y))
            st.markdown("""
            **What does this formula mean?**
            
            - **$P(x_{%d}, y_{%d})$**: This is the average color (or intensity) of the neighborhood around the pixel $(x_{%d}, y_{%d})$.
            - **$\sum_{(k,l) \in N(x_{%d}, y_{%d})}$**: This means we are adding up the values of all pixels $(k,l)$ in the neighborhood $N(x_{%d}, y_{%d})$.
            - **$I(k,l)$**: This is the intensity (or color value) of pixel $(k,l)$.
            - **$|N(x_{%d}, y_{%d})|$**: This is the number of pixels in the neighborhood (for example, if the neighborhood is 3x3, there are 9 pixels).

            This formula calculates the average intensity of the pixels in the neighborhood around the target pixel. It's used to compare how similar different areas of the image are.
            """ % (x, y, x, y, x, y, x, y, x, y))

            st.markdown("""
            ### Additional Notes:
            - The **search window** ($\Omega$) is the area where we look for similar pixels to help denoise the target pixel.
            - The **neighborhood size** affects how we measure the similarity between pixels.
            - The **smoothing strength ($h$)** controls how much noise is removed. Larger $h$ means more aggressive denoising.
            """)

# Helper function to describe the search window in simpler terms
def get_search_window_description(search_size):
    if search_size == "full":
        return "We search the entire image for similar pixels."
    else:
        return f"A search window of size {search_size}x{search_size} centered around the target pixel."

# ---------------------------- Main Entry Point ---------------------------- #


def handle_image_analysis(tab: st.tabs, image_np: np.ndarray, kernel_size: int, stride: int, max_pixels: int, animation_speed: float, cmap: str, technique: str = "speckle", search_window_size: Optional[int] = None, filter_strength: float = 0.1) -> Tuple[np.ndarray, ...]:
    with tab:
        st.header(f"{technique.capitalize()} Analysis", divider="rainbow")
        
        # Create placeholders
        placeholders = create_placeholders(technique)
        
        # Create sections
        create_sections(placeholders, technique)
        
        # Calculation and visualization loop
        results = run_analysis_loop(image_np, kernel_size, stride, max_pixels, animation_speed, cmap, technique, placeholders, search_window_size, filter_strength)
        
        # Save Results Section
        create_save_section(results, technique)

        return results
