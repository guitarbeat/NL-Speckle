import numpy as np
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def clip_indices(value, min_value, max_value):
    """
    Ensure that the value is within the specified bounds.
    """
    return max(min_value, min(value, max_value))

def get_patch(image, x, y, patch_size):
    """
    Extract a patch from the image centered at (x, y).
    """
    half = patch_size // 2
    # TODO: Use clip_indices to ensure patch boundaries are within image
    # TODO: Extract and return the patch from the image
    pass

def pad_patch(patch, target_shape):
    """
    Pad a given patch to match the target dimensions.
    """
    # TODO: Use np.pad to add zero-padding to the patch
    # TODO: Ensure the padded patch matches the target_shape
    pass

def calculate_weight(patch1, patch2, h):
    """
    Assign weights to these patches based on their similarity.
    """
    # TODO: Pad patches to the same size if necessary
    # TODO: Calculate the squared difference between patches
    # TODO: Apply exponential decay function: np.exp(-diff / (h ** 2))
    pass

def create_search_window(x, y, search_window_size, image_shape, full_image=False):
    """
    Define the search window around the current pixel (x, y).
    """
    if full_image:
        return 0, 0, image_shape[1], image_shape[0]
    else:
        # TODO: Calculate search window boundaries using clip_indices
        pass

def compute_weights_and_filtered_value(image, x, y, patch_size, search_window_size, h, full_image=False):
    """
    Compute the weights and filtered value for a given pixel (x, y).
    """
    # TODO: Get search window boundaries
    # TODO: Initialize weights array and accumulators
    # TODO: Iterate over search window, calculate weights, and accumulate values
    # TODO: Compute final filtered value
    pass

def update_rectangle_bounds(rectangle, x_start, y_start, width, height):
    """
    Update the bounds of a rectangle for visualization.
    """
    rectangle.set_bounds(x_start, y_start, width, height)

def update_visual_elements(search_window, current_pixel, current_patch, x, y, search_x_start, search_y_start, search_x_end, search_y_end, patch_size):
    """
    Update the visual elements like the search window, current pixel, and patch rectangles.
    """
    half_patch = patch_size // 2
    update_rectangle_bounds(search_window, search_x_start, search_y_start, search_x_end - search_x_start, search_y_end - search_y_start)
    update_rectangle_bounds(current_pixel, x, y, 1, 1)
    update_rectangle_bounds(current_patch, x - half_patch, y - half_patch, patch_size, patch_size)

def setup_plot(image):
    """
    Set up the plot for the animation.
    """
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    # TODO: Set up subplots for original image, weight map, and output image
    # TODO: Create and return necessary plot elements (imshow objects, rectangles)
    return fig, (ax1, ax2, ax3), (im1, im2, im3), (search_window, current_pixel, current_patch)

def update_frame(frame, image, patch_size, search_window_size, h, output, search_window, current_pixel, current_patch, im2, im3, full_image=False):
    """
    Update function for each frame in the animation.
    """
    y, x = np.unravel_index(frame, image.shape)
    # TODO: Compute weights and filtered value for current pixel
    # TODO: Update output image
    # TODO: Update weight map display
    # TODO: Update visual elements
    return im2, im3, search_window, current_pixel, current_patch

def nlm_filter_animation(image, patch_size=3, search_window_size=7, h=0.1, interval=200, filename=None, full_image=False):
    """
    Create an animation of the Non-Local Means filtering process.
    """
    # TODO: Set up the plot
    # TODO: Initialize output image
    # TODO: Create animation using matplotlib.animation.FuncAnimation
    # TODO: Save animation if filename is provided
    pass

def handle_non_local_means_tab(tab, image_np, patch_size, search_window_size, filter_strength, stride, max_pixels, animation_speed, cmap):
    """
    Handle the Non-Local Means (NL-means) tab within the Streamlit app.
    """
    with tab:
        st.header("Non-Local Means (NL-means) Denoising")
        
        # TODO: Display original image
        # TODO: Process image with NL-means and visualization
        # TODO: Display denoised image
        # TODO: Add download button for denoised image
        # TODO: Handle animation controls
        
        pass

def process_nlm_with_visualization(image_np, patch_size, search_window_size, filter_strength, stride, max_pixels, animation_speed, cmap):
    """
    Process the image using the NL-means algorithm, considering the search window size,
    and prepare rectangles for visualization.
    """
    use_full_image_window = search_window_size is None
    
    # TODO: Apply NL-means filter based on window configuration
    # TODO: Create visualization rectangles
    
    denoised_image = np.copy(image_np)  # Placeholder, replace with actual denoised image
    patch_rectangle = patches.Rectangle((0, 0), patch_size, patch_size, edgecolor='r', facecolor='none')
    search_window_rectangle = patches.Rectangle((0, 0), search_window_size, search_window_size, edgecolor='b', facecolor='none')
    
    return denoised_image, patch_rectangle, search_window_rectangle



#---------------------------------------------------------------------------------------------------#


def display_weights_image_section():
    st.subheader("Weights Image")
    weights_image_placeholder = st.empty()
    return weights_image_placeholder

def display_nlm_section():
    st.subheader("Non-Local Means Result")
    nlm_image_placeholder = st.empty()
    return nlm_image_placeholder

def handle_nlm_calculation(max_pixels, image_np, kernel_size, stride, search_window_size, filter_strength,
                           original_image_placeholder, weights_image_placeholder, nlm_image_placeholder,
                           formula_placeholder, animation_speed, cmap):

    # Display original image
    display_image(original_image_placeholder, image_np, "Original Image", cmap)

    # Initialize progress bar
    progress_bar = st.progress(0)

    # Initialize output images
    weights_image = np.zeros_like(image_np, dtype=np.float32)
    nlm_image = np.zeros_like(image_np, dtype=np.float32)

    # Perform NLM calculation
    total_pixels = image_np.shape[0] * image_np.shape[1]
    for i in range(0, image_np.shape[0] - kernel_size + 1, stride):
        for j in range(0, image_np.shape[1] - kernel_size + 1, stride):
            # Calculate weights and NLM value for the current pixel
            weights, nlm_value = calculate_nlm(image_np, i, j, kernel_size, search_window_size, filter_strength)
            
            # Update weights and NLM images
            weights_image[i:i+kernel_size, j:j+kernel_size] += weights
            nlm_image[i:i+kernel_size, j:j+kernel_size] += nlm_value
            
            # Update progress
            progress = (i * image_np.shape[1] + j) / total_pixels
            progress_bar.progress(progress)
            
            # Update displays
            if (i * image_np.shape[1] + j) % animation_speed == 0:
                display_image(weights_image_placeholder, weights_image, "Weights Image", cmap)
                display_image(nlm_image_placeholder, nlm_image, "NLM Image", cmap)
                display_nlm_formula(formula_placeholder, i, j, kernel_size, search_window_size, filter_strength)
                time.sleep(0.1)

    # Normalize and finalize images
    weights_image /= np.max(weights_image)
    nlm_image /= np.max(nlm_image)

    # Display final images
    display_image(weights_image_placeholder, weights_image, "Final Weights Image", cmap)
    display_image(nlm_image_placeholder, nlm_image, "Final NLM Image", cmap)

    return weights_image, nlm_image

def calculate_nlm(image, i, j, kernel_size, search_window_size, filter_strength):
    # Implement the NLM calculation for a single pixel
    # This is a placeholder and needs to be implemented
    return np.random.rand(kernel_size, kernel_size), np.random.rand(kernel_size, kernel_size)

def display_image(placeholder, image, title, cmap):
    placeholder.image(image, caption=title, use_column_width=True, clamp=True, channels="GRAY", output_format="PNG")







# Explanation Section
def explain_nl_means():
    with st.expander("View Non-Local Means (NL-means) Explanation", expanded=False):
        st.markdown("## Non-Local Means (NL-means) Algorithm")
        st.markdown("""
        NL-means is an advanced image denoising technique that leverages the self-similarity 
        of images across different regions to achieve superior noise reduction.
        """)

        st.markdown("### Key Concepts:")
        st.markdown("""
        1. **Patch Comparison**: Instead of only comparing nearby pixels, NL-means compares patches from across the entire image.
        2. **Weighted Average**: Each pixel is denoised by taking a weighted average of all similar patches in the image.
        3. **Similarity Measure**: Patch similarity is determined using a Gaussian-weighted Euclidean distance.
        """)

        st.markdown("### Mathematical Formulation:")
        st.markdown("""
        The denoised value of a pixel \(i\) is given by:
        """)

        st.latex(r'''
        \text{NL}[v](i) = \frac{1}{C(i)} \sum_{j \in \Omega} w(i,j) v(j)
        ''')

        st.markdown("""
        Where:
        - \(v\) is the noisy image.
        - \(w(i,j)\) is the weight between pixels \(i\) and \(j\), determined by the similarity of their respective patches.
        - \(C(i)\) is a normalization factor: \(C(i) = \sum_j w(i,j)\). This ensures that the sum of weights equals 1, maintaining the intensity scale of the image.

        ### Weight Calculation:
        The weight \(w(i,j)\) is determined by the similarity between the patches centered at \(i\) and \(j\):
        """)

        st.latex(r'''
        w(i,j) = \exp\left(-\frac{\|v(N_i) - v(N_j)\|_{2,a}^2}{h^2}\right)
        ''')

        st.markdown("""
        Where:
        - \(N_i\) and \(N_j\) are neighborhoods (patches) around pixels \(i\) and \(j\).
        - \(h\) is a filtering parameter that controls the decay of the exponential function, affecting how strongly similar patches are weighted.
        - \( \| \cdot \|_{2,a} \) is the Gaussian-weighted Euclidean distance, which emphasizes differences in texture and structure.

        ### Implementation Example:
        Below is a simplified Python implementation of the NL-means core calculation:
        """)

        st.code('''
        def compute_weights_and_filtered_value(image, x, y, patch_size, search_window_size, h, full_image=False):
            """
            Compute the weights and filtered value for a given pixel (x, y).
            """
            search_x_start, search_y_start, search_x_end, search_y_end = create_search_window(x, y, search_window_size, image.shape, full_image)
            weights = np.zeros_like(image)
            center_patch = get_patch(image, x, y, patch_size)
            weighted_sum = 0
            weight_sum = 0

            for sy in range(search_y_start, search_y_end):
                for sx in range(search_x_start, search_x_end):
                    comparison_patch = get_patch(image, sx, sy, patch_size)
                    weight = calculate_weight(center_patch, comparison_patch, h)
                    weights[sy, sx] = weight
                    weighted_sum += weight * image[sy, sx]
                    weight_sum += weight

            filtered_value = weighted_sum / weight_sum if weight_sum > 0 else image[y, x]
            return filtered_value, weights
        ''', language="python")

        st.markdown("### Key Components:")
        st.markdown("""
        1. **Patch Extraction**: `get_patch()` extracts local patches around each pixel.
        2. **Search Window**: `create_search_window()` defines the area to search for similar patches.
        3. **Weight Calculation**: `calculate_weight()` computes the similarity between patches.
        4. **Weighted Average**: The filtered value is the weighted average of similar pixels.

        ### Advantages:
        - **Detail Preservation**: Preserves fine details and textures better than local methods.
        - **Noise Reduction**: Effectively removes Gaussian noise while maintaining the structure of the image.
        - **Adaptability**: Can be adjusted to different noise levels by modifying the filtering parameter \(h\).

        ### Considerations:
        - **Computational Cost**: NL-means can be computationally intensive, especially with large search windows.
        - **Parameter Sensitivity**: The choice of patch size, search window size, and \(h\) can significantly impact the results.

        ### Summary:
        NL-means is a powerful denoising technique, especially suitable for images where preserving detail is critical. However, the algorithm's effectiveness depends heavily on parameter tuning and computational resources.
        """)


# Attempt 2:

def handle_nlm_calculation(
    max_pixels, image_np, kernel_size, stride, search_window_size, filter_strength, 
    original_image_placeholder, weights_image_placeholder, nlm_image_placeholder, 
    formula_placeholder, animation_speed, cmap
):
    """Handle the Non-Local Means calculation and update Streamlit placeholders."""

    for i in range(1, max_pixels + 1) if st.session_state.is_animating else [max_pixels]:

        nlm_image, weights_image, last_x, last_y = calculate_nlm(
            image_np, kernel_size, stride, search_window_size, filter_strength, i, st.session_state.cache 
        )

        display_nlm_formula(formula_placeholder, last_x, last_y, kernel_size, search_window_size, filter_strength)

        # # 3. Display original image with current pixel highlighted
        # fig_original, axs_original = plt.subplots(1, 1, figsize=(5, 5))
        # original_image_placeholder.pyplot(
        #     update_plot(fig_original, [axs_original], image_np, [nlm_image], last_x, last_y, kernel_size, cmap)
        # )

        # 4. Display weights image 
        weights_image_placeholder.image(weights_image, caption="Weights Image", use_column_width=True)


        # 5. Display NLM filtered image
        nlm_image_placeholder.image(nlm_image, caption="NLM Filtered Image", use_column_width=True)

        if not st.session_state.is_animating:
            break

        # time.sleep(animation_speed)

    return weights_image, nlm_image

def calculate_nlm(image_np, kernel_size, stride, search_window_size, filter_strength, max_pixels, cache):
    """Calculate Non-Local Means filtered image and weights image."""

    output_height = (image_np.shape[0] - kernel_size) // stride + 1
    output_width = (image_np.shape[1] - kernel_size) // stride + 1

    total_pixels = min(max_pixels, output_height * output_width)

    nlm_image = np.zeros_like(image_np)  # Initialize NLM filtered image
    weights_image = np.zeros_like(image_np)  # Initialize weights image

    cache_key = (image_np.shape, kernel_size, stride, search_window_size, filter_strength)
    if cache_key not in cache:
        cache[cache_key] = {}

    for pixel in range(total_pixels):
        row, col = divmod(pixel, output_width)
        y, x = row * stride, col * stride  # Top-left corner of the kernel

        if (row, col) in cache[cache_key]:
            nlm_pixel_value, weights_pixel_value = cache[cache_key][(row, col)]
        else:
            nlm_pixel_value, weights_pixel_value = calculate_nlm_pixel(
                image_np, y, x, kernel_size, search_window_size, filter_strength
            )
            cache[cache_key][(row, col)] = (nlm_pixel_value, weights_pixel_value)

        # Update NLM and weights images 
        nlm_image[y:y + kernel_size, x:x + kernel_size] = nlm_pixel_value

        # Calculate the slice of weights_image corresponding to the search window
        weights_start_y = max(0, y - search_window_size // 2)
        weights_end_y = min(weights_image.shape[0], y + search_window_size // 2 + kernel_size)
        weights_start_x = max(0, x - search_window_size // 2)
        weights_end_x = min(weights_image.shape[1], x + search_window_size // 2 + kernel_size)

        weights_image[weights_start_y:weights_end_y, weights_start_x:weights_end_x] = weights_pixel_value

    last_x, last_y = x, y
    return nlm_image, weights_image, last_x, last_y

def calculate_nlm_pixel(image, y, x, kernel_size, search_window_size, filter_strength):
    """Calculate NLM filtered value and weights for a single pixel."""

    # 1. Extract neighborhood around the current pixel
    neighborhood_p = image[y:y + kernel_size, x:x + kernel_size]

    # 2. Define search window
    search_window_start_y = max(0, y - search_window_size // 2)
    search_window_end_y = min(image.shape[0] - kernel_size, y + search_window_size // 2)
    search_window_start_x = max(0, x - search_window_size // 2)
    search_window_end_x = min(image.shape[1] - kernel_size, x + search_window_size // 2)

    # 3. Calculate weights for pixels in the search window
    weights = []
    for i in range(search_window_start_y, search_window_end_y):
        for j in range(search_window_start_x, search_window_end_x):
            neighborhood_q = image[i:i + kernel_size, j:j + kernel_size]
            distance = np.sum((neighborhood_p - neighborhood_q)**2)  # Euclidean distance
            weight = np.exp(-distance / (filter_strength**2))
            weights.append(weight)

    # 4. Calculate weighted average (NLM filtered value)
    image_slice = image[search_window_start_y:search_window_end_y, 
                        search_window_start_x:search_window_end_x]

    weights_reshaped = np.array(weights).reshape(image_slice.shape)  

    nlm_pixel_value = np.sum(weights_reshaped * image_slice) / np.sum(weights) 

    # 5. Normalize weights to create weights image (for visualization)
    weights_pixel_value = np.array(weights).reshape(image_slice.shape) 
    weights_pixel_value = weights_pixel_value / np.max(weights_pixel_value) 
    weights_pixel_value = pad_weights(weights_pixel_value, kernel_size) # Call the padding function

    return nlm_pixel_value, weights_pixel_value

def pad_weights(weights, kernel_size):
    """Pad the weights array to match the kernel size."""
    pad_y = (kernel_size - weights.shape[0]) // 2
    pad_x = (kernel_size - weights.shape[1]) // 2
    padded_weights = np.pad(weights, ((pad_y, pad_y), (pad_x, pad_x)), 'constant')
    return padded_weights

