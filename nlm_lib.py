import numpy as np
import streamlit as st
from PIL import Image
from config import configure_sidebar, initialize_session_state, load_image, process_image, get_image_download_link
from ui.components import (
    handle_animation_controls,
    display_original_image_section,
    display_speckle_contrast_section,
    display_image_comparison,
    apply_colormap_to_images
)
from helpers import (
    clear_axes,
    configure_axes,
    add_kernel_rectangle,
    display_speckle_contrast_process
)

# -------------------------------------------
# Utility Functions
# -------------------------------------------

# -- Index Management --
def clip_indices(value, min_value, max_value):
    """
    Ensure that the value is within the specified bounds.
    
    - Purpose: Prevents out-of-bound errors during patch extraction or search window creation.
    - Implementation: Returns the value clamped between min_value and max_value.
    """
    return max(min_value, min(value, max_value))

# -- Patch Extraction and Manipulation --
def get_patch(image, x, y, patch_size):
    """
    Extract a patch from the image centered at (x, y).
    
    - Purpose: Provides a subregion of the image that will be compared with other patches.
    - Implementation: Calculates the start and end indices for both x and y directions, then extracts the subarray.
    - Expected Outcome: A patch (subarray) of the specified size centered at (x, y).
    """
    half = patch_size // 2
    y_start = clip_indices(y - half, 0, image.shape[0])
    y_end = clip_indices(y + half + 1, 0, image.shape[0])
    x_start = clip_indices(x - half, 0, image.shape[1])
    x_end = clip_indices(x + half + 1, 0, image.shape[1])
    return image[y_start:y_end, x_start:x_end]

def pad_patch(patch, target_shape):
    """
    Pad a given patch to match the target dimensions.
    
    - Purpose: Ensures that patches of different sizes can be compared by padding them to the same size.
    - Implementation: Uses numpy's pad function to add zero-padding to the patch.
    - Expected Outcome: A padded patch with the specified target dimensions.
    """
    target_height, target_width = target_shape
    return np.pad(patch, ((0, target_height - patch.shape[0]), (0, target_width - patch.shape[1])), mode='constant')

# -------------------------------------------
# Core Algorithm: Non-Local Means Filter
# -------------------------------------------

# -- Weight Calculation --
def calculate_weight(patch1, patch2, h):
    """
    Assign weights to these patches based on their similarity.
    
    - Purpose: Determines the contribution of each patch to the final denoised pixel value.
    - Implementation: Calculates the squared difference between two patches, applies an exponential decay function.
    - Expected Outcome: A weight value that indicates how similar patch2 is to patch1.
    """
    target_shape = (max(patch1.shape[0], patch2.shape[0]), max(patch1.shape[1], patch2.shape[1]))
    pad1 = pad_patch(patch1, target_shape)
    pad2 = pad_patch(patch2, target_shape)
    diff = np.sum((pad1 - pad2) ** 2)
    return np.exp(-diff / (h ** 2))

# -- Search Window Definition --
def create_search_window(x, y, search_window_size, image_shape, full_image=False):
    """
    Define the search window around the current pixel (x, y).
    
    - Purpose: Specifies the area in the image where similar patches will be searched.
    - Implementation: Either creates a search window of the specified size or sets it to cover the entire image.
    - Expected Outcome: Coordinates defining the boundaries of the search window.
    """
    if full_image:
        search_x_start, search_y_start = 0, 0
        search_x_end, search_y_end = image_shape[1], image_shape[0]
    else:
        half_search = search_window_size // 2
        search_x_start = clip_indices(x - half_search, 0, image_shape[1])
        search_x_end = clip_indices(x + half_search + 1, 0, image_shape[1])
        search_y_start = clip_indices(y - half_search, 0, image_shape[0])
        search_y_end = clip_indices(y + half_search + 1, 0, image_shape[0])
    
    return search_x_start, search_y_start, search_x_end, search_y_end

# -- Weight Calculation and Pixel Filtering --
def compute_weights_and_filtered_value(image, x, y, patch_size, search_window_size, h, full_image=False):
    """
    Compute the weights and filtered value for a given pixel (x, y).
    
    - Purpose: Determines the new value of the pixel by averaging similar patches found within the search window.
    - Implementation: Iterates over the search window, compares patches, and calculates weights to apply for the new pixel value.
    - Expected Outcome: A denoised pixel value based on the weighted average of similar patches, and a weight map.
    """
    search_x_start, search_y_start, search_x_end, search_y_end = create_search_window(x, y, search_window_size, image.shape, full_image=full_image)
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

# -------------------------------------------
# Visualization and Animation
# -------------------------------------------

# -- Visualization Utilities --
def update_rectangle_bounds(rectangle, x_start, y_start, width, height):
    """
    Update the bounds of a rectangle for visualization.
    
    - Purpose: Dynamically updates the position and size of rectangles used to highlight patches or search windows.
    - Implementation: Sets the rectangle's bounds based on the provided coordinates.
    - Expected Outcome: A rectangle that visually represents the current search window or patch.
    """

# -- Visual Elements Update --
def update_visual_elements(search_window, current_pixel, current_patch, x, y, search_x_start, search_y_start, search_x_end, search_y_end, patch_size):
    """
    Update the visual elements like the search window, current pixel, and patch rectangles.
    
    - Purpose: Synchronize the UI elements with the current state of the algorithm, showing which pixel and patches are being processed.
    - Implementation: Uses update_rectangle_bounds to adjust the positions and sizes of rectangles.
    - Expected Outcome: A visualization where the search window, patch, and pixel highlights move as the algorithm processes the image.
    """

# -- Plot Setup for Animation --
def setup_plot(image):
    """
    Set up the plot for the animation.
    
    - Purpose: Prepares the figure and axes for displaying the input image, weights, and output image during the animation.
    - Implementation: Creates a 3-part subplot, configures axes, and sets up colorbars.
    - Expected Outcome: A figure ready to display the step-by-step process of the NL-means algorithm.
    """

# -- Frame Update for Animation --
def update_frame(frame, image, patch_size, search_window_size, h, output, search_window, current_pixel, current_patch, im2, im3, full_image=False):
    """
    Update function for each frame in the animation.
    
    - Purpose: Progresses the animation by processing one pixel at a time and updating the visual elements accordingly.
    - Implementation: Calls compute_weights_and_filtered_value to process the current pixel, then updates the UI elements and images.
    - Expected Outcome: A frame-by-frame update of the animation, showing how the NL-means filter processes the image.
    """

# -- Non-Local Means Filter Animation --
def nlm_filter_animation(image, patch_size=3, search_window_size=7, h=0.1, interval=200, filename=None, full_image=False):
    """
    Create an animation of the Non-Local Means filtering process.
    
    - Purpose: Visualize the NL-means algorithm as it processes the image, or simply apply the filter without animation.
    - Implementation: Sets up the plot, processes the image frame by frame, and optionally saves the animation as a GIF.
    - Expected Outcome: Either an animated GIF showing the filtering process or a fully denoised image.
    """


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
