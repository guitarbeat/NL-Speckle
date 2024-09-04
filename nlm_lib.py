
import streamlit as st

from helpers import create_plot  # Make sure to import this

def handle_non_local_means_tab(tab, image_np, kernel_size, stride, search_window_size, filter_strength, max_pixels, animation_speed, cmap):
    with tab:
        st.header("Non-Local Means Denoising", divider="rainbow")

        formula_placeholder = st.empty()
        display_nlm_formula(formula_placeholder, 0, 0, kernel_size, search_window_size, filter_strength)

        # Add placeholders for the original image and other visualizations
        placeholders = create_nlm_placeholders()

        # Update the original image with search window
        update_nlm_original_image(image_np, kernel_size, search_window_size, cmap, placeholders)

        return 

def create_nlm_placeholders():
    """Create and return a dictionary of placeholders for NLM visualizations."""
    placeholders = {
        'original_image': st.empty(),
        # Add more placeholders as needed for other visualizations
    }
    return placeholders

def update_nlm_original_image(image_np, kernel_size, search_window_size, cmap, placeholders):
    """Update the original image display with the current kernel and search window."""
    # For demonstration, we'll use the center of the image
    center_y, center_x = image_np.shape[0] // 2, image_np.shape[1] // 2
    
    fig_original = create_plot(
        image_np, [], center_x, center_y, kernel_size,
        ["Original Image with Current Kernel and Search Window"], cmap=cmap, 
        search_window=search_window_size, figsize=(5, 5)
    )
    placeholders['original_image'].pyplot(fig_original)

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

