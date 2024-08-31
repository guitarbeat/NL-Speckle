import streamlit as st
from streamlit_image_comparison import image_comparison
import numpy as np
import matplotlib.pyplot as plt


def handle_animation_controls():
    """Handle the start/pause/stop animation buttons."""
    col1, col2, col3 = st.columns(3)
    with col1:
        start_button = st.button('Start')
    with col2:
        pause_button = st.button('Pause')
    with col3:
        stop_button = st.button('Stop')

    if start_button:
        st.session_state.is_animating = True
        st.session_state.is_paused = False
    if pause_button:
        st.session_state.is_paused = not st.session_state.is_paused
    if stop_button:
        st.session_state.is_animating = False
        st.session_state.is_paused = False

def create_section(title: str, expanded_main: bool, expanded_zoomed: bool):
    with st.expander(title, expanded=expanded_main):
        main_placeholder = st.empty()
        with st.expander(f"Zoomed-in {title.split()[0]}"):
            zoomed_placeholder = st.empty()
    return main_placeholder, zoomed_placeholder

def apply_colormap_to_images(img1, img2, cmap):
    img1_normalized = (img1 - np.min(img1)) / (np.max(img1) - np.min(img1))
    img2_normalized = (img2 - np.min(img2)) / (np.max(img2) - np.min(img2))

    img1_colored = cmap(img1_normalized)
    img2_colored = cmap(img2_normalized)

    img1_uint8 = (img1_colored[:, :, :3] * 255).astype(np.uint8)
    img2_uint8 = (img2_colored[:, :, :3] * 255).astype(np.uint8)
    
    return img1_uint8, img2_uint8

#----------------------------- Comparison Plotting Stuff ------------------------------ #
def display_image_comparison(img1, img2, label1, label2, cmap):
    img1_uint8, img2_uint8 = apply_colormap_to_images(img1, img2, cmap)
    image_comparison(img1=img1_uint8, img2=img2_uint8, label1=label1, label2=label2, make_responsive=True)
    st.subheader("Selected Images")
    st.image([img1_uint8, img2_uint8], caption=[label1, label2])

#----------------------------- Speckle Tab Plotting Stuff ----------------------------- #
def display_data_and_zoomed_view(data, full_data, last_x, last_y, stride, title, data_placeholder, zoomed_placeholder, cmap="viridis", zoom_size=1, fontsize=10, text_color="red"):
    """
    Display data and its zoomed-in view using Streamlit placeholders, with color scaling based on full data.
    
    Parameters:
    - data: 2D numpy array of the data to display
    - full_data: 2D numpy array of the full image data for color scaling
    - last_x, last_y: Coordinates of the last processed point
    - stride: Step size between processed points
    - title: Title for the plot
    - data_placeholder: Streamlit placeholder for the full data plot
    - zoomed_placeholder: Streamlit placeholder for the zoomed view
    - cmap: Colormap to use (default: "viridis")
    - zoom_size: Size of the zoomed area (default: 1)
    - fontsize: Font size for annotations (default: 10)
    - text_color: Color of the annotations (default: "red")
    """
    # Display full data
    fig_full, ax_full = plt.subplots(figsize=(5, 5))
    vmin, vmax = np.min(full_data), np.max(full_data)
    ax_full.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
    ax_full.set_title(title)
    ax_full.axis("off")
    data_placeholder.pyplot(fig_full)
    
    # Display zoomed data
    zoomed_data = data[last_y // stride : last_y // stride + zoom_size, 
                       last_x // stride : last_x // stride + zoom_size]
    
    fig_zoom, ax_zoom = plt.subplots(figsize=(5, 5))
    ax_zoom.imshow(zoomed_data, cmap=cmap, vmin=vmin, vmax=vmax)
    ax_zoom.set_title(f"Zoomed-In {title}", fontsize=12)
    ax_zoom.axis("off")
    
    for i, row in enumerate(zoomed_data):
        for j, val in enumerate(row):
            ax_zoom.text(j, i, f"{val:.2f}", ha="center", va="center", color=text_color, fontsize=fontsize)
    
    fig_zoom.tight_layout(pad=2)
    zoomed_placeholder.pyplot(fig_zoom)

def display_kernel_view(kernel_data, full_image_data, title, placeholder, cmap="viridis", fontsize=10, text_color="red"):
    """
    Display the kernel view with pixel values annotated, using the color scale of the full image.
    
    Parameters:
    - kernel_data: 2D numpy array of the kernel data
    - full_image_data: 2D numpy array of the full image data
    - title: Title for the plot
    - placeholder: Streamlit placeholder for the kernel view
    - cmap: Colormap to use (default: "viridis")
    - fontsize: Font size for annotations (default: 10)
    - text_color: Color of the annotations (default: "red")
    """
    fig, ax = plt.subplots(figsize=(5, 5))
    
    vmin, vmax = np.min(full_image_data), np.max(full_image_data)
    im = ax.imshow(kernel_data, cmap=cmap, vmin=vmin, vmax=vmax)
    
    ax.set_title(title, fontsize=12)
    ax.axis("off")
    
    for i, row in enumerate(kernel_data):
        for j, val in enumerate(row):
            val_str = f"{val:.2f}"  # Adjust precision as needed
            ax.text(j, i, val_str, ha="center", va="center", color=text_color, fontsize=fontsize)
    
    fig.tight_layout(pad=2)
    placeholder.pyplot(fig)

def display_speckle_contrast_formula(formula_placeholder):
    with formula_placeholder.container():
        st.latex(r'SC_{{({}, {})}} = \frac{{\sigma}}{{\mu}} = \frac{{{:.3f}}}{{{:.3f}}} = {:.3f}'.format(0, 0, 0.0, 0.0, 0.0))
