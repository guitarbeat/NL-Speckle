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


def display_speckle_contrast_formula(formula_placeholder):
    with formula_placeholder.container():
        st.latex(r'SC_{{({}, {})}} = \frac{{\sigma}}{{\mu}} = \frac{{{:.3f}}}{{{:.3f}}} = {:.3f}'.format(0, 0, 0.0, 0.0, 0.0))

def display_image_comparison_error(image_choice_1, image_choice_2):
    if image_choice_1 == image_choice_2 and image_choice_1:
        st.error("Please select two different images for comparison.")
    else:
        st.info("Select two images to compare.")

def apply_colormap_to_images(img1, img2, cmap):
    img1_normalized = (img1 - np.min(img1)) / (np.max(img1) - np.min(img1))
    img2_normalized = (img2 - np.min(img2)) / (np.max(img2) - np.min(img2))

    img1_colored = cmap(img1_normalized)
    img2_colored = cmap(img2_normalized)

    img1_uint8 = (img1_colored[:, :, :3] * 255).astype(np.uint8)
    img2_uint8 = (img2_colored[:, :, :3] * 255).astype(np.uint8)
    
    return img1_uint8, img2_uint8


def display_image_comparison(img1, img2, label1, label2, cmap):
    img1_uint8, img2_uint8 = apply_colormap_to_images(img1, img2, cmap)
    image_comparison(img1=img1_uint8, img2=img2_uint8, label1=label1, label2=label2, make_responsive=True)
    st.subheader("Selected Images")
    st.image([img1_uint8, img2_uint8], caption=[label1, label2])



def display_data_and_zoomed_view(data, last_x, last_y, stride, title, data_placeholder, zoomed_placeholder, cmap="viridis", zoom_size=1):
    """
    Display data and its zoomed-in view.
    
    Parameters:
    - data: 2D numpy array of the data to display
    - last_x, last_y: Coordinates of the last processed point
    - stride: Step size between processed points
    - title: Title for the plot
    - data_placeholder: Streamlit placeholder for the full data plot
    - zoomed_placeholder: Streamlit placeholder for the zoomed view
    - cmap: Colormap to use (default: "viridis")
    - zoom_size: Size of the zoomed area (default: 1)
    """
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.imshow(data, cmap=cmap)
    ax.set_title(title)
    ax.axis("off")
    data_placeholder.pyplot(fig)
    
    zoomed_data = data[last_y // stride : last_y // stride + zoom_size, 
                       last_x // stride : last_x // stride + zoom_size]
    zoomed_placeholder.pyplot(
        plot_zoomed_views(
            [zoomed_data],
            ["Zoomed-In " + title],
            cmap
        )
    )


def plot_zoomed_views(zoomed_data, titles, cmap="viridis", fontsize=10, text_color="red"):
    """
    Plot zoomed-in views with values annotated.
    
    Parameters:
    - zoomed_data: list of 2D numpy arrays to display
    - titles: list of titles for each zoomed view
    - cmap: colormap to use (default: "viridis")
    - fontsize: font size for annotations (default: 10)
    - text_color: color of the annotations (default: "red")
    """
    zoom_fig, zoom_axs = plt.subplots(1, len(zoomed_data), figsize=(5 * len(zoomed_data), 5))
    zoom_axs = zoom_axs if isinstance(zoom_axs, np.ndarray) else [zoom_axs]
    
    for ax, data, title in zip(zoom_axs, zoomed_data, titles):
        im = ax.imshow(data, cmap=cmap)
        ax.set_title(title, fontsize=12)
        ax.axis("off")
        plt.colorbar(im, ax=ax)
        for i, row in enumerate(data):
            for j, val in enumerate(row):
                ax.text(j, i, f"{val:.3f}", ha="center", va="center", color=text_color, fontsize=fontsize)
    
    zoom_fig.tight_layout(pad=2)
    return zoom_fig

