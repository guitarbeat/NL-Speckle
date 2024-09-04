
import streamlit as st
import streamlit_nested_layout  # noqa: F401

# Project Structure:
# .
# ├── app.py               # Main application file, contains Streamlit app structure and high-level logic.
# ├── helpers.py           # Helper functions for plotting and displaying results.
# ├── speckle_lib          # Functions for speckle contrast and non-local means calculations.
# ├── ui_components.py     # Streamlit UI components and layouts.
# └── config.py            # Configuration variables and constants and image loading and processing.

from config import set_page_config
from ui_components import handle_comparison_tab, configure_sidebar
from speckle_lib import handle_speckle_contrast_tab
from nlm_lib import handle_non_local_means_tab

set_page_config()

st.title("Speckle Contrast Analysis")
st.markdown("""
Welcome to the Interactive Speckle Contrast Analysis tool! This application allows you to visualize and analyze speckle contrast in images.

**How to use this tool:**
1. Choose an image source (preloaded or upload your own) in the sidebar.
2. Adjust the processing parameters and visualization settings.
3. Explore the speckle contrast analysis results in the tabs below.
4. Use the animation controls to see the analysis process in action.

Let's get started!
""")

def main():
 
    image, kernel_size, search_window_size, filter_strength, stride, cmap, animation_speed, image_np = configure_sidebar()


    max_pixels = st.slider("Pixels to process", 1, image.width * image.height, image.width * image.height)

    tabs = st.tabs(["Speckle Contrast Calculation", "Non-Local Means Denoising", "Speckle Contrast Comparison"])

    # Tab 1
    std_dev_image, speckle_contrast_image, mean_image, image_np = handle_speckle_contrast_tab(
        tabs[0], image_np, kernel_size, stride, max_pixels, animation_speed, cmap
    )
    handle_non_local_means_tab(tabs[1], image_np, kernel_size, stride, search_window_size, filter_strength, max_pixels, animation_speed, cmap)

    # Tab 3
    handle_comparison_tab(tabs[2], cmap, std_dev_image, speckle_contrast_image, mean_image, image_np)


if __name__ == "__main__":
    main()

