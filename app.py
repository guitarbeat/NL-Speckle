import streamlit as st
import streamlit_nested_layout  # noqa: F401

# Project Structure:
# .
# ├── app.py               # Main application file, contains Streamlit app structure and high-level logic.
# ├── helpers.py           # Helper functions for plotting and displaying results.
# ├── image_processing_lib # Functions for speckle contrast and non-local means calculations.
# ├── ui_components.py     # Streamlit UI components and layouts.
# └── config.py            # Configuration variables and constants and image loading and processing.

from config import set_page_config
from ui_components import handle_comparison_tab, configure_sidebar
from image_processing_lib import handle_image_analysis


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

    # Tab 1: Speckle Contrast Calculation
    speckle_results = handle_image_analysis(
        tabs[0], image_np, kernel_size, stride, max_pixels, animation_speed, cmap, "speckle"
    )
    std_dev_image, speckle_contrast_image, mean_image = speckle_results[:3]

    # Tab 2: Non-Local Means Denoising
    nlm_results = handle_image_analysis(
        tabs[1], image_np, kernel_size, stride, max_pixels, animation_speed, cmap, "nlm",
        search_window_size=search_window_size, filter_strength=filter_strength
    )
    denoised_image = nlm_results[0] if nlm_results else None

    # Tab 3: Comparison
    handle_comparison_tab(tabs[2], cmap, std_dev_image, speckle_contrast_image, mean_image, image_np, denoised_image)

if __name__ == "__main__":
    main()

