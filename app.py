import streamlit as st
import streamlit_nested_layout  # noqa: F401

from config import set_page_config
from ui_components import handle_comparison_tab, configure_sidebar
from image_processing import handle_image_analysis

# Constants
TABS = ["Speckle Contrast Calculation", "Non-Local Means Denoising", "Speckle Contrast Comparison"]



def process_speckle_contrast(tab, image_np, kernel_size, stride, max_pixels, animation_speed, cmap):
    results = handle_image_analysis(
        tab, image_np, kernel_size, stride, max_pixels, animation_speed, cmap, "speckle"
    )
    return results[:3]  # std_dev_image, speckle_contrast_image, mean_image

def process_nlm_denoising(tab, image_np, kernel_size, stride, max_pixels, animation_speed, cmap, search_window_size, filter_strength):
    results = handle_image_analysis(
        tab, image_np, kernel_size, stride, max_pixels, animation_speed, cmap, "nlm",
        search_window_size=search_window_size, filter_strength=filter_strength
    )
    return results[0] if results else None  # denoised_image

def main():
    set_page_config()

    # Configure sidebar and get parameters
    image, kernel_size, search_window_size, filter_strength, stride, cmap, animation_speed, image_np = configure_sidebar()

    # Calculate max processable pixels
    max_processable_pixels = (image.width - kernel_size + 1) * (image.height - kernel_size + 1)
    max_pixels = st.slider("Pixels to process", 1, max_processable_pixels, max_processable_pixels)

    # Create tabs
    tabs = st.tabs(TABS)

    # Process images
    std_dev_image, speckle_contrast_image, mean_image = process_speckle_contrast(
        tabs[0], image_np, kernel_size, stride, max_pixels, animation_speed, cmap
    )
    
    denoised_image = process_nlm_denoising(
        tabs[1], image_np, kernel_size, stride, max_pixels, animation_speed, cmap,
        search_window_size, filter_strength
    )

    # Handle comparison tab
    handle_comparison_tab(
        tab=tabs[2],
        cmap_name=cmap,
        images={
            'Unprocessed Image': image_np,
            'Standard Deviation': std_dev_image,
            'Speckle Contrast': speckle_contrast_image,
            'Mean Filter': mean_image,
            'Denoised Image': denoised_image
        }
    )

if __name__ == "__main__":
    main()

