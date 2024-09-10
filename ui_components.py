import streamlit as st
from typing import Dict, Any, Tuple
import numpy as np
import matplotlib.pyplot as plt
from streamlit_image_comparison import image_comparison
from PIL import Image

from image_processing import (update_pixels, 
                              load_image, get_search_window_size,
                              update_images_analyze_and_visualize
                               )

from constants import COLOR_MAPS, PRELOADED_IMAGES

# ---------------------------- UI Components and Layout ---------------------------- #

def process_and_display_images(analysis_params):
    for technique, tab in zip(["speckle", "nlm"], st.session_state.tabs[:2]):
        placeholders = create_placeholders_and_sections(technique, tab, analysis_params['show_full_processed'])
        st.session_state[f"{technique}_placeholders"] = placeholders
        
        params, results = update_images_analyze_and_visualize(
            image_np=analysis_params['image_np'],
            kernel_size=analysis_params['kernel_size'],
            cmap=analysis_params['cmap'],
            technique=technique,
            search_window_size=analysis_params['search_window_size'],
            filter_strength=analysis_params['filter_strength'],
            show_full_processed=analysis_params['show_full_processed'],
            update_state=True,
            handle_visualization=True
        )
        st.session_state[f"{technique}_results"] = results

def create_placeholders_and_sections(technique: str, tab: st.delta_generator.DeltaGenerator, show_full_processed: bool) -> dict:
    """
    Create placeholders for the Streamlit UI based on the selected technique and display options.

    Args:
        technique (str): The image processing technique ("speckle" or "nlm").
        tab (st.delta_generator.DeltaGenerator): The Streamlit tab to display the UI components.
        show_full_processed (bool): Whether to show the full processed image or zoomed views.

    Returns:
        Dict[str, Any]: A dictionary of placeholders for the UI components.
    """
    with tab:
        placeholders = {'formula': st.empty(), 'original_image': st.empty()}

        filter_options = {
            "speckle": ["Mean Filter", "Std Dev Filter", "Speckle Contrast"],
            "nlm": ["Weight Map", "NL-Means Image", "Difference Map"]
        }

        selected_filters = st.multiselect(
            "Select views to display",
            filter_options[technique],
            default={"speckle": ["Speckle Contrast"], "nlm": ["NL-Means Image"]}[technique]
        )

        columns = st.columns(len(selected_filters) + 1)  # +1 for the original image

        with columns[0]:
            if not show_full_processed:
                placeholders['original_image'], placeholders['zoomed_original_image'] = create_section("Original Image", expanded_main=True)
            else:
                placeholders['original_image'] = st.empty()

        for i, filter_name in enumerate(selected_filters, start=1):
            with columns[i]:
                key = filter_name.lower().replace(" ", "_")
                if show_full_processed:
                    placeholders[key] = st.empty()
                else:
                    placeholders[key], placeholders[f'zoomed_{key}'] = create_section(filter_name, expanded_main=True)

        if not show_full_processed:
            placeholders['zoomed_kernel'] = placeholders.get('zoomed_kernel', st.empty())

        return placeholders

def create_section(title: str, expanded_main: bool = False, expanded_zoomed: bool = False) -> Tuple[Any, Any]:
    """
    Create a section with main and zoomed views in the Streamlit UI.

    Args:
        title (str): The title of the section.
        expanded_main (bool): Whether the main view should be expanded by default. Defaults to False.
        expanded_zoomed (bool): Whether the zoomed view should be expanded by default. Defaults to False.

    Returns:
        Tuple[st.empty, st.empty]: Placeholders for the main and zoomed views.
    """
    with st.expander(title, expanded=expanded_main):
        main_placeholder = st.empty()
        zoomed_placeholder = st.expander(f"Zoomed-in {title.split()[0]}", expanded=expanded_zoomed).empty()
    return main_placeholder, zoomed_placeholder

def normalize_and_apply_cmap(img: np.ndarray, cmap_name: str) -> np.ndarray:
    """
    Normalize an image and apply a colormap.

    Args:
        img (np.ndarray): The image to normalize and apply colormap to.
        cmap_name (str): The name of the colormap to apply.

    Returns:
        np.ndarray: The normalized image with colormap applied.
    """
    cmap = plt.get_cmap(cmap_name)
    normalized = (img - np.min(img)) / (np.max(img) - np.min(img))
    return (cmap(normalized)[:, :, :3] * 255).astype(np.uint8)

def display_image_comparison(image1: np.ndarray, image2: np.ndarray, label1: str, label2: str):
    """
    Display an interactive image comparison tool and the selected images.

    Args:
        image1 (np.ndarray): The first image to compare.
        image2 (np.ndarray): The second image to compare.
        label1 (str): The label for the first image.
        label2 (str): The label for the second image.
    """
    image_comparison(img1=image1, img2=image2, label1=label1, label2=label2, make_responsive=True)
    st.subheader("Selected Images")
    st.image([image1, image2], caption=[label1, label2])

def display_comparison_options(image1: np.ndarray, image2: np.ndarray):
    """
    Display additional comparison options when the same image is selected for comparison.

    Args:
        image1 (np.ndarray): The first image.
        image2 (np.ndarray): The second image.
    """
    st.subheader("Comparison Options")
    diff_map = np.abs(image1 - image2)
    st.image(diff_map, caption="Difference Map", use_column_width=True)

def handle_image_comparison(tab: st.delta_generator.DeltaGenerator, cmap_name: str, images: Dict[str, np.ndarray]):
    """
    Display an interactive image comparison tool in the Streamlit UI.

    Args:
        tab (st.delta_generator.DeltaGenerator): The Streamlit tab to display the image comparison tool.
        cmap_name (str): The name of the colormap to apply to the images.
        images (Dict[str, np.ndarray]): A dictionary of images to compare, with keys as image names and values as image data.
    """
    with tab:
        st.header("Image Comparison")
        
        if images is None or len(images) == 0:
            st.warning("No images available for comparison.")
            return

        available_images = list(images.keys())
        col1, col2 = st.columns(2)
        image_choice_1 = col1.selectbox('Select first image to compare:', [''] + available_images, index=0)
        image_choice_2 = col2.selectbox('Select second image to compare:', [''] + available_images, index=0)
        
        if image_choice_1 and image_choice_2:
            if image_choice_1 != image_choice_2:
                img1, img2 = images[image_choice_1], images[image_choice_2]
                img1_uint8, img2_uint8 = map(lambda img: normalize_and_apply_cmap(img, cmap_name), [img1, img2])
                display_image_comparison(img1_uint8, img2_uint8, image_choice_1, image_choice_2)
            else:
                st.error("Please select two different images for comparison.")
                display_comparison_options(images[image_choice_1], images[image_choice_2])
        else:
            st.info("Select two images to compare.")

def create_placeholders(tab: st.delta_generator.DeltaGenerator, technique: str, show_full_processed: bool) -> Dict[str, Any]:
    """
    Create placeholders for the Streamlit UI based on the selected technique and display options.

    Args:
        tab (st.delta_generator.DeltaGenerator): The Streamlit tab to display the UI components.
        technique (str): The image processing technique ("speckle" or "nlm").
        show_full_processed (bool): Whether to show the full processed image or zoomed views.

    Returns:
        Dict[str, Any]: A dictionary of placeholders for the UI components.
    """
    with tab:
        placeholders = {'formula': st.empty(), 'original_image': st.empty()}

        filter_options = {
            "speckle": ["Mean Filter", "Std Dev Filter", "Speckle Contrast"],
            "nlm": ["Weight Map", "NL-Means Image", "Difference Map"]
        }

        selected_filters = st.multiselect(
            "Select views to display",
            filter_options[technique],
            default={"speckle": ["Speckle Contrast"], "nlm": ["NL-Means Image"]}[technique]
        )

        columns = st.columns(len(selected_filters) + 1)  # +1 for the original image

        with columns[0]:
            if not show_full_processed:
                placeholders['original_image'], placeholders['zoomed_original_image'] = create_section("Original Image", expanded_main=True)
            else:
                placeholders['original_image'] = st.empty()

        for i, filter_name in enumerate(selected_filters, start=1):
            with columns[i]:
                key = filter_name.lower().replace(" ", "_")
                if show_full_processed:
                    placeholders[key] = st.empty()
                else:
                    placeholders[key], placeholders[f'zoomed_{key}'] = create_section(filter_name, expanded_main=True)

        if not show_full_processed:
            placeholders['zoomed_kernel'] = placeholders.get('zoomed_kernel', st.empty())

        return placeholders

def create_tabs():
    return st.tabs(["Speckle", "NL-Means", "Image Comparison"])

def update_session_state(image: Image.Image, processing_params: Dict[str, Any], 
                         display_options: Dict[str, Any], advanced_options: Dict[str, Any]) -> None:
    st.session_state.image_np = advanced_options["image_np"]
    st.session_state.kernel_size = processing_params["kernel_size"]
    st.session_state.cmap = processing_params["cmap"]
    st.session_state.technique = processing_params["technique"]
    st.session_state.search_window_size = processing_params["search_window_size"]
    st.session_state.filter_strength = processing_params["filter_strength"]
    st.session_state.show_full_processed = display_options["show_full_processed"]
    
def setup_sidebar() -> Dict[str, Any]:
    if 'current_position' not in st.session_state:
        st.session_state.current_position = 1

    with st.sidebar:
        st.title("Image Processing Settings")
        image = setup_image_source()
        processing_params = setup_processing_parameters(image)
        display_options = setup_display_options(image, processing_params['kernel_size'])
        advanced_options = setup_advanced_options(image)

    # Initialize session state variables
    if 'image_np' not in st.session_state:
        st.session_state.image_np = advanced_options["image_np"]
    if 'kernel_size' not in st.session_state:
        st.session_state.kernel_size = processing_params["kernel_size"]
    if 'cmap' not in st.session_state:
        st.session_state.cmap = processing_params["cmap"]
    if 'technique' not in st.session_state:
        st.session_state.technique = processing_params["technique"]
    if 'search_window_size' not in st.session_state:
        st.session_state.search_window_size = processing_params["search_window_size"]
    if 'filter_strength' not in st.session_state:
        st.session_state.filter_strength = processing_params["filter_strength"]
    if 'show_full_processed' not in st.session_state:
        st.session_state.show_full_processed = display_options["show_full_processed"]

    return create_return_dict(image, processing_params, display_options, advanced_options)

def setup_image_source() -> Image.Image:
    st.markdown("### 📷 Image Source")
    image_source = st.radio("Choose Image Source", ["Preloaded Image", "Upload Image"])
    
    if image_source == "Preloaded Image":
        selected_image = st.selectbox("Select Image", list(PRELOADED_IMAGES))
        uploaded_file = None
    else:
        uploaded_file = st.file_uploader("Upload Image")
        selected_image = None

    image = load_image(image_source, selected_image, uploaded_file)
    st.image(image, "Input Image", use_column_width=True)
    return image

def setup_processing_parameters(image: Image.Image) -> Dict[str, Any]:
    with st.expander("⚙️ Processing Parameters", expanded=True):
        kernel_size = st.number_input('Kernel Size', min_value=3, max_value=21, value=7, step=2,
                                      help="Size of the kernel for image processing")
        
        max_pixels = (image.width - kernel_size + 1) * (image.height - kernel_size + 1)
        technique = st.session_state.get('technique', 'speckle')
        
        use_full_image = st.checkbox("Use Full Image for Search", value=False)
        search_window_size = get_search_window_size(use_full_image, kernel_size, image)
        
        filter_strength = st.number_input("Filter Strength (h)", 
                                          min_value=0.01, 
                                          max_value=30.0, 
                                          value=0.10, 
                                          step=0.01,
                                          format="%.2f",
                                          help="Strength of the NL-Means filter")
        
        cmap = st.selectbox("🎨 Color Map", COLOR_MAPS, index=0)

    return {
        "kernel_size": kernel_size,
        "max_pixels": max_pixels,
        "technique": technique,
        "search_window_size": search_window_size,
        "filter_strength": filter_strength,
        "cmap": cmap
    }

def setup_display_options(image: Image.Image, kernel_size: int) -> Dict[str, Any]:
    with st.expander("🖼️ Display Options", expanded=True):
        show_full_processed = st.checkbox(
            "Show Fully Processed Image", 
            value=True,
            help="Toggle to switch between progressive processing and full image processing."
        )
        
        if not show_full_processed:
            play_pause, reset = setup_animation_controls()
            pixels_to_process = setup_pixel_processing(image, kernel_size)
        else:
            play_pause, reset = False, False
            pixels_to_process = (image.width - kernel_size + 1) * (image.height - kernel_size + 1)

    return {
        "show_full_processed": show_full_processed,
        "play_pause": play_pause,
        "reset": reset,
        "pixels_to_process": pixels_to_process
    }

def setup_animation_controls() -> Tuple[bool, bool]:
    col1, col2 = st.columns(2)
    play_pause = col1.button("▶️/⏸️", use_container_width=True) 
    reset = col2.button("🔄 Reset", use_container_width=True)
    return play_pause, reset

def setup_pixel_processing(image: Image.Image, kernel_size: int) -> int:
    max_pixels = (image.width - kernel_size + 1) * (image.height - kernel_size + 1)
    pixels_to_process = st.slider(
        "Pixels to process (slider)",
        min_value=1,
        max_value=max_pixels,
        value=st.session_state.get('current_position', 1),
        step=1,
        key="pixels_slider",
        on_change=update_pixels,
        args=('slider',)
    )
    
    st.number_input(
        "Pixels to process (input)",
        min_value=1,
        max_value=max_pixels,
        value=st.session_state.get('current_position', 1),
        step=1,
        key="pixels_input",
        on_change=update_pixels,
        args=('input',)
    )
    
    st.session_state.current_position = pixels_to_process
    return pixels_to_process

def setup_advanced_options(image: Image.Image) -> Dict[str, Any]:
    with st.expander("🔬 Advanced Options"):
        add_noise = st.checkbox("Toggle Gaussian Noise")
        if add_noise:
            noise_mean = st.number_input("Noise Mean", 
                                         min_value=0.0, 
                                         max_value=1.0, 
                                         value=0.0, 
                                         step=0.01,
                                         format="%.2f")
            noise_std = st.number_input("Noise Std", 
                                        min_value=0.0, 
                                        max_value=1.0, 
                                        value=0.1, 
                                        step=0.01,
                                        format="%.2f")
            image_np = np.clip(np.array(image) / 255.0 + np.random.normal(noise_mean, noise_std, np.array(image).shape), 0, 1)
        else:
            image_np = np.array(image) / 255.0

    return {
        "add_noise": add_noise,
        "image_np": image_np
    }

def create_return_dict(image: Image.Image, processing_params: Dict[str, Any], 
                       display_options: Dict[str, Any], advanced_options: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "image": image,
        "image_np": advanced_options["image_np"],
        "show_full_processed": display_options["show_full_processed"],
        "animation_params": {
            "play_pause": display_options["play_pause"],
            "reset": display_options["reset"],
            "pixels_to_process": display_options["pixels_to_process"]  
        },
        "kernel_size": processing_params["kernel_size"],
        "search_window_size": processing_params["search_window_size"],
        "filter_strength": processing_params["filter_strength"],
        "cmap": processing_params["cmap"],
        "max_pixels": processing_params["max_pixels"] if display_options["show_full_processed"] else display_options["pixels_to_process"],
        "technique": processing_params["technique"]
    }
