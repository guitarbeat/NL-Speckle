import streamlit as st
from PIL import Image
import numpy as np
import time
from typing import Dict, Any, Optional, Tuple
from image_processing import handle_image_comparison, create_placeholders_and_sections
import streamlit_nested_layout # type: ignore  # noqa: F401

# For updating the image processing technique
from image_processing import process_image, display_formula, create_combined_plot, FORMULA_CONFIG
import matplotlib.pyplot as plt


# Constants
TABS = ["Speckle Contrast Calculation", "Non-Local Means Denoising", "Speckle Contrast Comparison"]
PAGE_CONFIG = {
    "page_title": "Speckle Contrast Visualization",
    "layout": "wide",
    "page_icon": "favicon.png",
    "initial_sidebar_state": "expanded"
}
PRELOADED_IMAGES = {
    "image50.png": "media/image50.png",
    "spatial.tif": "media/spatial.tif",
    "logo.jpg": "media/logo.jpg"
}
COLOR_MAPS = ["viridis", "plasma", "inferno", "magma", "cividis", "gray", "pink"]

@st.cache_data
def load_image(image_source: str, selected_image: str = None, uploaded_file: Any = None) -> Image.Image:
    if image_source == "Preloaded Image" and selected_image:
        return Image.open(PRELOADED_IMAGES[selected_image]).convert('L')
    elif image_source == "Upload Image" and uploaded_file:
        return Image.open(uploaded_file).convert('L')
    st.warning('Please upload or select an image.')
    st.stop()



def calculate_max_processable_pixels(image_width, image_height, kernel_size):
    return (image_width - kernel_size + 1) * (image_height - kernel_size + 1)

def setup_sidebar() -> Dict[str, Any]:
    # Initialize current_position if it doesn't exist
    if 'current_position' not in st.session_state:
        st.session_state.current_position = 1

    def update_pixels(key):
        if key == 'slider':
            st.session_state.pixels_input = st.session_state.pixels_slider
        else:
            st.session_state.pixels_slider = st.session_state.pixels_input
        st.session_state.current_position = st.session_state.pixels_slider
        update_images_analyze_and_visualize(
            image_np=st.session_state.image_np,
            kernel_size=st.session_state.kernel_size,
            cmap=st.session_state.cmap,
            technique=st.session_state.technique,
            search_window_size=st.session_state.search_window_size,
            filter_strength=st.session_state.filter_strength,
            show_full_processed=st.session_state.show_full_processed,
            update_state=True,
            handle_visualization=True
        )

    with st.sidebar:
        st.title("Image Processing Settings")

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

        with st.expander("⚙️ Processing Parameters", expanded=True):
            kernel_size = st.number_input('Kernel Size', min_value=3, max_value=21, value=7, step=2,
                                          help="Size of the kernel for image processing")
            
            # Calculate max processable pixels
            max_pixels = calculate_max_processable_pixels(image.width, image.height, kernel_size)
            
            # based on the active tab, set the technique
            technique = st.session_state.get('technique', 'speckle')
            
            use_full_image = st.checkbox("Use Full Image for Search", value=False)
            if not use_full_image:
                search_window_size = st.number_input("Search Window Size", 
                                                     min_value=kernel_size + 2, 
                                                     max_value=min(max(image.width, image.height) // 2, 35),
                                                     value=kernel_size + 2,
                                                     step=2,
                                                     help="Size of the search window for NL-Means denoising")
            else:
                search_window_size = None  # Changed from "full" to None
            
            filter_strength = st.number_input("Filter Strength (h)", 
                                              min_value=0.01, 
                                              max_value=30.0, 
                                              value=0.10, 
                                              step=0.01,
                                              format="%.2f",
                                              help="Strength of the NL-Means filter")
            
            cmap = st.selectbox("🎨 Color Map", COLOR_MAPS, index=0)

        with st.expander("🖼️ Display Options", expanded=True):
            show_full_processed = st.checkbox(
                "Show Fully Processed Image", 
                value=True,
                help="Toggle to switch between progressive processing and full image processing."
            )
            
            if not show_full_processed:
                col1, col2 = st.columns(2)
                play_pause = col1.button("▶️/⏸️", use_container_width=True) 
                reset = col2.button("🔄 Reset", use_container_width=True)
                
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
            else:
                play_pause, reset, pixels_to_process = False, False, max_pixels

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

    # Store values in session state for use in update_pixels function
    st.session_state.image_np = image_np
    st.session_state.kernel_size = kernel_size
    st.session_state.cmap = cmap
    st.session_state.technique = technique
    st.session_state.search_window_size = search_window_size
    st.session_state.filter_strength = filter_strength
    st.session_state.show_full_processed = show_full_processed

    return {
        "image": image,
        "image_np": image_np,
        "show_full_processed": show_full_processed,
        "animation_params": {
            "play_pause": play_pause,
            "reset": reset,
            "pixels_to_process": pixels_to_process  
        },
        "kernel_size": kernel_size,
        "search_window_size": search_window_size,
        "filter_strength": filter_strength,
        "cmap": cmap,
        "max_pixels": max_pixels if show_full_processed else pixels_to_process,
        "technique": technique
    }

def handle_animation(
    animation_params: Dict[str, Any], 
    max_processable_pixels: int, 
    update_images_analyze_and_visualize: callable,
    image_np: np.ndarray,
    kernel_size: int,
    cmap: str,
    technique: str,
    search_window_size: Optional[int] = None,
    filter_strength: float = 0.1,
    show_full_processed: bool = False
):
    if animation_params['play_pause']:
        st.session_state.animate = not st.session_state.get('animate', False)
    
    if animation_params['reset']:
        st.session_state.current_position = 1
        st.session_state.animate = False

    if st.session_state.get('animate', False):
        for i in range(st.session_state.current_position, max_processable_pixels + 1):
            st.session_state.current_position = i
            update_images_analyze_and_visualize(
                image_np=image_np,
                kernel_size=kernel_size,
                cmap=cmap,
                technique=technique,
                search_window_size=search_window_size,
                filter_strength=filter_strength,
                show_full_processed=show_full_processed,
                update_state=True,
                handle_visualization=True
            )
            time.sleep(0.01)
            if not st.session_state.get('animate', False):
                break




def update_images_analyze_and_visualize(
    image_np: np.ndarray,
    kernel_size: int,
    cmap: str,
    technique: str = "speckle",
    search_window_size: Optional[int] = None,
    filter_strength: float = 0.1,
    show_full_processed: bool = False,
    update_state: bool = True,
    handle_visualization: bool = True,
    height: Optional[int] = None,
    width: Optional[int] = None
) -> Tuple[Dict[str, Any], Optional[Tuple[np.ndarray, ...]]]:
    
    if update_state:
        time.sleep(0.1)  # Add a small delay
        st.session_state.current_position = st.session_state.get('current_position', 1)
    
    # Prepare parameters for image analysis
    params = {
        "tabs": st.session_state.get('tabs', []),
        "analysis_params": {
            "image_np": image_np,
            "kernel_size": kernel_size,
            "max_pixels": st.session_state.current_position if update_state else st.session_state.get('current_position', 1),
            "cmap": cmap,
            "search_window_size": search_window_size,
            "filter_strength": filter_strength
        },
        "show_full_processed": show_full_processed
    }

    placeholders = {
        "speckle": st.session_state.get('speckle_placeholders', {}),
        "nlm": st.session_state.get('nlm_placeholders', {})
    }

    results = None
    
    if handle_visualization:
        try:
            # Initial setup and calculations
            height = height or image_np.shape[0]
            width = width or image_np.shape[1]
            
            valid_height = height - kernel_size + 1
            valid_width = width - kernel_size + 1
            total_valid_pixels = valid_height * valid_width
            
            pixels_to_process = min(params['analysis_params']['max_pixels'], total_valid_pixels)
            
            # Image processing
            results = process_image(technique, image_np, kernel_size, pixels_to_process, height, width, 
                                    search_window_size, filter_strength)
            
            # Calculate the coordinates of the last processed pixel
            last_pixel = pixels_to_process - 1
            last_y = (last_pixel // valid_width) + kernel_size // 2
            last_x = (last_pixel % valid_width) + kernel_size // 2
            
            half_kernel = kernel_size // 2
            last_x, last_y = int(last_x), int(last_y)

            # Extract kernel values
            y_start = max(0, last_y - half_kernel)
            y_end = min(height, last_y + half_kernel + 1)
            x_start = max(0, last_x - half_kernel)
            x_end = min(width, last_x + half_kernel + 1)

            kernel_values = image_np[y_start:y_end, x_start:x_end]
            
            if kernel_values.size == 0:
                raise ValueError(f"Extracted kernel at ({last_x}, {last_y}) is empty. Image shape: {image_np.shape}, Kernel size: {kernel_size}")

            # Pad the kernel if necessary
            if kernel_values.shape != (kernel_size, kernel_size):
                pad_top = max(0, half_kernel - last_y)
                pad_bottom = max(0, last_y + half_kernel + 1 - height)
                pad_left = max(0, half_kernel - last_x)
                pad_right = max(0, last_x + half_kernel + 1 - width)
                kernel_values = np.pad(kernel_values, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='edge')

            kernel_matrix = [[float(kernel_values[i, j]) for j in range(kernel_size)] for i in range(kernel_size)]
            original_value = float(image_np[last_y, last_x])

            # Visualization logic
            vmin, vmax = np.min(image_np), np.max(image_np)

            # Display original image
            if show_full_processed:
                fig_original = plt.figure()
                plt.imshow(image_np, cmap=cmap, vmin=vmin, vmax=vmax)
                plt.axis('off')
                plt.title("Original Image")
            else:
                fig_original = create_combined_plot(image_np, last_x, last_y, kernel_size, "Original Image with Current Kernel", cmap, 
                                                    search_window_size if technique == "nlm" else None, vmin=vmin, vmax=vmax)
            
            placeholders[technique]['original_image'].pyplot(fig_original)

            if not show_full_processed:
                # Create zoomed view of original image
                zoom_size = kernel_size
                ky = int(max(0, last_y - zoom_size // 2))
                kx = int(max(0, last_x - zoom_size // 2))
                zoomed_original = image_np[ky:min(height, ky + zoom_size),
                                           kx:min(width, kx + zoom_size)]
                fig_zoom_original = create_combined_plot(zoomed_original, zoom_size // 2, zoom_size // 2, zoom_size, 
                                                       "Zoomed-In Original Image", cmap, zoom=True, vmin=vmin, vmax=vmax)
                if 'zoomed_original_image' in placeholders[technique] and placeholders[technique]['zoomed_original_image'] is not None:
                    placeholders[technique]['zoomed_original_image'].pyplot(fig_zoom_original)

            # Process and display results based on technique
            if technique == "speckle":
                filter_options = {
                    "Mean Filter": results[0],
                    "Std Dev Filter": results[1],
                    "Speckle Contrast": results[2]
                }
                specific_params = {
                    "std": results[5],
                    "mean": results[6],
                    "sc": results[7],
                    "total_pixels": pixels_to_process
                }
            elif technique == "nlm":
                denoised_image, weight_sum_map = results[:2]
                filter_options = {
                    "NL-Means Image": denoised_image,
                    "Weight Map": weight_sum_map,
                    "Difference Map": np.abs(image_np - denoised_image)
                }
                specific_params = {
                    "filter_strength": filter_strength,
                    "search_size": search_window_size,
                    "total_pixels": pixels_to_process,
                    "nlm_value": denoised_image[last_y, last_x]
                }
            else:
                filter_options = {}
                specific_params = {}

            # Display filter results
            for filter_name, filter_data in filter_options.items():
                key = filter_name.lower().replace(" ", "_")
                if key in placeholders[technique] and placeholders[technique][key] is not None:
                    filter_vmin, filter_vmax = np.min(filter_data), np.max(filter_data)
                    
                    if show_full_processed:
                        fig_full = plt.figure()
                        plt.imshow(filter_data, cmap=cmap, vmin=filter_vmin, vmax=filter_vmax)
                        plt.axis('off')
                        plt.title(filter_name)
                    else:
                        fig_full = create_combined_plot(filter_data, last_x, last_y, kernel_size, filter_name, cmap, vmin=filter_vmin, vmax=filter_vmax)
                    placeholders[technique][key].pyplot(fig_full)

                    if not show_full_processed:
                        zoomed_data = filter_data[ky:min(filter_data.shape[0], ky + zoom_size),
                                                  kx:min(filter_data.shape[1], kx + zoom_size)]
                        fig_zoom = create_combined_plot(zoomed_data, zoom_size // 2, zoom_size // 2, zoom_size, 
                                                        f"Zoomed-In {filter_name}", cmap, zoom=True, vmin=filter_vmin, vmax=filter_vmax)
                        
                        zoomed_key = f'zoomed_{key}'
                        if zoomed_key in placeholders[technique] and placeholders[technique][zoomed_key] is not None:
                            placeholders[technique][zoomed_key].pyplot(fig_zoom)

            plt.close('all')

            # Display formula
            display_formula(placeholders[technique]['formula'], technique, FORMULA_CONFIG,
                            x=last_x, y=last_y, 
                            input_x=last_x, input_y=last_y,
                            kernel_size=kernel_size,
                            kernel_matrix=kernel_matrix,
                            original_value=original_value,
                            **specific_params)

        except (ValueError, IndexError) as e:
            st.error(f"Error during image analysis and visualization: {str(e)}")
            return None

    # Update session state with results
    if update_state:
        st.session_state.processed_pixels = params['analysis_params']['max_pixels']
        if technique == "speckle":
            st.session_state.speckle_results = results
        elif technique == "nlm":
            st.session_state.nlm_results = results

    return params, results

def main():
    st.set_page_config(**PAGE_CONFIG)
    st.logo("media/logo.png")

    sidebar_params = setup_sidebar()
    tabs = st.tabs(TABS)
    
    speckle_placeholders = create_placeholders_and_sections("speckle", tabs[0], sidebar_params['show_full_processed'])
    nlm_placeholders = create_placeholders_and_sections("nlm", tabs[1], sidebar_params['show_full_processed'])

    analysis_params = {
        "image_np": sidebar_params['image_np'],
        "kernel_size": sidebar_params['kernel_size'],
        "search_window_size": sidebar_params['search_window_size'],
        "filter_strength": sidebar_params['filter_strength'],
        "cmap": sidebar_params['cmap'],
        "max_pixels": sidebar_params['max_pixels'],
        "height": sidebar_params['image_np'].shape[0],
        "width": sidebar_params['image_np'].shape[1]
    }

    # Store necessary variables in session state
    st.session_state.tabs = tabs
    st.session_state.speckle_placeholders = speckle_placeholders
    st.session_state.nlm_placeholders = nlm_placeholders
    st.session_state.sidebar_params = sidebar_params
    st.session_state.analysis_params = analysis_params

    if not sidebar_params['show_full_processed']:
        handle_animation(
            animation_params=sidebar_params['animation_params'],
            max_processable_pixels=sidebar_params['max_pixels'],
            update_images_analyze_and_visualize=update_images_analyze_and_visualize,
            image_np=sidebar_params['image_np'],
            kernel_size=sidebar_params['kernel_size'],
            cmap=sidebar_params['cmap'],
            technique=sidebar_params['technique'],
            search_window_size=sidebar_params['search_window_size'],
            filter_strength=sidebar_params['filter_strength'],
            show_full_processed=sidebar_params['show_full_processed']
        )

    # Process images for both techniques
    speckle_params, speckle_results = update_images_analyze_and_visualize(
        image_np=sidebar_params['image_np'],
        kernel_size=sidebar_params['kernel_size'],
        cmap=sidebar_params['cmap'],
        technique="speckle",
        search_window_size=sidebar_params['search_window_size'],
        filter_strength=sidebar_params['filter_strength'],
        show_full_processed=sidebar_params['show_full_processed'],
        update_state=True,
        handle_visualization=True
    )
    
    nlm_params, nlm_results = update_images_analyze_and_visualize(
        image_np=sidebar_params['image_np'],
        kernel_size=sidebar_params['kernel_size'],
        cmap=sidebar_params['cmap'],
        technique="nlm",
        search_window_size=sidebar_params['search_window_size'],
        filter_strength=sidebar_params['filter_strength'],
        show_full_processed=sidebar_params['show_full_processed'],
        update_state=True,
        handle_visualization=True
    )

    # Store results in session state
    st.session_state.speckle_results = speckle_results
    st.session_state.nlm_results = nlm_results

    # Handle image comparison
    if speckle_results is not None and nlm_results is not None:
        handle_image_comparison(tab=tabs[2], cmap_name=sidebar_params['cmap'], images={
            'Unprocessed Image': sidebar_params['image_np'],
            'Standard Deviation': speckle_results[1],
            'Speckle Contrast': speckle_results[2],
            'Mean Filter': speckle_results[0],
            'NL-Means Image': nlm_results[0]
        })
    else:
        with tabs[2]:
            st.warning("Please process the image before viewing comparisons.")

    # Add a section to display processing statistics
    with st.expander("Processing Statistics", expanded=False):
        st.write(f"Image size: {analysis_params['width']}x{analysis_params['height']}")
        st.write(f"Kernel size: {analysis_params['kernel_size']}x{analysis_params['kernel_size']}")
        st.write(f"Max processable pixels: {sidebar_params['max_pixels']}")
        st.write(f"Actual processed pixels: {min(sidebar_params['max_pixels'], (analysis_params['width'] - analysis_params['kernel_size'] + 1) * (analysis_params['height'] - analysis_params['kernel_size'] + 1))}")
        st.write(f"Processing technique: {sidebar_params['technique']}")


if __name__ == "__main__":
    main()