import numpy as np
from typing import Dict, List, Tuple, Optional
import streamlit as st
import matplotlib.pyplot as plt
from streamlit_image_comparison import image_comparison


# ----------------------------- Image Comparison ----------------------------- #


def handle_image_comparison(tab: st.delta_generator.DeltaGenerator, cmap_name: str, images: Dict[str, np.ndarray]):
    with tab:
        st.header("Image Comparison")
        if not images:
            st.warning("No images available for comparison.")
            return
        
        available_images = list(images.keys())
        col1, col2 = st.columns(2)
        image_choice_1 = col1.selectbox('Select first image to compare:', [''] + available_images, index=0)
        image_choice_2 = col2.selectbox('Select second image to compare:', [''] + available_images, index=0)
        
        if image_choice_1 and image_choice_2:
            img1, img2 = images[image_choice_1], images[image_choice_2]
            img1_uint8, img2_uint8 = map(lambda img: (plt.get_cmap(cmap_name)((img - np.min(img)) / (np.max(img) - np.min(img)))[:, :, :3] * 255).astype(np.uint8), [img1, img2])
            
            if image_choice_1 != image_choice_2:
                image_comparison(img1=img1_uint8, img2=img2_uint8, label1=image_choice_1, label2=image_choice_2, make_responsive=True)
                st.subheader("Selected Images")
                st.image([img1_uint8, img2_uint8], caption=[image_choice_1, image_choice_2])
            else:
                st.error("Please select two different images for comparison.")
                st.image(np.abs(img1 - img2), caption="Difference Map", use_column_width=True)
        else:
            st.info("Select two images to compare.")

# ----------------------------- Pixel Calculations ----------------------------- #

def calculate_processing_details(image: np.ndarray, kernel_size: int, max_pixels: Optional[int]) -> Dict[str, int]:
    """
    Calculate processing details for kernel-based image processing algorithms.

    Args:
    image (np.ndarray): Input image.
    kernel_size (int): Size of the kernel (assumed to be square).
    max_pixels (Optional[int]): Maximum number of pixels to process. If None, process all valid pixels.

    Returns:
    Dict[str, int]: A dictionary containing processing details.
    """
    height, width = image.shape
    half_kernel = kernel_size // 2
    valid_height = height - kernel_size + 1
    valid_width = width - kernel_size + 1
    total_valid_pixels = valid_height * valid_width
    pixels_to_process = total_valid_pixels if max_pixels is None else min(max_pixels, total_valid_pixels)

    first_x = first_y = half_kernel
    
    last_pixel = pixels_to_process - 1
    last_y = (last_pixel // valid_width) + half_kernel
    last_x = (last_pixel % valid_width) + half_kernel

    return {
        'height': height,
        'width': width,
        'first_x': first_x,
        'first_y': first_y,
        'last_x': int(last_x),
        'last_y': int(last_y),
        'pixels_to_process': pixels_to_process,
        'valid_height': valid_height,
        'valid_width': valid_width
    }

# ----------------------------- Kernel Extraction ----------------------------- #

def extract_kernel_info(image_np: np.ndarray, last_x: int, last_y: int, kernel_size: int) -> Tuple[List[List[float]], float]:
    """
    Extract and prepare kernel information from the image.

    Args:
    image_np (np.ndarray): The input image as a NumPy array.
    last_x (int): The x-coordinate of the center pixel.
    last_y (int): The y-coordinate of the center pixel.
    kernel_size (int): The size of the kernel.

    Returns:
    Tuple[List[List[float]], float]: A tuple containing the kernel matrix and the original center pixel value.
    """
    half_kernel = kernel_size // 2
    height, width = image_np.shape

    # Calculate kernel boundaries
    y_start = max(0, last_y - half_kernel)
    y_end = min(height, last_y + half_kernel + 1)
    x_start = max(0, last_x - half_kernel)
    x_end = min(width, last_x + half_kernel + 1)

    # Extract kernel values
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

    # Convert to list of lists and ensure float type
    kernel_matrix = [[float(kernel_values[i, j]) for j in range(kernel_size)] for i in range(kernel_size)]
    
    # Get the original center pixel value
    original_value = float(image_np[last_y, last_x])

    return kernel_matrix, original_value