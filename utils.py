import numpy as np
from typing import Dict, Optional
from typing import Any, List
import streamlit as st
import matplotlib.pyplot as plt
from visualization import  create_combined_plot

def visualize_image(
    image: np.ndarray,
    placeholder: Any,
    x: int,
    y: int,
    kernel_size: int,
    cmap: str,
    show_full: bool,
    vmin: float,
    vmax: float,
    title: str,
    technique: str = None,
    search_window_size: int = None,
    zoom: bool = False
):
    if show_full and not zoom:
        fig = plt.figure()
        plt.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax)
        plt.axis('off')
        plt.title(title)
    else:
        if zoom:
            zoom_size = kernel_size
            ky = int(max(0, y - zoom_size // 2))
            kx = int(max(0, x - zoom_size // 2))
            image = image[ky:min(image.shape[0], ky + zoom_size),
                          kx:min(image.shape[1], kx + zoom_size)]
            x, y = zoom_size // 2, zoom_size // 2
        
        fig = create_combined_plot(image, x, y, kernel_size, title, cmap, 
                                   search_window_size if technique == "nlm" else None, 
                                   zoom=zoom, vmin=vmin, vmax=vmax)
    
    placeholder.pyplot(fig)


def generate_kernel_matrix(kernel_size: int, kernel_matrix: List[List[float]]) -> str:
    center = kernel_size // 2
    center_value = kernel_matrix[center][center]
    
    matrix_rows = [
        " & ".join(
            r"\mathbf{{{:.3f}}}".format(center_value) if i == center and j == center 
            else r"{:.3f}".format(kernel_matrix[i][j])
            for j in range(kernel_size)
        )
        for i in range(kernel_size)
    ]

    return (r"\def\arraystretch{1.5}\begin{array}{|" + ":".join(["c"] * kernel_size) + "|}" +
            r"\hline" + r"\\ \hdashline".join(matrix_rows) + r"\\ \hline\end{array}")

def display_formula_section(config: Dict[str, Any], variables: Dict[str, Any], section_key: str):
    formula_key = 'formula' if section_key == 'formula' else f'{section_key}_formula'
    explanation_key = 'explanation'
    
    try:
        st.latex(config[formula_key].format(**variables))
        st.markdown(config[explanation_key].format(**variables))
    except KeyError as e:
        st.error(f"Missing key in {section_key} formula or explanation: {e}")

def display_additional_formulas(config: Dict[str, Any], variables: Dict[str, Any]):
    with st.expander("Additional Formulas", expanded=False):
        for additional_formula in config['additional_formulas']:
            with st.expander(additional_formula['title'], expanded=False):
                display_formula_section(additional_formula, variables, 'formula')

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
