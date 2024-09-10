import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from typing import Any, Dict, List, Optional, Union
from matplotlib.collections import LineCollection

from analysis.speckle import SPECKLE_FORMULA_CONFIG
from analysis.nlm import NLM_FORMULA_CONFIG

# ----------------------------- Plot Creation ----------------------------- #
def draw_kernel_overlay(ax: plt.Axes, x: int, y: int, kernel_size: int):
    kx, ky = int(x - kernel_size // 2), int(y - kernel_size // 2)
    ax.add_patch(plt.Rectangle((kx - 0.5, ky - 0.5), kernel_size, kernel_size, 
                            edgecolor="r", linewidth=1, facecolor="none"))
    lines = ([[(kx + i - 0.5, ky - 0.5), (kx + i - 0.5, ky + kernel_size - 0.5)] for i in range(1, kernel_size)] +
            [[(kx - 0.5, ky + i - 0.5), (kx + kernel_size - 0.5, ky + i - 0.5)] for i in range(1, kernel_size)])
    ax.add_collection(LineCollection(lines, colors='red', linestyles=':', linewidths=0.5))

def draw_search_window_overlay(ax: plt.Axes, image: np.ndarray, x: int, y: int, search_window: Optional[Union[str, int]]):
    if search_window == "full":
        rect = plt.Rectangle((-0.5, -0.5), image.shape[1], image.shape[0], 
                            edgecolor="blue", linewidth=2, facecolor="none")
        ax.add_patch(rect)
    elif isinstance(search_window, int):
        half_window = search_window // 2
        rect = plt.Rectangle((x - half_window - 0.5, y - half_window - 0.5), 
                            search_window, search_window, 
                            edgecolor="blue", linewidth=1, facecolor="none")
        ax.add_patch(rect)

def draw_value_annotations(ax: plt.Axes, image: np.ndarray):
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            ax.text(j, i, f"{image[i, j]:.2f}", ha="center", va="center", color="red", fontsize=8)

@st.cache_data(persist=True)
def create_combined_plot(plot_image: np.ndarray, plot_x: int, plot_y: int, plot_kernel_size: int, 
                         title: str, plot_cmap: str = "viridis", plot_search_window: Optional[Union[str, int]] = None, 
                         zoom: bool = False, vmin: Optional[float] = None, vmax: Optional[float] = None) -> plt.Figure:
    fig, ax = plt.subplots(1, 1)

    ax.imshow(plot_image, cmap=plot_cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.axis('off')

    ax.add_patch(plt.Rectangle((plot_x - 0.5, plot_y - 0.5), 1, 1, 
                                edgecolor="r", linewidth=0.5, facecolor="r", alpha=0.2))
    draw_kernel_overlay(ax, plot_x, plot_y, plot_kernel_size) 
    draw_search_window_overlay(ax, plot_image, plot_x, plot_y, plot_search_window)
    
    if zoom:
        draw_value_annotations(ax, plot_image)
    fig.tight_layout(pad=2)
    return fig

# ----------------------------- Data Preparation for Visualization ----------------------------- #

def prepare_comparison_images():
    speckle_results = st.session_state.get("speckle_results")
    nlm_results = st.session_state.get("nlm_results")
    analysis_params = st.session_state.analysis_params

    if speckle_results is not None and nlm_results is not None:
        return {
            'Unprocessed Image': analysis_params['image_np'],
            'Standard Deviation': speckle_results['std_dev_filter'],
            'Speckle Contrast': speckle_results['speckle_contrast_filter'],
            'Mean Filter': speckle_results['mean_filter'],
            'NL-Means Image': nlm_results['processed_image']
        }
    else:
        return None

# ---------------------------- Formula Display ---------------------------- #

def generate_kernel_matrix(kernel_size: int, kernel_matrix: List[List[float]]) -> str:
    center = kernel_size // 2
    center_value = kernel_matrix[center][center]
    
    matrix_rows = []
    for i in range(kernel_size):
        row = [r"\mathbf{{{:.3f}}}".format(center_value) if i == center and j == center 
               else r"{:.3f}".format(kernel_matrix[i][j]) for j in range(kernel_size)]
        matrix_rows.append(" & ".join(row))

    return (r"\def\arraystretch{1.5}\begin{array}{|" + ":".join(["c"] * kernel_size) + "|}" +
            r"\hline" + r"\\ \hdashline".join(matrix_rows) + r"\\ \hline\end{array}")

def display_formula(formula_placeholder: Any, technique: str, **kwargs):
    with formula_placeholder.container():
        if technique == "speckle":
            config = SPECKLE_FORMULA_CONFIG
        elif technique == "nlm":
            config = NLM_FORMULA_CONFIG
        else:
            st.error(f"Unknown technique: {technique}")
            return

        variables = kwargs.copy()
        
        # Only calculate input_x and input_y if they're not provided
        if 'input_x' not in variables or 'input_y' not in variables:
            kernel_size = variables.get('kernel_size', 3)  # Default to 3 if not provided
            variables['input_x'] = variables['x'] - kernel_size // 2
            variables['input_y'] = variables['y'] - kernel_size // 2

        if 'kernel_size' in variables and 'kernel_matrix' in variables:
            variables['kernel_matrix'] = generate_kernel_matrix(variables['kernel_size'], variables['kernel_matrix'])

        if technique == "nlm":
            search_size = variables.get('search_size')
            variables['search_window_description'] = (
                "We search the entire image for similar pixels." if search_size == "full" 
                else f"A search window of size {search_size}x{search_size} centered around the target pixel."
            )
        display_main_formula(config, variables)
        display_additional_formulas(config, variables)

def display_main_formula(config: Dict[str, Any], variables: Dict[str, Any]):
    try:
        st.latex(config['main_formula'].format(**variables))
        st.markdown(config['explanation'].format(**variables))
    except KeyError as e:
        st.error(f"Missing key in main formula or explanation: {e}")

def display_additional_formulas(config: Dict[str, Any], variables: Dict[str, Any]):
    with st.expander("Additional Formulas", expanded=False):
        for additional_formula in config['additional_formulas']:
            with st.expander(additional_formula['title'], expanded=False):
                try:
                    st.latex(additional_formula['formula'].format(**variables))
                    st.markdown(additional_formula['explanation'].format(**variables))
                except KeyError as e:
                    st.error(f"Missing key in additional formula: {e}")