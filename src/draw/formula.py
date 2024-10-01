"""
Streamlit app for displaying Non-Local Means (NLM) denoising and Speckle
Contrast formulas. Provides interactive explanations and visualizations of the
mathematical concepts.
"""

import numpy as np
from typing import Dict, Any
import streamlit as st
from src.session_state import (
    get_value, get_image_dimensions, kernel_size as get_kernel_size,
    get_use_whole_image,DEFAULT_SEARCH_WINDOW_SIZE,DEFAULT_FILTER_STRENGTH
)

# Constants
NLM_FORMULA_CONFIG: Dict[str, Any] = {
    "title": "Non-Local Means (NLM) Denoising",
    "main_formula": (
        r"I_{{{x},{y}}} = {original_value:.3f} \quad \rightarrow \quad "
        r"NLM_{{{x},{y}}} = \frac{{1}}{{C_{{{x},{y}}}}} "
        r"\sum_{{(i,j) \in \Omega_{{{x},{y}}}}} "
        r"I_{{i,j}} \cdot w_{{{x},{y}}}(i,j) = {nlm_value:.0f}"
    ),
    "explanation": r"""
    The Non-Local Means (NLM) algorithm denoises each pixel by replacing it with a 
    weighted average of pixels from a search window. The weights are based on the 
    similarity of patches around each pixel:

    1. For each pixel $(x,y)$, compare its patch $P_{{{x},{y}}}$ to patches $P_{{i,j}}$ 
       around other pixels $(i,j)$ in the search window $\Omega_{{{x},{y}}}$.
    2. Calculate a weight $w_{{{x},{y}}}(i,j)$ for each comparison based on patch similarity.
    3. Compute the NLM value $NLM_{{{x},{y}}}$ as a weighted average of intensities $I_{{i,j}}$
       using these weights, normalized by $C_{{{x},{y}}}$.

    This process replaces the original intensity $I_{{{x},{y}}}$ with the NLM value $NLM_{{{x},{y}}}$.
    """,
    "additional_formulas": [
        {
            "title": "Border Handling",
            "formula": r"P_{{x,y}}(i,j) = \begin{{cases}} I_{{x+i,y+j}} & \text{{if }} (x+i,y+j) \in \text{{valid region}} \\ 0 & \text{{otherwise}} \end{{cases}} \quad \text{{for }} i,j \in [-{half_kernel}, {half_kernel}]",
            "explanation": r"""
To avoid boundary issues, the algorithm only processes pixels within the valid region, which excludes pixels near the image border.

For a pixel at position $(x,y)$:
- The patch $P_{{x,y}}$ is defined using the kernel size {kernel_size} × {kernel_size}
- The valid processing region is determined by the kernel size:
  - Valid region: [{half_kernel}, {image_height} - {half_kernel}) × [{half_kernel}, {image_width} - {half_kernel})
  - Valid region size: {valid_height} × {valid_width}

The patch is constructed by considering pixels within the range $[-{half_kernel}, {half_kernel}]$ relative to the current pixel position.
"""
        },
        {
            "title": "Patch Analysis",
            "formula": (
                r"\quad\quad\text{{Patch }} P_{{{x},{y}}} \text{{ centered at: }}({{x}},{{y}})"
                r"\\\\"
                r"{kernel_matrix_latex}"
            ),
            "explanation": (
                r"The ${kernel_size} \times {kernel_size}$ patch $P_{{x,y}}$ centered at $({{x}}, {{y}})$ "
                r"is compared to other patches $P_{{i,j}}$ in the search window. The matrix shows "
                r"pixel values, with the **central value** being the pixel to be denoised."
            ),
        },
        {
            "title": "Weight Calculation",
            "formula": r"w_{{{x},{y}}}(i,j) = e^{{\displaystyle -\frac{{|P_{{x,y}} - P_{{i,j}}|^2}}{{h^2}}}} = e^{{\displaystyle -\frac{{|P_{{{x},{y}}} - P_{{i,j}}|^2}}{{{filter_strength:.0f}^2}}}}",
            "explanation": r"""
The weight $w(x,y,i,j)$ for pixel $(i,j)$ when denoising $(x,y)$ is calculated using a Gaussian function:

- $P_{{x,y}}$, $P_{{i,j}}$: Patches centered at $(x,y)$ and $(i,j)$
- $|P_{{x,y}} - P_{{i,j}}|^2$: Sum of squared differences between patches
- $h = {h:.0f}$: Filtering parameter controlling the decay of the weights
  - Larger $h$ includes more dissimilar patches, leading to stronger smoothing
  - Smaller $h$ restricts to very similar patches, preserving more details

Similar patches yield higher weights, while dissimilar patches are assigned lower weights.
""",
        },
        {
            "title": "Normalization Factor",
            "formula": r"{{C_{{{x},{y}}}}} = \sum_{{i,j \in \Omega(x,y)}} w_{{{x},{y}}}(i,j)",
            "explanation": r"Sum of all weights for pixel $(x,y)$, denoted as $C(x,y)$, ensuring the final weighted average preserves overall image brightness.",
        },
        {
            "title": "Search Window",
            "formula": r"\Omega(x,y) = \begin{{cases}} I & \text{{if search\_size = 'full'}} \\ [(x-s,y-s), (x+s,y+s)] \cap \text{{valid region}} & \text{{otherwise}} \end{{cases}}",
            "explanation": r"The search window $\Omega(x,y)$ defines the neighborhood around pixel $(x,y)$ in which similar patches are searched for. $I$ denotes the full image and $s$ is {search_window_size} pixels. {search_window_description}",
        },
        {
            "title": "NLM Calculation",
            "formula": r"NLM_{{{x},{y}}} = \frac{{1}}{{C_{{{x},{y}}}}} "
            r"\sum_{{i,j \in \Omega_{{{x},{y}}}}}"
            r"I_{{i,j}} \cdot w_{{{x},{y}}}(i,j) = {nlm_value:.3f}",
            "explanation": r"Final NLM value for pixel $(x,y)$: weighted average of pixel intensities $I_{{i,j}}$ in the search window, normalized by the sum of weights $C(x,y)$.",
        },
    ],
}

SPECKLE_FORMULA_CONFIG: Dict[str, Any] = {
    "title": "Speckle Contrast Calculation",
    "main_formula": r"I_{{{x},{y}}} = {original_value:.3f} \quad \rightarrow \quad SC_{{{x},{y}}} = \frac{{\sigma_{{{x},{y}}}}}{{\mu_{{{x},{y}}}}} = \frac{{{std:.3f}}}{{{mean:.3f}}} = {sc:.3f}",
    "explanation": r"This formula shows the transition from the original pixel intensity $I_{{{x},{y}}}$ to the Speckle Contrast (SC) for the same pixel position.",
    "additional_formulas": [

        {
            "title": "Border Handling",
            "formula": r"P_{{x,y}}(i,j) = \begin{{cases}} I_{{x+i,y+j}} & \text{{if }} (x+i,y+j) \in \text{{valid region}} \\ 0 & \text{{otherwise}} \end{{cases}} \quad \text{{for }} i,j \in [-{half_kernel}, {half_kernel}]",
            "explanation": r"""
To avoid boundary issues, the algorithm only processes pixels within the valid region, which excludes pixels near the image border.

For a pixel at position $(x,y)$:
- The patch $P_{{x,y}}$ is defined using the kernel size {kernel_size} × {kernel_size}
- The valid processing region is determined by the kernel size:
  - Valid region: [{half_kernel}, {image_height} - {half_kernel})  [{half_kernel}, {image_width} - {half_kernel})
  - Valid region size: {valid_height} × {valid_width}

The patch is constructed by considering pixels within the range $[-{half_kernel}, {half_kernel}]$ relative to the current pixel position.
"""
        },

        {
            "title": "Neighborhood Analysis",
            "formula": r"\text{{Kernel Size: }} {kernel_size} \times {kernel_size}"
            r"\quad\quad\text{{Centered at pixel: }}({x}, {y})"
            r"\\\\"
            "{kernel_matrix_latex}",
            "explanation": r"Analysis of a ${kernel_size}\times{kernel_size}$ neighborhood centered at pixel $({x},{y})$. The matrix shows pixel values, with the central value (in bold) being the processed pixel.",
        },

        {
            "title": "Mean Filter",
            "formula": r"\mu_{{{x},{y}}} = \frac{{1}}{{N}} \sum_{{i,j \in K_{{{x},{y}}}}} I_{{i,j}} = \frac{{1}}{{{kernel_size}^2}} \sum_{{i,j \in K_{{{x},{y}}}}} I_{{i,j}} = {mean:.3f}",
            "explanation": r"The mean intensity $\mu_{{{x},{y}}}$ at pixel $({x},{y})$ is calculated by summing the intensities $I_{{i,j}}$ of all pixels within the kernel $K$ centered at $({x},{y})$, and dividing by the total number of pixels $N = {kernel_size}^2 = {total_pixels}$ in the kernel.",
        },

        {
            "title": "Standard Deviation Calculation",
            "formula": r"\sigma_{{{x},{y}}} = \sqrt{{\frac{{1}}{{N}} \sum_{{i,j \in K_{{{x},{y}}}}} (I_{{i,j}} - \mu_{{{x},{y}}})^2}} = \sqrt{{\frac{{1}}{{{kernel_size}^2}} \sum_{{i,j \in K_{{{x},{y}}}}} (I_{{i,j}} - {mean:.3f})^2}} = {std:.3f}",
            "explanation": r"The standard deviation $\sigma_{{{x},{y}}}$ at pixel $({x},{y})$ measures the spread of intensities around the mean $\mu_{{{x},{y}}}$. It is calculated by taking the square root of the average squared difference between each pixel intensity $I_{{i,j}}$ in the kernel $K$ and the mean intensity $\mu_{{{x},{y}}}$.",
        },

        {
            "title": "Speckle Contrast Calculation",
            "formula": r"SC_{{{x},{y}}} = \frac{{\sigma_{{{x},{y}}}}}{{\mu_{{{x},{y}}}}} = \frac{{{std:.3f}}}{{{mean:.3f}}} = {sc:.3f}",
            "explanation": r"Speckle Contrast (SC): ratio of standard deviation $\sigma_{{{x},{y}}}$ to mean intensity $\mu_{{{x},{y}}}$ within the kernel centered at $({x},{y})$.",
        },
    ],
}

def get_formula_config(technique: str) -> Dict[str, Any]:
    try:
        return NLM_FORMULA_CONFIG if technique == "nlm" else SPECKLE_FORMULA_CONFIG
    except Exception as e:
        st.error(f"Error in get_formula_config: {str(e)}")
        return {}

def create_specific_params(last_x: int, last_y: int) -> Dict[str, Any]:
    """Create specific parameters for formula display."""
    try:
        height, width = get_image_dimensions()
        current_kernel_size = get_kernel_size()
        half_kernel = current_kernel_size // 2

        specific_params = {
            "kernel_size": current_kernel_size,
            "pixels_processed": get_value('pixels_processed', 0),
            "x": last_x,
            "y": last_y,
            "image_height": height,
            "image_width": width,
            "half_kernel": half_kernel,
            "original_value": get_value('original_pixel_value', 0.0),
            "analysis_type": get_value('analysis_type', ''),
            "total_pixels": current_kernel_size ** 2,
            "valid_height": max(0, height - current_kernel_size + 1),
            "valid_width": max(0, width - current_kernel_size + 1),
            "search_window_size": get_value('search_window_size', "N/A"),
            "kernel_matrix": get_value('kernel_matrix', np.zeros((current_kernel_size, current_kernel_size))),
            # Add default values for all possible parameters
            "filter_strength": get_value('filter_strength', DEFAULT_FILTER_STRENGTH),
            "nlm_value": get_value('nlm_value', 0.0),
            "mean": get_value('mean', 0.0),
            "std": get_value('std', 0.0),
            "sc": get_value('sc', 0.0),
        }

        # Update specific parameters based on analysis type
        if specific_params["analysis_type"] == "nlm":
            specific_params.update({
                "search_window_size": get_value('search_window_size', DEFAULT_SEARCH_WINDOW_SIZE),
            })

        return specific_params
    except Exception as e:
        st.error(f"Error in create_specific_params: {str(e)}")
        return {}

def prepare_variables(variables: Dict[str, Any]) -> Dict[str, Any]:
    """Prepare variables for formula display."""
    try:
        prepared = variables.copy()
        kernel_size = prepared["kernel_size"]
        half_kernel = prepared["half_kernel"]

        # Handle kernel matrix
        kernel_matrix = prepared["kernel_matrix"]
        if isinstance(kernel_matrix, np.ndarray) and kernel_matrix.size > 0:
            prepared["kernel_matrix_latex"] = generate_kernel_matrix(
                kernel_matrix, kernel_size, 
                prepared["x"] - half_kernel, 
                prepared["y"] - half_kernel
            )
        else:
            prepared["kernel_matrix_latex"] = "Kernel matrix not available"

        # Analysis-specific preparations
        if prepared["analysis_type"] == "nlm":
            prepared["search_window_description"] = (
                "We search the entire image for similar pixels."
                if prepared["search_window_size"] == "full" or get_use_whole_image()
                else f"A search window of size {prepared['search_window_size']}x{prepared['search_window_size']} centered around the target pixel."
            )

        # Ensure all required keys have a value
        required_keys = [
            "original_value", "total_pixels", "valid_height", "valid_width",
            "pixels_processed", "analysis_type"
        ]
        for key in required_keys:
            if key not in prepared:
                prepared[key] = "N/A"

        return prepared
    except Exception as e:
        st.error(f"Error in prepare_variables: {str(e)}")
        return variables

def display_formula(config: Dict[str, Any], variables: Dict[str, Any], formula_placeholder: st.delta_generator.DeltaGenerator) -> None:
    """Display the formula and its details in a Streamlit container."""
    try:
        with formula_placeholder.container():
            with st.expander(f"{config['title']} Details"):
                display_formula_section(config, variables, "main")
                display_additional_formulas(config, variables)
    except Exception as e:
        st.error(f"Error in display_formula: {str(e)}")

def display_formula_section(config: Dict[str, Any], variables: Dict[str, Any], section_key: str) -> None:
    """Display a specific section of the formula (main or additional)."""
    formula_key = "formula" if section_key == "formula" else f"{section_key}_formula"
    try:
        formula_str = str(config[formula_key])
        # Use a defaultdict to provide a default value for missing keys
        from collections import defaultdict
        variables_with_defaults = defaultdict(lambda: "N/A", variables)
        formatted_formula = formula_str.format_map(variables_with_defaults)
        st.latex(formatted_formula)
        st.markdown(config["explanation"].format_map(variables_with_defaults))
    except Exception as e:
        st.error(f"Unexpected error in display_formula_section: {str(e)}")
        st.error(f"Formula: {formula_str}")
        st.error(f"Variables: {variables}")

def display_additional_formulas(config: Dict[str, Any], variables: Dict[str, Any]) -> None:
    """Display additional formulas in tabs."""
    try:
        st.write("Additional Formulas:")
        tabs = st.tabs([formula["title"] for formula in config["additional_formulas"]])
        for tab, additional_formula in zip(tabs, config["additional_formulas"]):
            with tab:
                display_formula_section(additional_formula, variables, "formula")
    except Exception as e:
        st.error(f"Error in display_additional_formulas: {str(e)}")

def generate_kernel_matrix(kernel_matrix: np.ndarray, kernel_size: int, x: int, y: int) -> str:
    """Generate a LaTeX representation of the kernel matrix."""
    try:
        center = kernel_size // 2
        matrix_rows = [
            " & ".join(
                rf"\mathbf{{{kernel_matrix[i, j]:.3f}}}" if i == center and j == center 
                else f"{kernel_matrix[i, j]:.3f}"
                for j in range(kernel_size)
            ) for i in range(kernel_size)
        ]
        return (r"\def\arraystretch{1.5}\begin{array}{|" + ":".join(["c"] * kernel_size) + "|}" + 
                r"\hline" + r"\\ \hdashline".join(matrix_rows) + r"\\ \hline\end{array}")
    except Exception as e:
        st.error(f"Error generating kernel matrix: {str(e)}")
        return "Error generating kernel matrix"

def display_formula_details(viz_params: Dict[str, Any]) -> None:
    """Main function to display formula details based on visualization parameters."""
    try:
        last_processed_pixel = viz_params.get('last_processed_pixel', (0, 0))
        last_x, last_y = last_processed_pixel

        specific_params = create_specific_params(last_x, last_y)
        specific_params = prepare_variables(specific_params)

        formula_config = get_formula_config(viz_params['technique'])
        
        formula_placeholder = viz_params['ui_placeholders'].get("formula")
        if formula_placeholder:
            display_formula(formula_config, specific_params, formula_placeholder)
        else:
            st.error("Formula placeholder not found.")
    except Exception as e:
        st.error(f"Error in display_formula_details: {str(e)}")