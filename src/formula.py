"""
Streamlit app for displaying Non-Local Means (NLM) denoising and Speckle
Contrast formulas. Provides interactive explanations and visualizations of the
mathematical concepts.
"""

import streamlit as st

NLM_FORMULA_CONFIG = {
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
                r"{kernel_matrix}"
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


SPECKLE_FORMULA_CONFIG = {
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
  - Valid region: [{half_kernel}, {image_height} - {half_kernel}) × [{half_kernel}, {image_width} - {half_kernel})
  - Valid region size: {valid_height} × {valid_width}

The patch is constructed by considering pixels within the range $[-{half_kernel}, {half_kernel}]$ relative to the current pixel position.
"""
        },

        {
            "title": "Neighborhood Analysis",
            "formula": r"\text{{Kernel Size: }} {kernel_size} \times {kernel_size}"
            r"\quad\quad\text{{Centered at pixel: }}({x}, {y})"
            r"\\\\"
            "{kernel_matrix}",
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

# ----------------------------- Formula Display Functions
# ----------------------------- #


def display_analysis_formula(
    specific_params,
    placeholders,
    analysis_type,
    end_x,
    end_y,
    kernel_size,
    kernel_matrix,
    original_value,
):
    """
    Display the analysis formula.

    Args:
        specific_params: A dictionary of specific parameters for the formula.
        placeholders: A dictionary of Streamlit placeholders. analysis_type: The
        type of analysis ('nlm' or 'speckle'). end_x: The x-coordinate of the
        end point. end_y: The y-coordinate of the end point. kernel_size: The
        size of the kernel. kernel_matrix: The kernel matrix. original_value:
        The original value.
    """
    variables = {
        "x": end_x,
        "y": end_y,
        "input_x": end_x,
        "input_y": end_y,
        "kernel_size": kernel_size,
        "kernel_matrix": kernel_matrix,
        "original_value": original_value,
        "total_pixels": kernel_size * kernel_size,
    }

    # Add default values for potentially missing keys
    default_values = {
        "nlm_value": 0.0,
        "std": 0.0,
        "mean": 0.0,
        "sc": 0.0,
    }

    variables |= default_values
    variables |= specific_params

    formula_config = (
        NLM_FORMULA_CONFIG if analysis_type == "nlm" else SPECKLE_FORMULA_CONFIG
    )
    if formula_placeholder := placeholders.get("formula"):
        display_formula(formula_config, variables, formula_placeholder)
    else:
        st.warning("Formula placeholder not found.")

# Prepares and adjusts variables for formula display based on the analysis type


def prepare_variables(kwargs, analysis_type):
    """
    Prepares and adjusts variables for formula display based on the analysis
    type.

    Args:
        kwargs: The input variables. analysis_type: The type of analysis ('nlm'
        or 'speckle').

    Returns:
        The prepared variables.
    """
    variables = kwargs.copy()
    kernel_size = variables.get("kernel_size", 3)

    if "input_x" not in variables or "input_y" not in variables:
        variables["input_x"] = variables["x"] - kernel_size // 2
        variables["input_y"] = variables["y"] - kernel_size // 2

    if "kernel_matrix" in variables:
        variables["kernel_matrix"] = generate_kernel_matrix(
            kernel_size, variables["kernel_matrix"]
        )

    if analysis_type == "nlm":
        variables["patch_size"] = kernel_size
        variables["h"] = variables.get("filter_strength", 1.0)
        search_window_size = variables.get("search_window_size")
        variables["search_window_description"] = (
            "We search the entire image for similar pixels."
            if search_window_size == "full"
            else f"A search window of size {search_window_size}x{search_window_size} centered around the target pixel."
        )
        variables["nlm_value"] = variables.get("nlm_value", 0.0)
    else:  # speckle
        variables["mean"] = variables.get("mean", 0.0)
        variables["std"] = variables.get("std", 0.0)
        variables["sc"] = variables.get("sc", 0.0)

    return variables

# Displays the main formula and additional formulas in an expandable section


def display_formula(config, variables, formula_placeholder):
    """
    Displays the main formula and additional formulas in an expandable section.

    Args:
        config: The formula configuration. variables: The variables for the
        formula. formula_placeholder: The Streamlit placeholder for the formula
        display.
    """
    with formula_placeholder.container():
        analysis_type = variables.get("analysis_type", "nlm")
        variables = prepare_variables(variables, analysis_type)
        with st.expander(f"{config['title']} Details"):
            display_formula_section(config, variables, "main")
            display_additional_formulas(config, variables)

# Displays a specific section of the formula (main or additional)


def display_formula_section(config, variables, section_key):
    """
    Displays a specific section of the formula (main or additional).

    Args:
        config: The formula configuration. variables: The variables for the
        formula. section_key: The key for the section to display ('main' or
        'additional').
    """
    formula_key = "formula" if section_key == "formula" else f"{
        section_key}_formula"
    explanation_key = "explanation"

    try:
        # Ensure the formula is a string Convert to string if necessary
        st.latex(str(config[formula_key]).format(**variables))
        st.markdown(config[explanation_key].format(**variables))
    except KeyError as e:
        st.error(f"Missing key in {section_key} formula or explanation: {e}")
    except ValueError as e:
        st.error(f"Formatting error in {section_key} formula: {e}")

# Displays additional formulas in separate tabs

def display_additional_formulas(config, variables):
    """
    Displays additional formulas in separate tabs.

    Args:
        config: The formula configuration. variables: The variables for the
        formula.
    """
    st.write("Additional Formulas:")
    tab_labels = [formula["title"]
                  for formula in config["additional_formulas"]]
    tabs = st.tabs(tab_labels)
    for tab, additional_formula in zip(tabs, config["additional_formulas"]):
        with tab:
            display_formula_section(additional_formula, variables, "formula")

# Generates a LaTeX representation of the kernel matrix


def generate_kernel_matrix(kernel_size, kernel_matrix):
    """
    Generates a LaTeX representation of the kernel matrix.

    Args:
        kernel_size: The size of the kernel. kernel_matrix: The kernel matrix.

    Returns:
        The LaTeX representation of the kernel matrix.
    """
    center = kernel_size // 2
    # center_value = kernel_matrix[center][center]
    center_value = kernel_matrix[center, center]

    matrix_rows = [
        " & ".join(
            (
                rf"\mathbf{{{center_value:.3f}}}"
                if i == center and j == center
                # else r"{:.3f}".format(kernel_matrix[i][j])
                else f"{kernel_matrix[i, j]:.3f}"
            )
            for j in range(kernel_size)
        )
        for i in range(kernel_size)
    ]

    return (
        r"\def\arraystretch{1.5}\begin{array}{|"
        + ":".join(["c"] * kernel_size)
        + "|}"
        + r"\hline"
        + r"\\ \hdashline".join(matrix_rows)
        + r"\\ \hline\end{array}"
    )
