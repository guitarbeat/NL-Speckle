import streamlit as st
from analysis.nlm import NLM_FORMULA_CONFIG
from analysis.speckle import SPECKLE_FORMULA_CONFIG

FULL_SEARCH = 'full'

# ----------------------------- Formula Display Functions ----------------------------- #

def display_analysis_formula(specific_params, placeholders, analysis_type, end_x, end_y, kernel_size, kernel_matrix, original_value):
    """
    Display the analysis formula.
    
    Args:
        specific_params: A dictionary of specific parameters for the formula.
        placeholders: A dictionary of Streamlit placeholders.
        analysis_type: The type of analysis ('nlm' or 'speckle').
        end_x: The x-coordinate of the end point.
        end_y: The y-coordinate of the end point.
        kernel_size: The size of the kernel.
        kernel_matrix: The kernel matrix.
        original_value: The original value.
    """
    specific_params |= {
        'x': end_x,
        'y': end_y,
        'input_x': end_x,
        'input_y': end_y,
        'kernel_size': kernel_size,
        'kernel_matrix': kernel_matrix,
        'original_value': original_value,
    }

    formula_config = NLM_FORMULA_CONFIG if analysis_type == 'nlm' else SPECKLE_FORMULA_CONFIG
    if formula_placeholder := placeholders.get('formula'):
        display_formula(formula_config, specific_params, formula_placeholder)
    else:
        st.warning("Formula placeholder not found.")

# Prepares and adjusts variables for formula display based on the analysis type
def prepare_variables(kwargs, analysis_type):
    """
    Prepares and adjusts variables for formula display based on the analysis type.
    
    Args:
        kwargs: The input variables.
        analysis_type: The type of analysis ('nlm' or 'speckle').
    
    Returns:
        The prepared variables.
    """
    variables = kwargs.copy()
    kernel_size = variables.get('kernel_size', 3)
    
    if 'input_x' not in variables or 'input_y' not in variables:
        variables['input_x'] = variables['x'] - kernel_size // 2
        variables['input_y'] = variables['y'] - kernel_size // 2
    
    if 'kernel_matrix' in variables:
        variables['kernel_matrix'] = generate_kernel_matrix(kernel_size, variables['kernel_matrix'])
    
    if analysis_type == 'nlm':
        variables['patch_size'] = kernel_size
        variables['h'] = variables.get('filter_strength', 1.0)
        search_window_size = variables.get('search_window_size')
        variables['search_window_description'] = (
            "We search the entire image for similar pixels." if search_window_size == FULL_SEARCH
            else f"A search window of size {search_window_size}x{search_window_size} centered around the target pixel."
        )
    
    return variables

# Displays the main formula and additional formulas in an expandable section
def display_formula(config, variables, formula_placeholder):
    """
    Displays the main formula and additional formulas in an expandable section.
    
    Args:
        config: The formula configuration.
        variables: The variables for the formula.
        formula_placeholder: The Streamlit placeholder for the formula display.
    """
    with formula_placeholder.container():
        analysis_type = variables.get('analysis_type', 'nlm')
        variables = prepare_variables(variables, analysis_type)
        with st.expander(f"{config['title']} Details"):
            display_formula_section(config, variables, 'main')
            display_additional_formulas(config, variables)

# Displays a specific section of the formula (main or additional)
def display_formula_section(config, variables, section_key):
    """
    Displays a specific section of the formula (main or additional).
    
    Args:
        config: The formula configuration.
        variables: The variables for the formula.
        section_key: The key for the section to display ('main' or 'additional').
    """
    formula_key = 'formula' if section_key == 'formula' else f'{section_key}_formula'
    explanation_key = 'explanation'
    
    try:
        st.latex(config[formula_key].format(**variables))
        st.markdown(config[explanation_key].format(**variables))
    except KeyError as e:
        st.error(f"Missing key in {section_key} formula or explanation: {e}")

# Displays additional formulas in separate tabs
def display_additional_formulas(config, variables):
    """
    Displays additional formulas in separate tabs.
    
    Args:
        config: The formula configuration.
        variables: The variables for the formula.
    """
    st.write("Additional Formulas:")
    tab_labels = [formula['title'] for formula in config['additional_formulas']]
    tabs = st.tabs(tab_labels)
    for tab, additional_formula in zip(tabs, config['additional_formulas']):
        with tab:
            display_formula_section(additional_formula, variables, 'formula')

# Generates a LaTeX representation of the kernel matrix
def generate_kernel_matrix(kernel_size, kernel_matrix):
    """
    Generates a LaTeX representation of the kernel matrix.
    
    Args:
        kernel_size: The size of the kernel.
        kernel_matrix: The kernel matrix.
    
    Returns:
        The LaTeX representation of the kernel matrix.
    """
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
