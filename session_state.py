"""
This module manages all session state operations for the Streamlit application.
"""

import streamlit as st
from typing import Dict, Any, List, Union, Tuple, Optional
import numpy as np
import logging
import traceback

# Constants
SPECKLE_CONTRAST = "Speckle Contrast"
ORIGINAL_IMAGE = "Original Image"
NON_LOCAL_MEANS = "Non-Local Means"
DEFAULT_SPECKLE_VIEW = [ORIGINAL_IMAGE, SPECKLE_CONTRAST]
DEFAULT_NLM_VIEW = [ORIGINAL_IMAGE, NON_LOCAL_MEANS]

def initialize_session_state():
    defaults: Dict[str, Any] = {
        "image": None,
        "image_array": None,
        "kernel_size": 3,
        "show_per_pixel": False,
        "color_map": "gray",
        "normalization": {
            "option": "None",
            "percentile_low": 2,
            "percentile_high": 98,
        },
        "gaussian_noise": {
            "enabled": False,
            "mean": 0.1,
            "std_dev": 0.1,
        },
        "use_sat": False,
        "nlm_options": {
            "use_whole_image": True,
            "filter_strength": 10.0,
            "search_window_size": 50,
        },
        "pixel_processing": {
            "percentage": 100,
            "exact_count": 0,
        },
        "filters": {
            "nlm": DEFAULT_NLM_VIEW,
            "speckle": DEFAULT_SPECKLE_VIEW,
        },
        "techniques": ["speckle", "nlm"],
        "tabs": None,
        "viz_config": None,
        "filter_selections": {
            "speckle": DEFAULT_SPECKLE_VIEW,
            "nlm": DEFAULT_NLM_VIEW,
        },
        "speckle_result": None,
        "nlm_result": None,
        "pixels_to_process": 0,
        "desired_exact_count": 0,
        "processed_image_np": None,
        "image_file": None,
        "apply_gaussian_noise": False,
        "kernel_size_slider": 3,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def update_pixel_processing_values():
    if st.session_state.image:
        total_pixels = calculate_total_pixels()
        st.session_state.desired_exact_count = total_pixels
        st.session_state.pixels_to_process = total_pixels

def calculate_total_pixels() -> int:
    if st.session_state.image:
        width, height = st.session_state.image.width, st.session_state.image.height
        kernel = st.session_state.kernel_size
        return max((width - kernel + 1) * (height - kernel + 1), 0)
    return 0

def get_filter_options(technique: str) -> List[str]:
    options = {
        "speckle": ["Original Image", "Mean Filter", "Std Dev Filter", "Speckle Contrast"],
        "nlm": ["Original Image", "Non-Local Means", "Normalization Factors",
                "Last Similarity Map", "Non-Local Standard Deviation", "Non-Local Speckle"]
    }
    return options.get(technique, ["Original Image"])

def get_default_view(technique: str) -> List[str]:
    return DEFAULT_SPECKLE_VIEW if technique == "speckle" else DEFAULT_NLM_VIEW

def update_filter_selection(technique: str, selection: List[str]):
    if 'filter_selections' not in st.session_state:
        st.session_state.filter_selections = {}
    st.session_state.filter_selections[technique] = selection

def get_filter_selection(technique: str) -> List[str]:
    return st.session_state.filter_selections.get(technique, get_default_view(technique))

def set_technique_result(technique: str, result: Any):
    st.session_state[f"{technique}_result"] = result

def get_technique_result(technique: str) -> Any:
    return st.session_state.get(f"{technique}_result")

def get_search_window_size() -> int:
    return st.session_state.get('search_window_size', 50)

def get_color_map() -> str:
    return st.session_state.get('color_map', 'gray')

def get_image_array() -> np.ndarray:
    return st.session_state.get('image_array', np.array([]))

def set_viz_config(config: Dict[str, Any]):
    st.session_state.viz_config = config

def get_show_per_pixel_processing() -> bool:
    return st.session_state.get('show_per_pixel', False)

def get_use_whole_image() -> bool:
    return st.session_state.get("use_whole_image", True)

def set_use_whole_image(value: bool):
    st.session_state['use_whole_image'] = value

def get_nlm_options() -> Dict[str, Any]:
    return st.session_state.get('nlm_options', {
        "use_whole_image": True,
        "filter_strength": 10.0,
        "search_window_size": 50,
    })

def get_default_filter_strength() -> float:
    return st.session_state.get("default_filter_strength", 10.0)

def get_default_search_window_size() -> int:
    return st.session_state.get("default_search_window_size", 50)

def safe_get(dict_obj: Dict[str, Any], *keys, default: Any = None) -> Any:
    for key in keys:
        if isinstance(dict_obj, dict) and key in dict_obj:
            dict_obj = dict_obj[key]
        else:
            return default
    return dict_obj

def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (ValueError, TypeError):
        return default

def safe_array_value(array: Union[np.ndarray, None], y: int, x: int, width: int) -> float:
    if array is None or array.size == 0:
        return 0.0
    try:
        if array.ndim == 2:
            return float(array[y, x])
        elif array.ndim == 1:
            index = y * width + x
            return float(array[index]) if 0 <= index < array.size else 0.0
        else:
            return 0.0
    except IndexError:
        return 0.0

def get_image_dimensions() -> Tuple[int, int]:
    image_array = st.session_state.get('image_array')
    if isinstance(image_array, np.ndarray):
        return image_array.shape[:2]
    return (0, 0)

def get_kernel_size() -> int:
    return safe_int(safe_get(st.session_state, 'kernel', 'size'), 3)

def get_analysis_type() -> str:
    return safe_get(st.session_state, 'technique', default="unknown")

def prepare_comparison_images() -> Optional[Dict[str, np.ndarray]]:
    comparison_images: Dict[str, np.ndarray] = {
        "Unprocessed Image": st.session_state.get("image_array", np.array([]))
    }

    for result_key in ["speckle_result", "nlm_result"]:
        results = st.session_state.get(result_key)
        if results:
            if isinstance(results, dict) and "filter_data" in results:
                comparison_images.update(results["filter_data"])
            elif isinstance(results, tuple) and len(results) > 0:
                comparison_images[result_key] = results[0]
            elif hasattr(results, "filter_data"):
                comparison_images.update(results.filter_data)
            else:
                st.warning(f"Unexpected format for {result_key}. Skipping this result.")

    return comparison_images if len(comparison_images) > 1 else None

def get_technique_params(technique: str) -> Dict[str, Any]:
    nlm_opts = st.session_state.get('nlm_options', {})
    return {
        'kernel_size': safe_int(st.session_state.get('kernel_size', 3)),
        'pixel_count': safe_int(st.session_state.get('pixels_to_process', -1)),
        'search_window_size': safe_int(nlm_opts.get('search_window_size', 50)),
        'filter_strength': float(nlm_opts.get('filter_strength', 10.0)),
        'use_full_image': bool(nlm_opts.get('use_whole_image', True))
    }

def update_technique_params(params: Dict[str, Any]):
    for key, value in params.items():
        st.session_state[key] = value

def log_error(message: str):
    logging.exception(message)

def handle_processing_error(error_message: str) -> None:
    full_error = f"An error occurred during processing:\n\n{error_message}\n\nFull traceback:\n{traceback.format_exc()}"
    st.error(full_error)
    log_error(full_error)

def get_gaussian_noise_settings() -> Dict[str, Any]:
    return st.session_state.get('gaussian_noise', {
        "enabled": False,
        "mean": 0.1,
        "std_dev": 0.1,
    })

def set_gaussian_noise_settings(enabled: bool, mean: float, std_dev: float):
    st.session_state['gaussian_noise'] = {
        "enabled": enabled,
        "mean": mean,
        "std_dev": std_dev,
    }

def get_processed_image_np() -> np.ndarray:
    return st.session_state.get('processed_image_np', np.array([]))

def set_processed_image_np(image: np.ndarray):
    st.session_state['processed_image_np'] = image

def get_image_file() -> str:
    return st.session_state.get('image_file', '')

def set_image_file(filename: str):
    st.session_state['image_file'] = filename