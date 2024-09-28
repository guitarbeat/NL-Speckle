from dataclasses import dataclass, field
import numpy as np
import streamlit as st
from typing import Any, Dict, List, Tuple, Optional, Type, Union
from contextlib import contextmanager
from functools import partial

from src.classes import (ResultCombinationError, 
                         ImageArray, KernelVisualizationConfig, 
                         SearchWindowConfig, VisualizationConfig,
                         visualize_analysis_results)

from src.math.speckle import process_speckle_contrast, SpeckleResult
from src.math.nlm import process_non_local_means, NLMResult
from src.utils import validate_input, calculate_processing_end

###############################################################################
#                              Result Classes                                 #
###############################################################################

@dataclass
class ImageResult:
    nlm_result: NLMResult
    speckle_result: SpeckleResult
    additional_images: Dict[str, np.ndarray] = field(default_factory=dict)
    image_dimensions: Tuple[int, int] = field(default=(0, 0))

    @property
    def processing_end_coord(self) -> Tuple[int, int]:
        return max(self.nlm_result.processing_end_coord, self.speckle_result.processing_end_coord)

    @property
    def kernel_size(self) -> int:
        return self.nlm_result.kernel_size  # Assuming both results use the same kernel size

    @property
    def pixels_processed(self) -> int:
        return max(self.nlm_result.pixels_processed, self.speckle_result.pixels_processed)

    @property
    def nlm_search_window_size(self) -> int:
        return self.nlm_result.search_window_size

    @property
    def nlm_filter_strength(self) -> float:
        return self.nlm_result.filter_strength

    def add_image(self, name: str, image: np.ndarray) -> None:
        self.additional_images[name] = image

    @property
    def all_images(self) -> Dict[str, np.ndarray]:
        images = {
            f"NLM {k}": v for k, v in self.nlm_result.filter_data.items()
        }
        images.update({
            f"Speckle {k}": v for k, v in self.speckle_result.filter_data.items()
        })
        return {**images, **self.additional_images}

    @property
    def filter_options(self) -> List[str]:
        return (
            [f"NLM {option}" for option in self.nlm_result.get_filter_options()] +
            [f"Speckle {option}" for option in self.speckle_result.get_filter_options()] +
            list(self.additional_images.keys())
        )

    @property
    def filter_data(self) -> Dict[str, Any]:
        return self.all_images


###############################################################################
#                           Technique Execution                               #
###############################################################################

def process_and_visualize_image(technique: str, tab: st.delta_generator.DeltaGenerator) -> None:
    if "image_array" not in st.session_state or st.session_state.image_array is None:
        st.warning("No image has been uploaded. Please upload an image before processing.")
        return

    show_per_pixel_processing = st.session_state.get("show_per_pixel", False)
    ui_placeholders = create_technique_ui(technique, tab, show_per_pixel_processing)
    st.session_state[f"{technique}_placeholders"] = ui_placeholders

    try:
        # Process image if not already processed
        if f"{technique}_result" not in st.session_state or st.session_state[f"{technique}_result"] is None:
            with create_processing_status() as (status, progress_bar):
                result = process_technique(technique, status, progress_bar)
            
            if result is None:
                st.warning(f"{technique.upper()} processing failed. Please check the logs for details.")
                return
            
            st.session_state[f"{technique}_result"] = result

        # Visualize results
        visualize_results(technique, ui_placeholders, show_per_pixel_processing)

    except Exception as e:
        st.error(f"Error for {technique}: {str(e)}. Please check the logs for details.")
        import traceback
        st.error(f"Traceback: {traceback.format_exc()}")

def process_technique(technique: str, status, progress_bar) -> Optional[Union[SpeckleResult, NLMResult, ImageResult]]:
    image, kernel_size, pixel_count, _ = initialize_processing(
        st.session_state.image_array.astype(np.float32),
        st.session_state.kernel_size,
        st.session_state.exact_pixel_count,
        st.session_state.search_window_size,
        st.session_state.filter_strength
    )

    update_progress = partial(update_progress_wrapper, status, progress_bar)

    if technique == "speckle":
        return process_speckle_contrast(status, image, kernel_size, pixel_count, 
                                        partial(update_progress, "Speckle"))
    elif technique == "nlm":
        return process_non_local_means(status, image, kernel_size, pixel_count,
                                       st.session_state.search_window_size,
                                       st.session_state.filter_strength,
                                       partial(update_progress, "NLM"))
    elif technique == "nl_speckle":
        speckle_result = process_speckle_contrast(status, image, kernel_size, pixel_count, 
                                                  partial(update_progress, "Speckle"))
        nlm_result = process_non_local_means(status, image, kernel_size, pixel_count,
                                             st.session_state.search_window_size,
                                             st.session_state.filter_strength,
                                             partial(update_progress, "NLM"))
        
        if speckle_result is None or nlm_result is None:
            raise ValueError("Processing failed to produce a result")

        return combine_results(status, speckle_result, nlm_result, image.shape)
    
    raise ValueError(f"Unknown technique: {technique}")

def visualize_results(technique: str, ui_placeholders, show_per_pixel_processing: bool):
    results = st.session_state[f"{technique}_result"]
    image_array = st.session_state.image_array
    height, width = image_array.shape

    half_kernel = results.kernel_size // 2
    end_x, end_y = results.processing_end_coord

    end_x = min(max(end_x, half_kernel), width - half_kernel - 1)
    end_y = min(max(end_y, half_kernel), height - half_kernel - 1)

    kernel_matrix = get_kernel_matrix(image_array, end_x, end_y, half_kernel, results.kernel_size)

    viz_config = create_visualization_config(technique, results, image_array, ui_placeholders,
                                             show_per_pixel_processing, end_x, end_y, half_kernel,
                                             kernel_matrix)

    visualize_analysis_results(viz_config)

def create_technique_ui(technique: str, tab: st.delta_generator.DeltaGenerator, show_per_pixel_processing: bool):
    from main import DEFAULT_SPECKLE_VIEW, DEFAULT_NLM_VIEW

    with tab:
        ui_placeholders = {"formula": st.empty(), "original_image": st.empty()}
        filter_options = ["Original Image"] + (SpeckleResult.get_filter_options() if technique == "speckle" else NLMResult.get_filter_options())
        default_selection = DEFAULT_SPECKLE_VIEW if technique == "speckle" else DEFAULT_NLM_VIEW
        selected_filters = st.multiselect("Select views to display", filter_options, default=default_selection, key=f"{technique}_filter_selection")
        st.session_state[f"{technique}_selected_filters"] = selected_filters

        if selected_filters:
            columns = st.columns(len(selected_filters))
            for i, filter_name in enumerate(selected_filters):
                filter_key = filter_name.lower().replace(" ", "_")
                ui_placeholders[filter_key] = columns[i].empty()
                if show_per_pixel_processing:
                    ui_placeholders[f"zoomed_{filter_key}"] = columns[i].expander(f"Zoomed-in {filter_name}", expanded=False).empty()
        else:
            st.warning("No views selected. Please select at least one view to display.")

        if show_per_pixel_processing:
            ui_placeholders["zoomed_kernel"] = st.empty()

    return ui_placeholders


###############################################################################
#                           Image Processing                                  #
###############################################################################

def update_progress_wrapper(status, progress_bar, task: str, progress: float):
    status.update(label=f"{task}: {progress:.1%}")
    progress_bar.progress(0.5 * progress if task == "Speckle" else 0.5 + 0.5 * progress)

def log_error(e: Exception, image_array: np.ndarray):
    error_message = f"Error in process_image: {type(e).__name__}: {str(e)}"
    image_info = f"Image shape: {image_array.shape}, Image size: {image_array.size}, Image dtype: {image_array.dtype}"
    st.error(f"{error_message}\n{image_info}")

def initialize_processing(image: np.ndarray, kernel_size: int, pixel_count: int,
                          nlm_search_window_size: int, nlm_filter_strength: float) -> Tuple[np.ndarray, int, int, int]:
    validate_input(image, kernel_size, pixel_count, nlm_search_window_size, nlm_filter_strength)
    height, width = image.shape
    valid_pixels = (height - kernel_size + 1) * (width - kernel_size + 1)
    pixel_count = min(pixel_count, valid_pixels)
    return image, kernel_size, pixel_count, valid_pixels


@contextmanager 
def create_processing_status():
    with st.status("Processing image...", expanded=True) as status:
        progress_bar = st.progress(0)
        yield status, progress_bar
        
def combine_results(status, speckle_result: SpeckleResult, nlm_result: NLMResult,
                    image_dimensions: Tuple[int, int]) -> ImageResult:
    status.write("🔗 Merging Speckle Contrast and Non-Local Means results")
    
    final_result = ImageResult(
        nlm_result=nlm_result,
        speckle_result=speckle_result,
        image_dimensions=image_dimensions
    )

    status.write("🏁 Analysis complete: NL-Speckle result generated")
    return final_result

def get_kernel_matrix(image_array, end_x, end_y, half_kernel, kernel_size):
    height, width = image_array.shape
    kernel_matrix = image_array[
        max(0, end_y - half_kernel):min(height, end_y + half_kernel + 1),
        max(0, end_x - half_kernel):min(width, end_x + half_kernel + 1)
    ].copy()

    if kernel_matrix.shape != (kernel_size, kernel_size):
        kernel_matrix = np.pad(
            kernel_matrix,
            (
                (max(0, half_kernel - end_y), max(0, end_y + half_kernel + 1 - height)),
                (max(0, half_kernel - end_x), max(0, end_x + half_kernel + 1 - width))
            ),
            mode="constant", constant_values=0
        )

    return np.array(kernel_matrix)

def create_visualization_config(technique, results, image_array, ui_placeholders,
                                show_per_pixel_processing, end_x, end_y, half_kernel,
                                kernel_matrix):
    return VisualizationConfig(
        image_array=ImageArray(data=image_array),
        technique=technique,
        results=results,
        ui_placeholders=ui_placeholders,
        show_per_pixel_processing=show_per_pixel_processing,
        processing_end=(end_x, end_y),
        kernel=KernelVisualizationConfig(
            size=results.kernel_size,
            origin=(half_kernel, half_kernel),
            kernel_matrix=kernel_matrix,
        ),
        search_window=SearchWindowConfig(
            size=getattr(results, 'search_window_size', None) if technique == "nlm" else None,
            use_full_image=st.session_state.get("use_whole_image", False),
        ),
        last_processed_pixel=(end_x, end_y),
        pixels_to_process=st.session_state.get("exact_pixel_count", results.pixels_processed),
        original_pixel_value=float(image_array[end_y, end_x]),
    )