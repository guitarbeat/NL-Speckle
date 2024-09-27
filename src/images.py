import os
import numpy as np
import streamlit as st
from typing import Tuple, Optional
from contextlib import contextmanager
from src.classes import NLSpeckleResult, get_checkpoint_path
from src.process import process_speckle, process_nlm
import concurrent.futures

@contextmanager
def _processing_status():
    with st.status("Processing image...", expanded=True) as status:
        progress_bar = st.progress(0)
        yield status, progress_bar

def _update_progress(status, progress_bar, current: int, total: int):
    progress = current / total
    status.update(label=f"Processing: {progress:.1%} complete")
    progress_bar.progress(progress)

def process_nl_speckle(image: Optional[np.ndarray], kernel_size: int, pixels_to_process: int, 
                       nlm_search_window_size: int = 21, nlm_h: float = 10.0) -> Optional[NLSpeckleResult]:
    try:
        if image is None:
            st.warning("No image has been uploaded. Please upload an image before processing.")
            return None

        image = image.astype(np.float32)
        height, width = image.shape
        valid_pixels = (height - kernel_size + 1) * (width - kernel_size + 1)
        pixels_to_process = min(pixels_to_process, valid_pixels)

        checkpoint_file = get_checkpoint_path(
            os.path.splitext(st.session_state.image_file)[0],
            kernel_size, pixels_to_process, nlm_search_window_size, nlm_h
        )

        if os.path.exists(checkpoint_file):
            loaded_result = NLSpeckleResult.load_checkpoint(checkpoint_file)
            return loaded_result.add_image("Loaded From", np.array([checkpoint_file]))

        with _processing_status() as (status, progress_bar):
            with concurrent.futures.ThreadPoolExecutor() as executor:
                speckle_future = executor.submit(process_speckle, image, kernel_size, pixels_to_process)
                nlm_future = executor.submit(process_nlm, image, kernel_size, pixels_to_process, 
                                             nlm_search_window_size, nlm_h)
                
                speckle_result = speckle_future.result()
                _update_progress(status, progress_bar, pixels_to_process // 2, pixels_to_process)
                
                nlm_result = nlm_future.result()
                _update_progress(status, progress_bar, pixels_to_process, pixels_to_process)

            if speckle_result is None or nlm_result is None:
                raise ValueError("Processing failed to produce a result")

            final_result = NLSpeckleResult(
                nlm_result=nlm_result,
                speckle_result=speckle_result,
                additional_images={},
                processing_end_coord=_calculate_processing_end(width, height, kernel_size, pixels_to_process),
                kernel_size=kernel_size,
                pixels_processed=pixels_to_process,
                image_dimensions=(height, width),
                nlm_search_window_size=nlm_search_window_size,
                nlm_h=nlm_h
            )

            try:
                final_result.save_checkpoint(checkpoint_file)
                status.update(label="Processing complete!", state="complete")
            except Exception as e:
                st.error(f"Error in saving checkpoint: {str(e)}")
                status.update(label="Processing complete, but failed to save checkpoint.", state="error")

        return final_result
    except Exception as e:
        error_message = f"Error in process_nl_speckle: {type(e).__name__}: {str(e)}"
        image_info = f"Image shape: {image.shape}, Image size: {image.size}, Image dtype: {image.dtype}"
        st.error(f"{error_message}\n{image_info}")
        return None

def _calculate_processing_end(width: int, height: int, kernel_size: int, pixels_to_process: int) -> Tuple[int, int]:
    valid_width = width - kernel_size + 1
    end_y, end_x = divmod(pixels_to_process - 1, valid_width)
    return (min(end_x + kernel_size // 2, width - 1), 
            min(end_y + kernel_size // 2, height - 1))