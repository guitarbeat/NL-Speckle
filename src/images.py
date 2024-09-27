import os
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any

import numpy as np
import dill
import streamlit as st

from src.nlm import process_nlm, NLMResult
from src.speckle import process_speckle, SpeckleResult

os.makedirs("checkpoints", exist_ok=True)

@dataclass
class NLSpeckleResult:
    nlm_result: NLMResult
    speckle_result: SpeckleResult
    additional_images: Dict[str, np.ndarray] = field(default_factory=dict)
    processing_end_coord: Tuple[int, int] = field(default=(0, 0))
    kernel_size: int = field(default=0)
    pixels_processed: int = field(default=0)
    image_dimensions: Tuple[int, int] = field(default=(0, 0))
    nlm_search_window_size: int = field(default=0)
    nlm_h: float = field(default=0.0)

    def add_image(self, name: str, image: np.ndarray):
        self.additional_images[name] = image

    def get_all_images(self) -> Dict[str, np.ndarray]:
        images = {}
        for prefix, result in [("NLM", self.nlm_result), ("Speckle", self.speckle_result)]:
            images.update({f"{prefix} {k}": v for k, v in result.get_filter_data().items()})
        return {**images, **self.additional_images}

    def get_filter_options(self) -> List[str]:
        return [f"{prefix} {option}" for prefix, result in [("NLM", self.nlm_result), ("Speckle", self.speckle_result)]
                for option in result.get_filter_options()] + list(self.additional_images.keys())

    def get_filter_data(self) -> Dict[str, Any]:
        return self.get_all_images()

    def get_last_processed_coordinates(self) -> Tuple[int, int]:
        return self.processing_end_coord

    def save_checkpoint(self, filename: str):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'wb') as f:
            # dill.dump(self, f)
            dill.dump(self, f, protocol=dill.HIGHEST_PROTOCOL)

    @classmethod
    def load_checkpoint(cls, filename: str) -> 'NLSpeckleResult':
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Checkpoint file not found: {filename}")
        with open(filename, 'rb') as f:
            return dill.load(f)

    @classmethod
    def combine(cls, nlm_results: List[NLMResult], speckle_results: List[SpeckleResult], 
                kernel_size: int, pixels_processed: int, image_dimensions: Tuple[int, int],
                nlm_search_window_size: int, nlm_h: float) -> 'NLSpeckleResult':
        return cls(
            nlm_result=NLMResult.combine(nlm_results),
            speckle_result=SpeckleResult.combine(speckle_results),
            processing_end_coord=(0, 0),
            kernel_size=kernel_size,
            pixels_processed=pixels_processed,
            image_dimensions=image_dimensions,
            nlm_search_window_size=nlm_search_window_size,
            nlm_h=nlm_h
        )
    

def process_nl_speckle(
    image: np.ndarray,
    kernel_size: int,
    pixels_to_process: int,
    nlm_search_window_size: int = 21,
    nlm_h: float = 10.0
) -> NLSpeckleResult:
    try:
        image = image.astype(np.float32)
        height, width = image.shape
        valid_height, valid_width = height - kernel_size + 1, width - kernel_size + 1
        pixels_to_process = min(pixels_to_process, valid_height * valid_width)

        # Create a unique filename based on parameters
        image_name = os.path.splitext(st.session_state.image_file)[0]
        checkpoint_filename = f"k{kernel_size}_p{pixels_to_process}_w{nlm_search_window_size}_h{nlm_h}.joblib"
        checkpoint_dir = os.path.join("checkpoints", image_name)
        final_file = os.path.join(checkpoint_dir, checkpoint_filename)

        os.makedirs(checkpoint_dir, exist_ok=True)

        if os.path.exists(final_file):
            loaded_result = NLSpeckleResult.load_checkpoint(final_file)
            st.info(f"Loading previously calculated results from: {final_file}")
            loaded_result.add_image("Loaded From", np.array([final_file]))
            return loaded_result

        with st.status("Processing image...", expanded=True) as status:
            progress_bar = st.progress(0)

            speckle_result = process_speckle(image, kernel_size, pixels_to_process)
            update_progress(status, progress_bar, pixels_to_process // 2, pixels_to_process)

            nlm_result = process_nlm(image, kernel_size, pixels_to_process, nlm_search_window_size, nlm_h)
            update_progress(status, progress_bar, pixels_to_process, pixels_to_process)

            final_result = create_final_result(
                speckle_result, 
                nlm_result, 
                kernel_size, 
                pixels_to_process, 
                (height, width),
                nlm_search_window_size,
                nlm_h
            )
            finalize_results(final_result, checkpoint_filename, status)

        return final_result
    except Exception as e:
        handle_error(e, image)

# Helper functions for process_nl_speckle
def update_progress(status, progress_bar, current, total):
    progress = current / total
    percentage = round(progress * 100, 1)
    status.update(label=f"Processing: {percentage}% complete")
    progress_bar.progress(progress)

def create_final_result(speckle_result, nlm_result, kernel_size, pixels_processed, image_dimensions, nlm_search_window_size, nlm_h):
    height, width = image_dimensions
    valid_width = width - kernel_size + 1
    end_y, end_x = divmod(pixels_processed - 1, valid_width)
    processing_end = (min(end_x + kernel_size // 2, width - 1), min(end_y + kernel_size // 2, height - 1))
    
    return NLSpeckleResult(
        nlm_result=nlm_result,
        speckle_result=speckle_result,
        processing_end_coord=processing_end,
        kernel_size=kernel_size,
        pixels_processed=pixels_processed,
        image_dimensions=image_dimensions,
        nlm_search_window_size=nlm_search_window_size,
        nlm_h=nlm_h
    )

def finalize_results(final_result, checkpoint_filename, status):
    image_name = os.path.splitext(st.session_state.image_file)[0]
    checkpoint_dir = os.path.join("checkpoints", image_name)
    final_file = os.path.join(checkpoint_dir, checkpoint_filename)
    final_result.save_checkpoint(final_file)
    status.update(label="Processing complete!", state="complete")

def handle_error(e, image):
    st.error(f"Error in process_nl_speckle: {type(e).__name__}: {str(e)}")
    st.error(f"Image shape: {image.shape}, Image size: {image.size}, Image dtype: {image.dtype}")
    raise
