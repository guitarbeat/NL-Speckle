from dataclasses import dataclass,field
from typing import List, Tuple, Dict, Any
import numpy as np
import joblib
import os
from src.nlm import process_nlm, NLMResult
from src.speckle import process_speckle, SpeckleResult
import streamlit as st

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

    def add_image(self, name: str, image: np.ndarray):
        self.additional_images[name] = image

    def get_all_images(self) -> Dict[str, np.ndarray]:
        images = {}
        for prefix, result in [("NLM", self.nlm_result), ("Speckle", self.speckle_result)]:
            images.update({f"{prefix} {k}": v for k, v in result.get_filter_data().items()})
        images.update(self.additional_images)
        return images

    def get_filter_options(self) -> List[str]:
        return [f"{prefix} {option}" for prefix, result in [("NLM", self.nlm_result), ("Speckle", self.speckle_result)]
                for option in result.get_filter_options()] + list(self.additional_images.keys())

    def get_filter_data(self) -> Dict[str, Any]:
        return self.get_all_images()

    def get_last_processed_coordinates(self) -> Tuple[int, int]:
        return self.processing_end_coord

    def save_checkpoint(self, filename: str):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        joblib.dump(self, filename)

    @classmethod
    def load_checkpoint(cls, filename: str) -> 'NLSpeckleResult':
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Checkpoint file not found: {filename}")
        return joblib.load(filename)

    @classmethod
    def combine(cls, nlm_results: List[NLMResult], speckle_results: List[SpeckleResult], 
                kernel_size: int, pixels_processed: int, image_dimensions: Tuple[int, int]) -> 'NLSpeckleResult':
        return cls(
            nlm_result=NLMResult.combine(nlm_results),
            speckle_result=SpeckleResult.combine(speckle_results),
            processing_end_coord=(0, 0),
            kernel_size=kernel_size,
            pixels_processed=pixels_processed,
            image_dimensions=image_dimensions
        )
    

def process_nl_speckle(
    image: np.ndarray,
    kernel_size: int,
    pixels_to_process: int,
    nlm_search_window_size: int = 21,
    nlm_h: float = 10.0,
    checkpoint_interval: int = 10000
) -> NLSpeckleResult:
    try:
        image = image.astype(np.float32)
        height, width = image.shape
        valid_height, valid_width = height - kernel_size + 1, width - kernel_size + 1
        pixels_to_process = min(pixels_to_process, valid_height * valid_width)
        checkpoint_file = os.path.join("checkpoints", f"nl_speckle_checkpoint_{st.session_state.image_file}.joblib")

        with st.status("Processing image...", expanded=True) as status:
            start_pixel, speckle_results, nlm_results = load_or_initialize_checkpoint(checkpoint_file, status)
            progress_bar = st.progress(start_pixel / pixels_to_process)

            for chunk_start in range(start_pixel, pixels_to_process, checkpoint_interval):
                chunk_end = min(chunk_start + checkpoint_interval, pixels_to_process)
                chunk_size = chunk_end - chunk_start

                speckle_result = process_speckle(image, kernel_size, chunk_size, start_pixel=chunk_start)
                nlm_result = process_nlm(image, kernel_size, chunk_size, nlm_search_window_size, nlm_h, start_pixel=chunk_start)
                
                speckle_results.append(speckle_result)
                nlm_results.append(nlm_result)
                
                save_checkpoint(speckle_results, nlm_results, kernel_size, chunk_end, (height, width), checkpoint_file)
                update_progress(status, progress_bar, chunk_end, pixels_to_process)

            finalize_results(speckle_results, nlm_results, kernel_size, pixels_to_process, (height, width), 
                             st.session_state.image_file, checkpoint_file, status)

        return create_final_result(speckle_results, nlm_results, kernel_size, pixels_to_process, (height, width))
    except Exception as e:
        st.error(f"Error in process_nl_speckle: {type(e).__name__}: {str(e)}")
        st.error(f"Image shape: {image.shape}, Image size: {image.size}, Image dtype: {image.dtype}")
        raise

def load_or_initialize_checkpoint(checkpoint_file, status):
    if os.path.exists(checkpoint_file):
        try:
            checkpoint_result = NLSpeckleResult.load_checkpoint(checkpoint_file)
            status.update(label=f"Resuming from checkpoint. {checkpoint_result.pixels_processed} pixels already processed.")
            return checkpoint_result.pixels_processed, [checkpoint_result.speckle_result], [checkpoint_result.nlm_result]
        except Exception as e:
            status.update(label=f"Error loading checkpoint: {str(e)}. Starting fresh calculation.")
    else:
        status.update(label="No checkpoint found. Starting fresh calculation.")
    return 0, [], []

def save_checkpoint(speckle_results, nlm_results, kernel_size, pixels_processed, image_dimensions, checkpoint_file):
    combined_result = NLSpeckleResult(
        nlm_result=NLMResult.combine(nlm_results),
        speckle_result=SpeckleResult.combine(speckle_results),
        processing_end_coord=(0, 0),
        kernel_size=kernel_size,
        pixels_processed=pixels_processed,
        image_dimensions=image_dimensions
    )
    combined_result.save_checkpoint(checkpoint_file)

def update_progress(status, progress_bar, current, total):
    progress = current / total
    status.update(label=f"Processing: {current}/{total} pixels")
    progress_bar.progress(progress)

def create_final_result(speckle_results, nlm_results, kernel_size, pixels_processed, image_dimensions):
    height, width = image_dimensions
    valid_width = width - kernel_size + 1
    end_y, end_x = divmod(pixels_processed - 1, valid_width)
    processing_end = (min(end_x + kernel_size // 2, width - 1), min(end_y + kernel_size // 2, height - 1))
    
    return NLSpeckleResult(
        nlm_result=NLMResult.combine(nlm_results),
        speckle_result=SpeckleResult.combine(speckle_results),
        processing_end_coord=processing_end,
        kernel_size=kernel_size,
        pixels_processed=pixels_processed,
        image_dimensions=image_dimensions
    )

def finalize_results(speckle_results, nlm_results, kernel_size, pixels_processed, image_dimensions, 
                     image_file, checkpoint_file, status):
    final_result = create_final_result(speckle_results, nlm_results, kernel_size, pixels_processed, image_dimensions)
    final_file = os.path.join("checkpoints", f"nl_speckle_final_{image_file}.joblib")
    final_result.save_checkpoint(final_file)
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)
    status.update(label="Processing complete!", state="complete")
