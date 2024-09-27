import os
import asyncio
import glob
import numpy as np
import streamlit as st
from typing import Tuple, Optional
from contextlib import contextmanager
from src.classes import NLSpeckleResult, get_checkpoint_path
from src.process import process_speckle, process_nlm

class ProcessingManager:
    def __init__(self, image: np.ndarray, kernel_size: int, pixels_to_process: int, 
                 nlm_search_window_size: int = 21, nlm_h: float = 10.0):
        self.image, self.height, self.width, self.pixels_to_process = self._prepare_image(image, kernel_size, pixels_to_process)
        self.kernel_size = kernel_size
        self.nlm_search_window_size = nlm_search_window_size
        self.nlm_h = nlm_h
        self.checkpoint_file = self._get_checkpoint_file()

    def _prepare_image(self, image: np.ndarray, kernel_size: int, pixels_to_process: int) -> Tuple[np.ndarray, int, int, int]:
        image = image.astype(np.float32)
        height, width = image.shape
        valid_pixels = (height - kernel_size + 1) * (width - kernel_size + 1)
        return image, height, width, min(pixels_to_process, valid_pixels)

    def _get_checkpoint_file(self) -> str:
        image_name = os.path.splitext(st.session_state.image_file)[0]
        return get_checkpoint_path(image_name, self.kernel_size, self.pixels_to_process, 
                                   self.nlm_search_window_size, self.nlm_h)

    @contextmanager
    def _processing_status(self):
        with st.status("Processing image...", expanded=True) as status:
            progress_bar = st.progress(0)
            yield status, progress_bar

    def _update_progress(self, status, progress_bar, current: int, total: int):
        progress = current / total
        status.update(label=f"Processing: {progress:.1%} complete")
        progress_bar.progress(progress)

    def process(self) -> Optional[NLSpeckleResult]:
        try:
            if os.path.exists(self.checkpoint_file):
                return self._load_checkpoint()

            with self._processing_status() as (status, progress_bar):
                speckle_result = self._run_speckle_processing(status, progress_bar)
                nlm_result = self._run_nlm_processing(status, progress_bar)
                final_result = self._create_final_result(speckle_result, nlm_result)
                self._save_and_cleanup_checkpoints(final_result, status)

            return final_result
        except Exception as e:
            self._handle_error(e)
            return None

    def _run_speckle_processing(self, status, progress_bar) -> np.ndarray:
        result = process_speckle(self.image, self.kernel_size, self.pixels_to_process)
        if result is None:
            raise ValueError("Speckle processing failed to produce a result")
        self._update_progress(status, progress_bar, self.pixels_to_process // 2, self.pixels_to_process)
        
        st.session_state.nl_speckle_result = self._create_partial_result(result)
        return result

    def _run_nlm_processing(self, status, progress_bar) -> np.ndarray:
        result = process_nlm(self.image, self.kernel_size, self.pixels_to_process, 
                             self.nlm_search_window_size, self.nlm_h)
        if result is None:
            raise ValueError("NLM processing failed to produce a result")
        self._update_progress(status, progress_bar, self.pixels_to_process, self.pixels_to_process)
        return result

    def _create_final_result(self, speckle_result: np.ndarray, nlm_result: np.ndarray) -> NLSpeckleResult:
        return NLSpeckleResult(
            nlm_result=nlm_result,
            speckle_result=speckle_result,
            additional_images={},
            processing_end_coord=self._calculate_processing_end(),
            kernel_size=self.kernel_size,
            pixels_processed=self.pixels_to_process,
            image_dimensions=(self.height, self.width),
            nlm_search_window_size=self.nlm_search_window_size,
            nlm_h=self.nlm_h
        )

    def _create_partial_result(self, speckle_result: np.ndarray) -> NLSpeckleResult:
        return NLSpeckleResult(
            speckle_result=speckle_result,
            nlm_result=None,
            additional_images={},
            processing_end_coord=self._calculate_processing_end(),
            kernel_size=self.kernel_size,
            pixels_processed=self.pixels_to_process,
            image_dimensions=(self.height, self.width),
            nlm_search_window_size=self.nlm_search_window_size,
            nlm_h=self.nlm_h
        )

    def _calculate_processing_end(self) -> Tuple[int, int]:
        valid_width = self.width - self.kernel_size + 1
        end_y, end_x = divmod(self.pixels_to_process - 1, valid_width)
        return (min(end_x + self.kernel_size // 2, self.width - 1), 
                min(end_y + self.kernel_size // 2, self.height - 1))

    def _load_checkpoint(self) -> NLSpeckleResult:
        loaded_result = NLSpeckleResult.load_checkpoint(self.checkpoint_file)
        return loaded_result.add_image("Loaded From", np.array([self.checkpoint_file]))

    async def _async_cleanup_checkpoints(self, matching_checkpoints: list[str]):
        tasks = [asyncio.to_thread(os.remove, old) for old in matching_checkpoints if old != self.checkpoint_file]
        await asyncio.gather(*tasks)

    def _save_and_cleanup_checkpoints(self, final_result: NLSpeckleResult, status):
        try:
            final_result.save_checkpoint(self.checkpoint_file)
            status.update(label="Checkpoint saved. Cleaning up old checkpoints...", state="running")
            
            checkpoint_dir = os.path.dirname(self.checkpoint_file)
            pattern = os.path.basename(self.checkpoint_file).replace(str(final_result.pixels_processed), '*')
            matching_checkpoints = glob.glob(os.path.join(checkpoint_dir, pattern))
            
            asyncio.run(self._async_cleanup_checkpoints(matching_checkpoints))
            
            status.update(label="Processing complete!", state="complete")
        except Exception as e:
            st.error(f"Error in saving checkpoint: {str(e)}")
            status.update(label="Processing complete, but failed to save checkpoint.", state="error")

    def _handle_error(self, e: Exception):
        error_message = f"Error in process_nl_speckle: {type(e).__name__}: {str(e)}"
        image_info = f"Image shape: {self.image.shape}, Image size: {self.image.size}, Image dtype: {self.image.dtype}"
        st.error(f"{error_message}\n{image_info}")
        raise

def process_nl_speckle(image: np.ndarray, kernel_size: int, pixels_to_process: int, 
                       nlm_search_window_size: int = 21, nlm_h: float = 10.0) -> Optional[NLSpeckleResult]:
    try:
        manager = ProcessingManager(image, kernel_size, pixels_to_process, nlm_search_window_size, nlm_h)
        return manager.process()
    except Exception as e:
        st.error(f"Error in process_nl_speckle: {type(e).__name__}: {str(e)}")
        raise