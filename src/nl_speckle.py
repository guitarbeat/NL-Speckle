"""
This module combines functionality from Non-Local Means (NLM) denoising
and Speckle analysis to provide comprehensive image processing capabilities.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple
import numpy as np
import streamlit as st
from src.nlm import NLMResult, process_nlm
from src.speckle import SpeckleResult, process_speckle
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class NLSpeckleResult:
    """
    Data class to hold the combined results of NLM denoising and Speckle analysis.
    """
    nlm_result: NLMResult
    speckle_result: SpeckleResult
    additional_images: Dict[str, np.ndarray] = field(default_factory=dict)
    processing_end_coord: Tuple[int, int] = field(default=(0, 0))
    kernel_size: int = field(default=0)
    pixels_processed: int = field(default=0)
    image_dimensions: Tuple[int, int] = field(default=(0, 0))

    def add_image(self, name: str, image: np.ndarray):
        """
        Add an additional image to the result.

        Args:
            name (str): Name of the image
            image (np.ndarray): The image array
        """
        self.additional_images[name] = image

    def get_all_images(self) -> Dict[str, np.ndarray]:
        """
        Get all images, including NLM, Speckle, and additional images.

        Returns:
            Dict[str, np.ndarray]: Dictionary of all images
        """
        images = {}
        
        # Dynamically add NLM images
        nlm_data = self.nlm_result.get_filter_data()
        for key, value in nlm_data.items():
            images[f"NLM {key}"] = value

        # Dynamically add Speckle images
        speckle_data = self.speckle_result.get_filter_data()
        for key, value in speckle_data.items():
            images[f"Speckle {key}"] = value

        # Add additional images
        images.update(self.additional_images)
        return images

    def get_filter_options(self) -> List[str]:
        """
        Get all available filter options.

        Returns:
            List[str]: List of all filter options
        """
        nlm_options = [f"NLM {option}" for option in self.nlm_result.get_filter_options()] if self.nlm_result else []
        speckle_options = [f"Speckle {option}" for option in self.speckle_result.get_filter_options()] if self.speckle_result else []
        additional_options = list(self.additional_images.keys())
        return nlm_options + speckle_options + additional_options

    def get_filter_data(self) -> Dict[str, Any]:
        """
        Get combined filter data from NLM and Speckle results.

        Returns:
            Dict[str, Any]: Combined filter data
        """
        data = {}
        if self.nlm_result:
            data.update({f"NLM {k}": v for k, v in self.nlm_result.get_filter_data().items()})
        if self.speckle_result:
            data.update({f"Speckle {k}": v for k, v in self.speckle_result.get_filter_data().items()})
        data.update(self.additional_images)
        return data

    def get_last_processed_coordinates(self) -> Tuple[int, int]:
        """Get the last processed pixel coordinates."""
        return self.processing_end_coord

def process_nl_speckle(
    image: np.ndarray,
    kernel_size: int,
    pixels_to_process: int,
    nlm_search_window_size: int = 21,
    nlm_h: float = 10.0,
    checkpoint_interval: int = 1000000
) -> NLSpeckleResult:
    """
    Process an image using both NLM denoising and Speckle analysis with optimizations.
    """
    try:
        # Convert image to float32
        image = image.astype(np.float32)
        height, width = image.shape
        half_kernel = kernel_size // 2
        valid_height, valid_width = height - kernel_size + 1, width - kernel_size + 1
        pixels_to_process = min(pixels_to_process, valid_height * valid_width)

        # Process in chunks for checkpointing
        chunks = range(0, pixels_to_process, checkpoint_interval)
        nlm_results = []
        speckle_results = []

        with ThreadPoolExecutor() as executor:
            futures = []
            for start in chunks:
                end = min(start + checkpoint_interval, pixels_to_process)
                nlm_future = executor.submit(process_nlm, image, kernel_size, end - start, nlm_search_window_size, nlm_h)
                speckle_future = executor.submit(process_speckle, image, kernel_size, end - start)
                futures.append((nlm_future, speckle_future))

            for i, (nlm_future, speckle_future) in enumerate(tqdm(futures, desc="Processing chunks")):
                try:
                    nlm_results.append(nlm_future.result())
                    speckle_results.append(speckle_future.result())
                    logger.info(f"Processed chunk {i+1}/{len(chunks)}")
                except Exception as e:
                    logger.error(f"Error processing chunk {i+1}: {str(e)}")

        # Combine results
        nlm_result = NLMResult.combine(nlm_results)
        speckle_result = SpeckleResult.combine(speckle_results)

        # Calculate processing end coordinates
        end_y, end_x = divmod(pixels_to_process - 1, valid_width)
        end_y, end_x = end_y + half_kernel, end_x + half_kernel
        processing_end = (min(end_x, width - 1), min(end_y, height - 1))

        return NLSpeckleResult(
            nlm_result=nlm_result,
            speckle_result=speckle_result,
            processing_end_coord=processing_end,
            kernel_size=kernel_size,
            pixels_processed=pixels_to_process,
            image_dimensions=(height, width)
        )
    except Exception as e:
        logger.error(f"Error in process_nl_speckle: {type(e).__name__}: {str(e)}")
        logger.error(f"Image shape: {image.shape}, Image size: {image.size}, Image dtype: {image.dtype}")
        raise

# Additional utility functions can be added here as needed