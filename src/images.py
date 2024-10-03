"""
This module handles image processing and visualization for the Streamlit application.
"""

import os
import glob
import time
import pickle
from enum import Enum
from typing import Dict, Any, List, Tuple, Optional, Union

import numpy as np
import streamlit as st

import src.session_state as session_state
from multiprocessing import Pool, cpu_count


class Technique(Enum):
    NLM = "nlm"
    LSCI = "lsci"


## Configuration Functions ##


def create_shared_config(
    technique: Union[Technique, str], params: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Create a shared configuration for both processing and overlays.

    Args:
        technique (Union[Technique, str]): The image processing technique to use.
        params (Dict[str, Any]): Parameters for the technique.

    Returns:
        Dict[str, Any]: A dictionary containing shared configuration settings.
    """
    nlm_options = session_state.get_nlm_options()
    image_array = session_state.get_image_array()
    height, width = image_array.shape
    kernel_size = params["kernel_size"]
    half_kernel = kernel_size // 2

    config = {
        "technique": technique.value if isinstance(technique, Technique) else technique,
        "kernel_size": kernel_size,
        "half_kernel": half_kernel,
        "search_window_size": params.get(
            "search_window_size", nlm_options["search_window_size"]
        ),
        "use_full_image": params.get("use_full_image", nlm_options["use_whole_image"]),
        "show_per_pixel_processing": session_state.get_session_state(
            "show_per_pixel", False
        ),
        "image_shape": (height, width),
        "total_pixels": image_array.size,
        "pixels_to_process": session_state.get_session_state(
            "pixels_to_process", image_array.size
        ),
        "processable_area": {
            "top": half_kernel,
            "bottom": height - half_kernel,
            "left": half_kernel,
            "right": width - half_kernel,
        },
        "total_area": {
            "top": 0,
            "bottom": height,
            "left": 0,
            "right": width,
        },
    }

    return config


## Progress Tracking ##


def update_progress(
    current: int,
    total: int,
    start_time: float,
    progress_bar: Any,
    status: st.delta_generator.DeltaGenerator,
):
    """
    Update the progress bar and status message.

    Args:
        current: Current number of processed items
        total: Total number of items to process
        start_time: Time when processing started
        progress_bar: Streamlit progress bar object
        status: Streamlit status message object
    """
    progress = current / total
    progress_bar.progress(progress)
    if progress > 0:
        elapsed_time = time.time() - start_time
        estimated_total_time = elapsed_time / progress
        remaining_time = estimated_total_time - elapsed_time
        status.text(
            f"Processed {current}/{total} pixels. Estimated time remaining: {remaining_time:.2f} seconds"
        )
    else:
        status.text("Initializing processing...")


## Pixel Processing Helpers ##


def create_pixel_list(
    processable_area: Dict[str, int], pixels_to_process: int
) -> List[Tuple[int, int]]:
    """Create a list of pixels to process based on the processable area."""
    all_pixels = [
        (y_coord, x_coord)
        for y_coord in range(processable_area["top"], processable_area["bottom"])
        for x_coord in range(processable_area["left"], processable_area["right"])
    ]
    return all_pixels[:pixels_to_process]


def initialize_result_images(image: np.ndarray, num_images: int) -> List[np.ndarray]:
    """Initialize result images."""
    return [np.zeros_like(image, dtype=np.float32) for _ in range(num_images)]


def update_result_images(
    result_images: List[np.ndarray],
    result: Tuple[int, int, float, float, np.ndarray],
    image_shape: Tuple[int, int],
) -> Tuple[int, int]:
    """Update result images with pixel processing results."""
    height, width = image_shape
    y_coord, x_coord, nlm_value, average_weight, similarity_map = result
    
    if 0 <= y_coord < height and 0 <= x_coord < width:
        result_images[0][y_coord, x_coord] = float(nlm_value)
        result_images[1][y_coord, x_coord] = float(average_weight)
    
    # Update the entire similarity map
    result_images[2] = similarity_map
    
    return y_coord, x_coord

## Result Formatting ##


def create_filter_data(
    technique: str,
    filter_images: List[np.ndarray],
    shared_config: Dict[str, Any],
    params: Dict[str, Any],
) -> Dict[str, Any]:
    """Create a dictionary of filter data for the given technique."""
    filter_names = {
        "nlm": ["NL Means", "Normalization Factors", "Last Similarity Map"],
        "lsci": ["Mean Filter", "Std Dev Filter", "LSCI"],
    }

    result = {
        name.lower().replace(" ", "_"): image
        for name, image in zip(filter_names[technique], filter_images)
    }
    result["filter_data"] = dict(zip(filter_names[technique], filter_images))

    if technique == "nlm":
        result.update(
            {
                "search_window_size": shared_config["search_window_size"],
                "filter_strength": params["filter_strength"],
            }
        )

    return result


def format_processing_result(
    processing_end: Tuple[int, int],
    shared_config: Dict[str, Any],
    params: Dict[str, Any],
    pixels: List[Tuple[int, int]],
    image_shape: Tuple[int, int],
    last_processed_pixel: Tuple[int, int],
    image: np.ndarray,
    technique: str,
    result_images: List[np.ndarray],
) -> Dict[str, Any]:
    """Format the processing results."""
    base_result = {
        "processing_end_coord": processing_end,
        "kernel_size": shared_config["kernel_size"],
        "pixels_processed": len(pixels),
        "image_dimensions": image_shape,
        "processable_area": shared_config["processable_area"],
        "last_processed_pixel": last_processed_pixel,
        "last_pixel_intensity": image[last_processed_pixel]
        if last_processed_pixel
        else None,
    }

    technique_specific_result = create_filter_data(
        technique, result_images, shared_config, params
    )  # Pass params here
    return {**base_result, **technique_specific_result}


## Generic Image Processor ##


class ImageProcessor:
    def __init__(
        self,
        image: np.ndarray,
        technique: Union[Technique, str],
        params: Dict[str, Any],
        image_name: str,
    ):
        """
        Initialize the ImageProcessor with the given image, technique, and parameters.
        """
        self.image = image
        self.technique = Technique(technique) if isinstance(technique, str) else technique
        self.params = params
        self.image_name = image_name
        self.height, self.width = image.shape
        
        self.shared_config = create_shared_config(technique, params)
        self.pixels = self._create_pixel_list()
        self.save_folder = "processing_states"
        self.save_path = self._generate_save_path()
        
        self._initialize_state()
        self.processor = self._initialize_processor()

    def _initialize_state(self):
        self.result_images = initialize_result_images(
            self.image, 5 if self.technique == Technique.NLM else 3
        )
        self.current_pixel: Optional[Tuple[int, int]] = None
        self.last_processed_pixel: Optional[Tuple[int, int]] = None
        # Change from list to set for faster lookup
        self.processed_pixels: set = set()
        self.load_state()

    def _create_pixel_list(self) -> List[Tuple[int, int]]:
        return create_pixel_list(
            self.shared_config["processable_area"],
            self.shared_config["pixels_to_process"],
        )

    def _initialize_processor(self):
        if self.technique == Technique.LSCI:
            from src.nl_lsci import LSCIProcessor
            return LSCIProcessor(self.image, self.shared_config["kernel_size"])
        elif self.technique == Technique.NLM:
            from src.nl_lsci import NLMProcessor
            return NLMProcessor(
                self.image,
                self.shared_config["kernel_size"],
                self.shared_config["search_window_size"],
                self.shared_config["use_full_image"],
                self.params["filter_strength"],
            )
        else:
            raise ValueError(f"Unsupported technique: {self.technique}")

    def _generate_save_path(self) -> str:
        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder)

        filename = f"{self.image_name}_{self.technique.value}_k{self.shared_config['kernel_size']}"
        if self.technique == Technique.NLM:
            filename += f"_s{self.shared_config['search_window_size']}_f{self.params['filter_strength']}"

        return os.path.join(self.save_folder, f"{filename}_*.pkl")

    def _get_state_file_path(self, pixels_processed):
        return self.save_path.replace("*", f"{pixels_processed}")

    def save_state(self) -> None:
        current_pixels_processed = len(self.processed_pixels)
        current_state = {
            "result_images": self.result_images,
            "processed_pixels": list(self.processed_pixels),  # Convert set to list for serialization
            "last_processed_pixel": self.last_processed_pixel,
            "pixels_processed": current_pixels_processed,
        }

        existing_states = glob.glob(self.save_path)
        max_pixels_processed = max((int(state_file.split("_")[-1].split(".")[0]) for state_file in existing_states), default=0)

        if current_pixels_processed > max_pixels_processed:
            save_path = self._get_state_file_path(current_pixels_processed)
            try:
                with open(save_path, "wb") as f:
                    pickle.dump(current_state, f)
                for old_state in existing_states:
                    os.remove(old_state)
            except (OSError, pickle.PicklingError) as e:
                st.error(f"Failed to save state: {e}")

    def load_state(self, target_pixels: Optional[int] = None) -> int:
        existing_states = glob.glob(self.save_path)
        if not existing_states:
            return 0

        best_state_file = max(existing_states, key=lambda x: int(x.split("_")[-1].split(".")[0]))

        try:
            with open(best_state_file, "rb") as f:
                state = pickle.load(f)
        except (OSError, pickle.UnpicklingError) as e:
            st.error(f"Failed to load state: {e}")
            return 0

        self._update_state_from_loaded_data(state)
        loaded_pixels = len(self.processed_pixels)

        if target_pixels is not None and target_pixels < loaded_pixels:
            self._trim_state_to_target(target_pixels)
            loaded_pixels = target_pixels

        return loaded_pixels

    def _update_state_from_loaded_data(self, state):
        self.result_images = state["result_images"]
        # Convert list back to set
        self.processed_pixels = set(state["processed_pixels"])
        self.last_processed_pixel = state["last_processed_pixel"]

    def _trim_state_to_target(self, target_pixels):
        # Convert set to list to maintain order if necessary
        processed_list = list(self.processed_pixels)[:target_pixels]
        self.processed_pixels = set(processed_list)
        self._update_result_images_for_trimmed_state()
        self.last_processed_pixel = processed_list[-1] if processed_list else None

    def _update_result_images_for_trimmed_state(self):
        mask_indices = np.array(list(self.processed_pixels)).T
        if mask_indices.size > 0:
            y_indices, x_indices = mask_indices
            for i, img in enumerate(self.result_images):
                temp_image = np.zeros_like(img, dtype=img.dtype)
                temp_image[y_indices, x_indices] = img[y_indices, x_indices]
                self.result_images[i] = temp_image
        else:
            self.result_images = [np.zeros_like(img) for img in self.result_images]

    def process_pixel(self, y_coord: int, x_coord: int):
        self.current_pixel = (y_coord, x_coord)
        return self.processor.process_pixel(y_coord, x_coord)

    def _handle_pixel_result(self, result: Tuple[int, int, float, float, float]):
        self.last_processed_pixel = update_result_images(
            self.result_images, result, (self.height, self.width)
        )
        # Add to set instead of list
        self.processed_pixels.add((result[0], result[1]))

    def format_result(self, processing_end: Tuple[int, int], target_pixels: int = None) -> Dict[str, Any]:
        processed_pixels = list(self.processed_pixels)[:target_pixels] if target_pixels is not None else list(self.processed_pixels)

        mask = np.zeros(self.image.shape, dtype=bool)
        if processed_pixels:
            y_coords, x_coords = zip(*processed_pixels)
            mask[y_coords, x_coords] = True

        result_images = [np.where(mask, img, 0) for img in self.result_images]

        return format_processing_result(
            processing_end,
            self.shared_config,
            self.params,
            processed_pixels,
            self.image.shape,
            self.last_processed_pixel,
            self.image,
            self.technique.value,
            result_images,
        )

    def run_sequential_processing(self, target_pixels: Optional[int] = None) -> Dict[str, Any]:
        progress_bar = st.progress(0)
        status = st.empty()
        start_time = time.time()

        target_pixels = target_pixels or self.params.get("pixels_to_process", self.shared_config["pixels_to_process"])
        loaded_pixels = self.load_state(target_pixels)

        try:
            with st.spinner(f"Processing {self.technique.value.upper()}..."):
                if self.technique == Technique.NLM:
                    self._process_pixels_parallel(progress_bar, status, start_time, loaded_pixels, target_pixels)
                else:
                    self._process_pixels(progress_bar, status, start_time, loaded_pixels, target_pixels)
        except Exception as e:
            st.error(f"Error during processing: {e}")
        finally:
            self.save_state()

        processing_end = self.current_pixel or (0, 0)
        result = self.format_result(processing_end, target_pixels)
        result.update({
            "pixels_processed": len(self.processed_pixels),
            "last_processed_pixel": self.last_processed_pixel,
            "loaded_state_pixels": loaded_pixels,
        })

        return result

    def _process_pixels(
        self,
        progress_bar: Any,
        status: st.delta_generator.DeltaGenerator,
        start_time: float,
        loaded_pixels: int,
        target_pixels: int = None,
    ):
        # Existing setup
        save_interval = 5000  # Increase interval to reduce frequency

        total_pixels = min(target_pixels or len(self.pixels), len(self.pixels))
        total_image_pixels = self.image.size

        # Cache frequently accessed attributes
        processed_pixels = self.processed_pixels
        pixels = self.pixels
        process_pixel = self.process_pixel
        handle_pixel_result = self._handle_pixel_result
        update_progress = self._update_progress
        save_state = self.save_state

        for i, (y_coord, x_coord) in enumerate(pixels[loaded_pixels:total_pixels], start=loaded_pixels + 1):
            if (y_coord, x_coord) not in processed_pixels:
                result = process_pixel(y_coord, x_coord)
                if result is not None:
                    handle_pixel_result(result)

            update_progress(i, total_pixels, total_image_pixels, start_time, progress_bar, status)

            if i % save_interval == 0:
                save_state()

        progress_bar.progress(1.0)
        self.last_processed_pixel = self.pixels[total_pixels - 1] if self.pixels else None

    def _update_progress(self, i, total_pixels, total_image_pixels, start_time, progress_bar, status):
        current_progress = i / total_pixels
        progress_bar.progress(min(current_progress, 1.0))

        if current_progress > 0:
            elapsed_time = time.time() - start_time
            estimated_total_time = elapsed_time / current_progress
            remaining_time = max(0, estimated_total_time - elapsed_time)

            time_str = self._format_time(remaining_time)
            percent_of_image = (i / total_image_pixels) * 100

            status.text(
                f"Processed {i}/{total_pixels} pixels ({percent_of_image:.2f}% of total image). "
                f"Estimated time remaining: {time_str}"
            )
        else:
            status.text("Initializing processing...")

    @staticmethod
    def _format_time(seconds):
        if seconds > 3600:
            return f"{seconds / 3600:.1f} hours"
        elif seconds > 60:
            return f"{seconds / 60:.1f} minutes"
        else:
            return f"{seconds:.1f} seconds"

    def _process_pixels_parallel(
        self,
        progress_bar: Any,
        status: st.delta_generator.DeltaGenerator,
        start_time: float,
        loaded_pixels: int,
        target_pixels: int = None,
    ):
        total_pixels = min(target_pixels or len(self.pixels), len(self.pixels))
        total_image_pixels = self.image.size

        # Use only the unprocessed pixels
        pixels_to_process = self.pixels[loaded_pixels:total_pixels]

        with Pool(processes=cpu_count()) as pool:
            for i, result in enumerate(pool.imap(self.processor.process_pixel, pixels_to_process), start=loaded_pixels + 1):
                if result is not None:
                    self._handle_pixel_result(result)

                self._update_progress(i, total_pixels, total_image_pixels, start_time, progress_bar, status)

                if i % 5000 == 0:  # Save state every 5000 pixels
                    self.save_state()

        progress_bar.progress(1.0)
        self.last_processed_pixel = self.pixels[total_pixels - 1] if self.pixels else None