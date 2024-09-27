import os
import fcntl
import logging
import numpy as np
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import dill
from typing import List, Tuple, Dict, Any

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Base Classes ---

class Checkpointable:
    def save_checkpoint(self, filename: str):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        temp_file = f"{filename}.tmp"
        lock_file = f"{filename}.lock"

        with open(lock_file, 'w') as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            try:
                with open(temp_file, 'wb') as f:
                    serialized_data = dill.dumps(self, recurse=True)
                    f.write(serialized_data)
                os.rename(temp_file, filename)
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
                os.remove(lock_file)

    @classmethod
    def load_checkpoint(cls, filename: str):
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Checkpoint file not found: {filename}")
        
        lock_file = f"{filename}.lock"
        with open(lock_file, 'w') as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_SH)
            try:
                with open(filename, 'rb') as f:
                    serialized_data = f.read()
                    return dill.loads(serialized_data)
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
                os.remove(lock_file)

@dataclass
class BaseResult(Checkpointable, ABC):
    processing_end_coord: Tuple[int, int]
    kernel_size: int
    pixels_processed: int
    image_dimensions: Tuple[int, int]

    def get_last_processed_coordinates(self) -> Tuple[int, int]:
        return self.processing_end_coord

    @classmethod
    def combine(cls, results: List["BaseResult"]) -> "BaseResult":
        if not results:
            raise ValueError("No results to combine")
        return cls(
            processing_end_coord=max(r.processing_end_coord for r in results),
            kernel_size=results[0].kernel_size,
            pixels_processed=sum(r.pixels_processed for r in results),
            image_dimensions=results[0].image_dimensions,
        )

    @classmethod
    def empty_result(cls) -> "BaseResult":
        return cls(
            processing_end_coord=(0, 0),
            kernel_size=0,
            pixels_processed=0,
            image_dimensions=(0, 0),
        )

    @staticmethod
    @abstractmethod
    def get_filter_options() -> List[str]:
        pass

    @abstractmethod
    def get_filter_data(self) -> Dict[str, np.ndarray]:
        pass

@dataclass
class NLMResult(BaseResult):
    nonlocal_means: np.ndarray
    normalization_factors: np.ndarray
    nonlocal_std: np.ndarray
    nonlocal_speckle: np.ndarray
    search_window_size: int
    filter_strength: float
    last_similarity_map: np.ndarray
    _combine: classmethod = field(default=None, repr=False, compare=False)

    @staticmethod
    def get_filter_options() -> List[str]:
        return [
            "Non-Local Means",
            "Normalization Factors",
            "Last Similarity Map",
            "Non-Local Standard Deviation",
            "Non-Local Speckle",
        ]

    def get_filter_data(self) -> Dict[str, np.ndarray]:
        return {
            "Non-Local Means": self.nonlocal_means,
            "Normalization Factors": self.normalization_factors,
            "Last Similarity Map": self.last_similarity_map,
            "Non-Local Standard Deviation": self.nonlocal_std,
            "Non-Local Speckle": self.nonlocal_speckle,
        }

    @classmethod
    def combine(cls, results: List["NLMResult"]) -> "NLMResult":
        if not results:
            return cls.empty_result()

        combined_arrays = {
            attr: np.maximum.reduce([getattr(r, attr) for r in results])
            for attr in ['nonlocal_means', 'normalization_factors', 'nonlocal_std', 'nonlocal_speckle']
        }

        return cls(
            **combined_arrays,
            **BaseResult.combine(results).__dict__,
            search_window_size=results[0].search_window_size,
            filter_strength=results[0].filter_strength,
            last_similarity_map=results[-1].last_similarity_map,
        )

    @classmethod
    def empty_result(cls) -> "NLMResult":
        return cls(
            nonlocal_means=np.array([]),
            normalization_factors=np.array([]),
            nonlocal_std=np.array([]),
            nonlocal_speckle=np.array([]),
            processing_end_coord=(0, 0),
            kernel_size=0,
            pixels_processed=0,
            image_dimensions=(0, 0),
            search_window_size=0,
            filter_strength=0,
            last_similarity_map=np.array([]),
        )

    @classmethod
    def merge(cls, new_result: 'NLMResult', existing_result: 'NLMResult') -> 'NLMResult':
        merged_arrays = {
            attr: np.maximum(getattr(new_result, attr), getattr(existing_result, attr))
            for attr in ['nonlocal_means', 'normalization_factors', 'nonlocal_std', 'nonlocal_speckle']
        }
        
        return cls(
            **merged_arrays,
            processing_end_coord=max(new_result.processing_end_coord, existing_result.processing_end_coord),
            kernel_size=new_result.kernel_size,
            pixels_processed=max(new_result.pixels_processed, existing_result.pixels_processed),
            image_dimensions=new_result.image_dimensions,
            search_window_size=new_result.search_window_size,
            filter_strength=new_result.filter_strength,
            last_similarity_map=new_result.last_similarity_map
        )

@dataclass
class SpeckleResult(BaseResult):
    mean_filter: np.ndarray
    std_dev_filter: np.ndarray
    speckle_contrast_filter: np.ndarray
    _combine: classmethod = field(default=None, repr=False, compare=False)

    @staticmethod
    def get_filter_options() -> List[str]:
        return ["Mean Filter", "Std Dev Filter", "Speckle Contrast"]

    def get_filter_data(self) -> Dict[str, np.ndarray]:
        return {
            "Mean Filter": self.mean_filter,
            "Std Dev Filter": self.std_dev_filter,
            "Speckle Contrast": self.speckle_contrast_filter,
        }

    @classmethod
    def combine(cls, results: List["SpeckleResult"]) -> "SpeckleResult":
        if not results:
            raise ValueError("No results to combine")

        combined_arrays = {
            attr: np.maximum.reduce([getattr(r, attr) for r in results])
            for attr in ['mean_filter', 'std_dev_filter', 'speckle_contrast_filter']
        }

        return cls(
            **combined_arrays,
            **BaseResult.combine(results).__dict__,
        )

    @classmethod
    def merge(cls, new_result: 'SpeckleResult', existing_result: 'SpeckleResult') -> 'SpeckleResult':
        merged_mean = np.maximum(new_result.mean_filter, existing_result.mean_filter)
        merged_std = np.maximum(new_result.std_dev_filter, existing_result.std_dev_filter)
        merged_sc = np.maximum(new_result.speckle_contrast_filter, existing_result.speckle_contrast_filter)
        
        return cls(
            mean_filter=merged_mean,
            std_dev_filter=merged_std,
            speckle_contrast_filter=merged_sc,
            processing_end_coord=max(new_result.processing_end_coord, existing_result.processing_end_coord),
            kernel_size=new_result.kernel_size,
            pixels_processed=max(new_result.pixels_processed, existing_result.pixels_processed),
            image_dimensions=new_result.image_dimensions
        )

@dataclass
class NLSpeckleResult(Checkpointable):
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
        return (
            [f"{prefix} {option}" for prefix, result in [("NLM", self.nlm_result), ("Speckle", self.speckle_result)]
             for option in result.get_filter_options()]
            + list(self.additional_images.keys())
        )

    def get_filter_data(self) -> Dict[str, Any]:
        return self.get_all_images()

    def get_last_processed_coordinates(self) -> Tuple[int, int]:
        return self.processing_end_coord

    @classmethod
    def combine(cls, nlm_results: List[NLMResult], speckle_results: List[SpeckleResult],
                kernel_size: int, pixels_processed: int, image_dimensions: Tuple[int, int],
                nlm_search_window_size: int, nlm_h: float) -> 'NLSpeckleResult':
        return cls(
            nlm_result=NLMResult.combine(nlm_results),
            speckle_result=SpeckleResult.combine(speckle_results),
            processing_end_coord=cls._get_max_processing_end_coord(nlm_results, speckle_results),
            kernel_size=kernel_size,
            pixels_processed=pixels_processed,
            image_dimensions=image_dimensions,
            nlm_search_window_size=nlm_search_window_size,
            nlm_h=nlm_h
        )

    @staticmethod
    def _get_max_processing_end_coord(nlm_results: List[NLMResult], speckle_results: List[SpeckleResult]) -> Tuple[int, int]:
        all_coords = [result.processing_end_coord for result in nlm_results + speckle_results]
        return max(all_coords, key=lambda coord: coord[0] * 10000 + coord[1])  # Prioritize y-coordinate

    def merge_with_existing(self, existing_result: 'NLSpeckleResult'):
        self.speckle_result = SpeckleResult.merge(self.speckle_result, existing_result.speckle_result)
        self.nlm_result = NLMResult.merge(self.nlm_result, existing_result.nlm_result)
        self.pixels_processed = max(self.pixels_processed, existing_result.pixels_processed)
        self.processing_end_coord = max(self.processing_end_coord, existing_result.processing_end_coord)


# --- Utility Functions ---

def get_checkpoint_path(image_name: str, kernel_size: int, pixels_to_process: int, 
                        nlm_search_window_size: int, nlm_h: float) -> str:
    checkpoint_filename = f"k{kernel_size}_p{pixels_to_process}_w{nlm_search_window_size}_h{nlm_h}.joblib"
    checkpoint_dir = os.path.join("checkpoints", image_name)
    return os.path.join(checkpoint_dir, checkpoint_filename)

# --- Processing Functions ---
