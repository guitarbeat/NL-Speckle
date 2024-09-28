import numpy as np
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any, Optional, Type

# Data Classes
@dataclass
class KernelVisualizationConfig:
    size: int
    kernel_matrix: Optional[np.ndarray] = None
    outline_color: str = "red"
    outline_width: float = 1
    grid_line_color: str = "red"
    grid_line_style: str = ":"
    grid_line_width: float = 1
    center_pixel_color: str = "green"
    center_pixel_outline_width: float = 2.0
    origin: Tuple[int, int] = (0, 0)

    def __post_init__(self):
        if self.kernel_matrix is not None and not isinstance(self.kernel_matrix, np.ndarray):
            raise TypeError("kernel_matrix must be a numpy array or None")
        
        if self.kernel_matrix is not None and self.kernel_matrix.shape != (self.size, self.size):
            raise ValueError(f"kernel_matrix shape {self.kernel_matrix.shape} does not match size {self.size}")

@dataclass
class SearchWindowConfig:
    size: Optional[int] = None
    outline_color: str = "blue"
    outline_width: float = 2.0
    use_full_image: bool = True

@dataclass
class PixelValueConfig:
    text_color: str = "red"
    font_size: int = 10

@dataclass
class VisualizationConfig:
    """Holds configuration for image visualization and analysis settings."""

    vmin: Optional[float] = None
    vmax: Optional[float] = None
    zoom: bool = False
    show_kernel: bool = False
    show_per_pixel_processing: bool = False
    image_array: Optional[np.ndarray] = None
    analysis_params: Dict[str, Any] = field(default_factory=dict)
    results: Optional[Any] = None
    ui_placeholders: Dict[str, Any] = field(default_factory=dict)
    last_processed_pixel: Optional[Tuple[int, int]] = None
    original_pixel_value: float = (0.0)
    technique: str = ""
    title: str = ""
    figure_size: Tuple[int, int] = (8, 8)
    kernel: KernelVisualizationConfig = field(default_factory=KernelVisualizationConfig)
    search_window: SearchWindowConfig = field(default_factory=SearchWindowConfig)
    pixel_value: PixelValueConfig = field(default_factory=PixelValueConfig)
    processing_end: Tuple[int, int] = field(default_factory=tuple)
    pixels_to_process: int = 0

    def __post_init__(self):
        """Post-initialization validation."""
        self._validate_vmin_vmax()

    def _validate_vmin_vmax(self):
        """Ensure vmin is not greater than vmax."""
        if self.vmin is not None and self.vmax is not None and self.vmin > self.vmax:
            raise ValueError("vmin cannot be greater than vmax.")

# --- Base Classes ---

class ResultCombinationError(Exception):
    """Exception raised when there's an error combining results."""
    pass

@dataclass
class BaseResult(ABC):
    processing_end_coord: Tuple[int, int]
    kernel_size: int
    pixels_processed: int
    image_dimensions: Tuple[int, int]

    @classmethod
    def combine(
        class_: Type["BaseResult"], results: List["BaseResult"]
    ) -> "BaseResult":
        if not results:
            raise ResultCombinationError("No results provided for combination")
        return class_(
            processing_end_coord=max(r.processing_end_coord for r in results),
            kernel_size=results[0].kernel_size,
            pixels_processed=sum(r.pixels_processed for r in results),
            image_dimensions=results[0].image_dimensions,
        )

    @classmethod
    def empty_result(class_: Type["BaseResult"]) -> "BaseResult":
        return class_(
            processing_end_coord=(0, 0),
            kernel_size=0,
            pixels_processed=0,
            image_dimensions=(0, 0),
        )

    @staticmethod
    @abstractmethod
    def get_filter_options() -> List[str]:
        pass

    @property
    @abstractmethod
    def filter_data(self) -> Dict[str, np.ndarray]:
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

    @staticmethod
    def get_filter_options() -> List[str]:
        return [
            "Non-Local Means",
            "Normalization Factors",
            "Last Similarity Map",
            "Non-Local Standard Deviation",
            "Non-Local Speckle",
        ]

    @property
    def filter_data(self) -> Dict[str, np.ndarray]:
        return {
            "Non-Local Means": self.nonlocal_means,
            "Normalization Factors": self.normalization_factors,
            "Last Similarity Map": self.last_similarity_map,
            "Non-Local Standard Deviation": self.nonlocal_std,
            "Non-Local Speckle": self.nonlocal_speckle,
        }

    @classmethod
    def combine(class_: Type["NLMResult"], results: List["NLMResult"]) -> "NLMResult":
        if not results:
            return class_.empty_result()

        try:
            combined_arrays: Dict[str, np.ndarray] = {
                attr: np.maximum.reduce([getattr(r, attr) for r in results])
                for attr in [
                    "nonlocal_means",
                    "normalization_factors",
                    "nonlocal_std",
                    "nonlocal_speckle",
                ]
            }
        except ValueError as e:
            raise ResultCombinationError(f"Error combining NLM results: {str(e)}")

        return class_(
            **combined_arrays,
            **BaseResult.combine(results).__dict__,
            search_window_size=results[0].search_window_size,
            filter_strength=results[0].filter_strength,
            last_similarity_map=results[-1].last_similarity_map,
        )

    @classmethod
    def empty_result(class_: Type["NLMResult"]) -> "NLMResult":
        return class_(
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
    def merge(
        class_: Type["NLMResult"], new_result: "NLMResult", existing_result: "NLMResult"
    ) -> "NLMResult":
        merged_arrays: Dict[str, np.ndarray] = {
            attr: np.maximum(getattr(new_result, attr), getattr(existing_result, attr))
            for attr in [
                "nonlocal_means",
                "normalization_factors",
                "nonlocal_std",
                "nonlocal_speckle",
            ]
        }

        return class_(
            **merged_arrays,
            processing_end_coord=max(
                new_result.processing_end_coord, existing_result.processing_end_coord
            ),
            kernel_size=new_result.kernel_size,
            pixels_processed=max(
                new_result.pixels_processed, existing_result.pixels_processed
            ),
            image_dimensions=new_result.image_dimensions,
            search_window_size=new_result.search_window_size,
            filter_strength=new_result.filter_strength,
            last_similarity_map=new_result.last_similarity_map,
        )

@dataclass
class SpeckleResult(BaseResult):
    mean_filter: np.ndarray
    std_dev_filter: np.ndarray
    speckle_contrast_filter: np.ndarray

    @staticmethod
    def get_filter_options() -> List[str]:
        return ["Mean Filter", "Std Dev Filter", "Speckle Contrast"]

    @property
    def filter_data(self) -> Dict[str, np.ndarray]:
        return {
            "Mean Filter": self.mean_filter,
            "Std Dev Filter": self.std_dev_filter,
            "Speckle Contrast": self.speckle_contrast_filter,
        }

    @classmethod
    def combine(
        class_: Type["SpeckleResult"], results: List["SpeckleResult"]
    ) -> "SpeckleResult":
        if not results:
            raise ResultCombinationError("No Speckle results provided for combination")

        try:
            combined_arrays: Dict[str, np.ndarray] = {
                attr: np.maximum.reduce([getattr(r, attr) for r in results])
                for attr in ["mean_filter", "std_dev_filter", "speckle_contrast_filter"]
            }
        except ValueError as e:
            raise ResultCombinationError(f"Error combining Speckle results: {str(e)}")

        return class_(
            **combined_arrays,
            **BaseResult.combine(results).__dict__,
        )

    @classmethod
    def merge(
        class_: Type["SpeckleResult"],
        new_result: "SpeckleResult",
        existing_result: "SpeckleResult",
    ) -> "SpeckleResult":
        merged_arrays: Dict[str, np.ndarray] = {
            attr: np.maximum(getattr(new_result, attr), getattr(existing_result, attr))
            for attr in ["mean_filter", "std_dev_filter", "speckle_contrast_filter"]
        }

        return class_(
            **merged_arrays,
            processing_end_coord=max(
                new_result.processing_end_coord, existing_result.processing_end_coord
            ),
            kernel_size=new_result.kernel_size,
            pixels_processed=max(
                new_result.pixels_processed, existing_result.pixels_processed
            ),
            image_dimensions=new_result.image_dimensions,
        )

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
    nlm_filter_strength: float = field(default=0.0)

    def __post_init__(self):
        # Ensure pixels_processed is consistent across results
        self.pixels_processed = max(self.pixels_processed, self.nlm_result.pixels_processed, self.speckle_result.pixels_processed)
        self.nlm_result.pixels_processed = self.pixels_processed
        self.speckle_result.pixels_processed = self.pixels_processed

    def add_image(self, name: str, image: np.ndarray) -> None:
        self.additional_images[name] = image

    @property
    def all_images(self) -> Dict[str, np.ndarray]:
        images: Dict[str, np.ndarray] = {}
        for prefix, result in [
            ("NLM", self.nlm_result),
            ("Speckle", self.speckle_result),
        ]:
            images.update({f"{prefix} {k}": v for k, v in result.filter_data.items()})
        return {**images, **self.additional_images}

    @property
    def filter_options(self) -> List[str]:
        return [
            f"{prefix} {option}"
            for prefix, result in [
                ("NLM", self.nlm_result),
                ("Speckle", self.speckle_result),
            ]
            for option in result.get_filter_options()
        ] + list(self.additional_images.keys())

    @property
    def filter_data(self) -> Dict[str, Any]:
        return self.all_images

    @classmethod
    def combine(
        class_: Type["NLSpeckleResult"],
        nlm_results: List[NLMResult],
        speckle_results: List[SpeckleResult],
        kernel_size: int,
        pixels_processed: int,
        image_dimensions: Tuple[int, int],
        nlm_search_window_size: int,
        nlm_filter_strength: float,
    ) -> "NLSpeckleResult":
        if not nlm_results or not speckle_results:
            raise ResultCombinationError(
                "Both NLM and Speckle results must be provided for combination"
            )

        combined_nlm = NLMResult.combine(nlm_results)
        combined_speckle = SpeckleResult.combine(speckle_results)

        return class_(
            nlm_result=combined_nlm,
            speckle_result=combined_speckle,
            processing_end_coord=class_._get_max_processing_end_coord(
                nlm_results, speckle_results
            ),
            kernel_size=kernel_size,
            pixels_processed=pixels_processed,  # Use the provided pixels_processed
            image_dimensions=image_dimensions,
            nlm_search_window_size=nlm_search_window_size,
            nlm_filter_strength=nlm_filter_strength,
        )

    @staticmethod
    def _get_max_processing_end_coord(
        nlm_results: List[NLMResult], speckle_results: List[SpeckleResult]
    ) -> Tuple[int, int]:
        all_coords: List[Tuple[int, int]] = [
            result.processing_end_coord for result in nlm_results + speckle_results
        ]
        return max(
            all_coords, key=lambda coord: coord[0] * 10000 + coord[1]
        )  # Prioritize y-coordinate

# --- Utility Functions ---
