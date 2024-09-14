from dataclasses import dataclass
from typing import Optional
import numpy as np

@dataclass
class ProcessingDetails:
    """
    Holds details about image processing parameters.
    
    Attributes:
    - image_height: Total height of the image
    - image_width: Total width of the image
    - start_x: X-coordinate of the first pixel to process
    - start_y: Y-coordinate of the first pixel to process
    - end_x: X-coordinate of the last pixel to process
    - end_y: Y-coordinate of the last pixel to process
    - pixels_to_process: Total number of pixels to process
    - valid_height: Height of the valid processing area
    - valid_width: Width of the valid processing area
    """
    image_height: int
    image_width: int
    start_x: int
    start_y: int
    end_x: int
    end_y: int
    pixels_to_process: int
    valid_height: int
    valid_width: int

def calculate_processing_details(image: np.ndarray, kernel_size: int, max_pixels: Optional[int] = None) -> ProcessingDetails:
    """
    Calculate processing details for an image.
    
    Args:
    - image: Input image as a numpy array
    - kernel_size: Size of the kernel for processing
    - max_pixels: Maximum number of pixels to process (optional)
    
    Returns:
    - ProcessingDetails object with calculated parameters
    """
    image_height, image_width = image.shape[:2]
    half_kernel = kernel_size // 2
    valid_height, valid_width = image_height - kernel_size + 1, image_width - kernel_size + 1
    pixels_to_process = min(valid_height * valid_width, max_pixels or float('inf'))
    last_pixel = pixels_to_process - 1
    end_y, end_x = divmod(last_pixel, valid_width)
    end_y, end_x = end_y + half_kernel, end_x + half_kernel

    return ProcessingDetails(
        image_height=image_height, 
        image_width=image_width,
        start_x=half_kernel, 
        start_y=half_kernel,
        end_x=end_x, 
        end_y=end_y,
        pixels_to_process=pixels_to_process,
        valid_height=valid_height, 
        valid_width=valid_width
    )