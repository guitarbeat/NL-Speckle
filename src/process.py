import numpy as np
from src.classes import NLMResult, SpeckleResult
from src.speckle import apply_speckle_contrast
from src.nlm import apply_nlm_to_image


# --- Processing Functions ---

def process_speckle(image, kernel_size, pixels_to_process, start_pixel=0):
    if not isinstance(image, np.ndarray):
        image = np.array(image)

    height, width = image.shape
    half_kernel = kernel_size // 2
    valid_height, valid_width = height - kernel_size + 1, width - kernel_size + 1
    pixels_to_process = min(pixels_to_process, valid_height * valid_width)

    # Calculate starting coordinates
    start_y, start_x = divmod(start_pixel, valid_width)
    start_y += half_kernel
    start_x += half_kernel

    mean_filter, std_dev_filter, sc_filter = apply_speckle_contrast(
        image, kernel_size, pixels_to_process, (start_x, start_y)
    )

    # Calculate processing end coordinates
    end_pixel = start_pixel + pixels_to_process
    end_y, end_x = divmod(end_pixel - 1, valid_width)
    end_y, end_x = end_y + half_kernel, end_x + half_kernel
    processing_end = (min(end_x, width - 1), min(end_y, height - 1))

    return SpeckleResult(
        mean_filter=mean_filter,
        std_dev_filter=std_dev_filter,
        speckle_contrast_filter=sc_filter,
        processing_end_coord=processing_end,
        kernel_size=kernel_size,
        pixels_processed=pixels_to_process,
        image_dimensions=(height, width),
    )

def process_nlm(
    image: np.ndarray,
    kernel_size: int,
    pixels_to_process: int,
    search_window_size: int = 21,
    h: float = 10.0,
    start_pixel: int = 0,
) -> NLMResult:
    height, width = image.shape
    half_kernel = kernel_size // 2
    valid_height, valid_width = height - kernel_size + 1, width - kernel_size + 1
    total_valid_pixels = valid_height * valid_width

    # Ensure we don't process beyond the valid pixels
    end_pixel = min(start_pixel + pixels_to_process, total_valid_pixels)
    pixels_to_process = end_pixel - start_pixel

    # Calculate starting coordinates
    start_y, start_x = divmod(start_pixel, valid_width)
    start_y += half_kernel
    start_x += half_kernel

    NLM_image, NLstd_image, NLSC_xy_image, C_xy_image, last_similarity_map = (
        apply_nlm_to_image(
            np.asarray(image, dtype=np.float32),
            kernel_size,
            search_window_size,
            h,
            pixels_to_process,
            (start_x, start_y),
        )
    )

    # Calculate processing end coordinates
    end_y, end_x = divmod(end_pixel - 1, valid_width)
    end_y, end_x = end_y + half_kernel, end_x + half_kernel
    processing_end = (min(end_x, width - 1), min(end_y, height - 1))

    return NLMResult(
        nonlocal_means=NLM_image,
        normalization_factors=C_xy_image,
        nonlocal_std=NLstd_image,
        nonlocal_speckle=NLSC_xy_image,
        processing_end_coord=processing_end,
        kernel_size=kernel_size,
        pixels_processed=pixels_to_process,
        image_dimensions=(height, width),
        search_window_size=search_window_size,
        filter_strength=h,
        last_similarity_map=last_similarity_map,
    )