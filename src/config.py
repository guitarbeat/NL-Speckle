
# Constants for Image Visualization
SPECKLE_CONTRAST = "Speckle Contrast"
ORIGINAL_IMAGE = "Original Image"
NON_LOCAL_MEANS = "Non-Local Means"

DEFAULT_SPECKLE_VIEW = [SPECKLE_CONTRAST, ORIGINAL_IMAGE]
DEFAULT_NLM_VIEW = [NON_LOCAL_MEANS, ORIGINAL_IMAGE]

AVAILABLE_COLOR_MAPS = [
    "gray",
    "plasma",
    "inferno",
    "magma",
    "pink",
    "hot",
    "cool",
    "YlOrRd",
]

PRELOADED_IMAGE_PATHS = {
    "image50.png": "media/image50.png",
    "spatial.tif": "media/spatial.tif",
    "logo.jpg": "media/logo.jpg",
}

# App configuration
APP_CONFIG = {
    "page_title": "Speckle Contrast Visualization",
    "layout": "wide",
    "page_icon": "favicon.png",
    "initial_sidebar_state": "expanded",
}

# Default values
DEFAULT_KERNEL_SIZE = 3
DEFAULT_SEARCH_WINDOW_SIZE = 21
DEFAULT_FILTER_STRENGTH = 10.0
