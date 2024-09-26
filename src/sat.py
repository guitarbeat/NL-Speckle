import numpy as np

def calculate_summed_area_table(image):
    """Calculate the Summed-Area Table (SAT) for an image."""
    return np.cumsum(np.cumsum(image, axis=0), axis=1).astype(np.float64)

def get_area_sum(sat, top, left, bottom, right):
    """Get the sum of a rectangular area using a Summed-Area Table."""
    top, left, bottom, right = max(0, top), max(0, left), min(sat.shape[0] - 1, bottom), min(sat.shape[1] - 1, right)
    result = sat[bottom, right]
    if top > 0:
        result -= sat[top-1, right]
    if left > 0:
        result -= sat[bottom, left-1]
    if top > 0 and left > 0:
        result += sat[top-1, left-1]
    return result