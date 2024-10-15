import cv2
import numpy as np
from typing import Tuple
import matplotlib.pyplot as plt
from enum import Enum, auto
import pixel_visions.utils as u
from typing import Dict, Tuple
"""
Interpolation Module

This module provides functionality for resizing images using different interpolation techniques such as:
- Nearest neighbor interpolation
- Bilinear interpolation
- Bicubic interpolation

The module allows reducing and restoring image resolution, comparing the effects of these methods, and performing
image subtraction to observe the differences between the original and restored images.

Functions:
    - resize_image: Resizes an image based on the specified scale and interpolation method.
    - nearest_neighbor_interpolation: Resizes an image using nearest neighbor interpolation.
    - bilinear_interpolation: Resizes an image using bilinear interpolation.
    - bicubic_interpolation: Resizes an image using bicubic interpolation.
    - compare_interpolation_methods: Reduces an image's resolution using the three interpolation methods and displays the results.
    - subtract_images: Performs image subtraction between the original and interpolated images.

This module is part of the Pixel Visions project, focused on image processing and analysis tasks.
"""

class InterpolationMethod(Enum):
    NEAREST_NEIGHBOR = cv2.INTER_NEAREST
    BILINEAR = cv2.INTER_LINEAR
    BICUBIC = cv2.INTER_CUBIC


def resize_image(image: np.ndarray, scale: float, interpolation: InterpolationMethod = InterpolationMethod.NEAREST_NEIGHBOR) -> np.ndarray:
    """
    Resizes an image using the given scale and interpolation method.

    Parameters:
        image (np.ndarray): The input image to resize.
        scale (float): The scaling factor (e.g., 0.1 for 1/10th size).
        interpolation (int): Interpolation method (e.g., NEAREST_NEIGHBOR, BILINEAR, BICUBIC).

    Returns:
        np.ndarray: The resized image.
    """

    if image is None:
        raise ValueError(f"The input image is empty") 
    
    if scale <= 0:
        raise ValueError("Scale must be a positive, non-zero value.")
    
    new_height = int(image.shape[0] * scale)
    new_width = int(image.shape[1] * scale)

    # Check if dimensions are valid
    if new_width <= 0 or new_height <= 0:
        raise ValueError("New dimensions must be positive, non-zero values.")

    resized_image = cv2.resize(image, (new_width, new_height), interpolation.value)

    return resized_image

def compare_interpolations(image:np.ndarray, scale: float) -> Dict[str, np.ndarray]:
    """
    Resizes an image using nearest neighbor, bilinear, and bicubic interpolation methods and displays the results.
    Also performs image subtraction between the original and restored images for comparison.

    Parameters:
        - image (np.ndarray): The input image.
        - scale (float): The scaling factor for reducing image resolution.    
    Returns:
        - Dict[str, np.ndarray] : Dictionary of resized images.
    """
    if image is None:
        raise ValueError(f"The input image is empty") 
    
    if scale <= 0:
        raise ValueError("Scale must be a positive, non-zero value.")

    resized_images = {}

    for interpolation_type in InterpolationMethod:
        resized_images[str(interpolation_type.name)] = resize_image(image, scale, interpolation_type)

    return resized_images
    
def restore_image(image: np.ndarray, original_size: tuple[int, int], interpolation: InterpolationMethod = InterpolationMethod.NEAREST_NEIGHBOR) -> np.ndarray:
    """
    Restores an image to its original size using the specified interpolation method.
    
    Parameters:
        image (np.ndarray): The input image to be restored.
        original_size (tuple[int, int]): The original size (width, height) to restore the image to.
        interpolation (InterpolationMethod): The interpolation method to use for resizing.
    
    Returns:
        np.ndarray: The restored image at the original size.
    """
    
    if image is None:
        raise ValueError(f"The input image is empty") 
    
     # Validate input parameters
    if not isinstance(image, np.ndarray):
        raise TypeError("Image must be a numpy array.")
    
    if not isinstance(original_size, tuple) or len(original_size) != 2:
        raise ValueError("original_size must be a tuple of (width, height).")
    
    
    # Perform image restoration
    restored_image = cv2.resize(image, original_size, interpolation=interpolation.value)

    return restored_image
