import cv2
import numpy as np
from typing import Tuple
import matplotlib.pyplot as plt
from enum import Enum, auto
import pixel_visions.utils as u
import pixel_visions.image_transformations as t

from typing import Dict, Tuple, List, Any, Callable

def histogram_based_segmentation_using_otsu(grayscale_image: np.ndarray,                                  
                                 outputfile:str,
                                 scale_factor:int = 2,
                                 ranges: List[Tuple[int, int]] = None, 
                                 use_otsu: bool = True) -> Dict[str, np.ndarray]:
    """
    Perform histogram-based segmentation on a grayscale image using thresholding.

    Parameters:
        image (np.ndarray): Input grayscale image as a 2D NumPy array.
        use_otsu (bool): Flag to determine whether to use Otsu's method for automatic threshold selection. Default is True.
                         If False, then use user defined ranges.
        ranges (List[Tuple[int, int]]): List of intensity ranges (tuples) for segmentation.
                                        Each tuple defines the (min, max) intensity for an object. Default is None
    Returns:
        Dict[str, np.ndarray]   : A dictionary containing the segmented image (binary mask) and the histogram of the image.
    """
    if grayscale_image is None:
        raise ValueError("The input image is empty")
    histograms_segments = {}
    # Calculate the histogram of the image.
    histogram = t.compute_histogram(grayscale_image)
    combined_binary_image = np.zeros(grayscale_image.shape, dtype=np.uint8)
    if use_otsu:
        _, segmented_image = cv2.threshold(grayscale_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # Scale the segmented image for better visualization
        height, width = grayscale_image.shape
        scaled_image = cv2.resize(segmented_image, (int(width * scale_factor), int(height * scale_factor)), interpolation=cv2.INTER_NEAREST)
        histograms_segments["Otsu_Segmented_Image"] = scaled_image
    else: # apply the range specified by user.
        for min_val, max_val in ranges:
            # Threshold the image to isolate the pixels in the given range
            image_range = cv2.inRange(grayscale_image, min_val, max_val)
            height, width = grayscale_image.shape
            scaled_image = cv2.resize(image_range, (int(width * scale_factor), int(height * scale_factor)), interpolation=cv2.INTER_NEAREST)
            histograms_segments[f"Segmented_Image_Range_{min_val}_{max_val}"] = scaled_image
            combined_binary_image = cv2.add(combined_binary_image, image_range) 
        _, combined_binary_image = cv2.threshold(combined_binary_image, 0, 255, cv2.THRESH_BINARY)
        histograms_segments["Combined Binary Image"] = combined_binary_image
    return histograms_segments


def histogram_based_segmentation(grayscale_image: np.ndarray,                                  
                                 outputfile:str,
                                 scale_factor:int = 2,
                                 ranges: List[Tuple[int, int]] = None, 
                                 ) -> Dict[str, np.ndarray]:
    """
    Perform histogram-based segmentation on a grayscale image using thresholding and compute the sample mean for each segment.    
    Parameters:
        image (np.ndarray): Input grayscale image as a 2D NumPy array.
        use_otsu (bool): Flag to determine whether to use Otsu's method for automatic threshold selection. Default is True.
                         If False, then use user defined ranges.
        ranges (List[Tuple[int, int]]): List of intensity ranges (tuples) for segmentation.
                                        Each tuple defines the (min, max) intensity for an object. Default is None
    Returns:
        Dict[str, np.ndarray]   : A dictionary containing the segmented image (binary mask), the reconstructed image, and the histogram of the image.

    """
    if grayscale_image is None:
        raise ValueError("The input image is empty")

    histograms_segments = {}
    # Calculate the histogram of the image. 
    histogram = t.compute_histogram(grayscale_image)
    # Create an empty ndarray for the reconstrcuted image
    reconstructed_image = np.zeros(grayscale_image.shape, dtype=np.uint8)
    for i, (min_val, max_val) in enumerate(ranges):
        image_range = cv2.inRange(grayscale_image, min_val, max_val)
        # Scale and save the image segment
        height, width = grayscale_image.shape
        scaled_image = cv2.resize(image_range, (int(width * scale_factor), int(height * scale_factor)), interpolation=cv2.INTER_NEAREST)
        histograms_segments[f"Segmented_Image_Range_{min_val}_{max_val}"] = scaled_image
        # Compute the mean intensity in the image range. 
        mean_intensity = np.sum(grayscale_image * (image_range > 0)) / np.sum(image_range > 0)
        # reconstruct the image
        reconstructed_image[image_range > 0] = mean_intensity
    # Scale and save the reconstructed image.
    scaled_reconstructed_image = cv2.resize(reconstructed_image, (int(width * scale_factor), int(height * scale_factor)), interpolation=cv2.INTER_NEAREST)
    histograms_segments["Reconstructed Image"] = scaled_reconstructed_image

    return histograms_segments


