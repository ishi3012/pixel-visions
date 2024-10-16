import cv2
import numpy as np
from typing import Tuple
import matplotlib.pyplot as plt
from enum import Enum, auto
import pixel_visions.utils as u
from typing import Dict, Tuple

class Normalization(Enum):
    MINMAX = auto()
    REVERSED = auto()
    LARGELOG = auto()
    GAMMA    = auto()
    EXTREME  = auto()
    RANDOM  = auto()

def negative_image(image: np.ndarray) -> np.ndarray:
    """
    Creates the negative of the given image. 

    Parameters:
        - image (np.ndarray): The input image to resize.
    
    Returns:
        np.ndarray: The negative image.

    """
    if image is None:
        raise ValueError(f"The input image is empty") 
    
    # Color Image
    if len(image.shape) > 2:
        channel_pixels = []
        for channel in range(image.shape[2]):
            negative_channel = 255 - image[:,:,channel]

            channel_pixels.append(negative_channel)
            # print(f"Channel = {channel}")
            # print(f"Image Channel = {image[:,:,channel]}")

        negative_image = np.stack(channel_pixels, axis = -1)
    else:
        negative_image = 255 - image
    return negative_image

def minmax_normalization(image: np.ndarray, 
                        lower_intensity: int = 0, 
                        upper_intensity: int = 255) -> np.ndarray:      
    """
    Performs min-max contrast stretching of the input image using lower_intensity 
    and upper_intensity as the bounds. 
    Parameters:
        - image (np.ndarray): The input image to perform contrast stretching.
        - lower_intensity (int): The lower bound of the desired output intensity range. Default is 0.
        - upper_intensity (int): The upper bound of the desired output intensity range. Default is 255.
    Returns:
        - np.ndarray: Returns stretched image.
    """
    if image is None:
        raise ValueError("The input image is empty") 

    # Check if the image is grayscale or colored
    if len(image.shape) == 2:  # Grayscale image
        return cv2.normalize(image, None, lower_intensity, upper_intensity, cv2.NORM_MINMAX)
    elif len(image.shape) == 3:  # Color image
        # Normalize each channel separately
        channels = []
        for channel in range(image.shape[2]):
            channel_normalized = cv2.normalize(image[:, :, channel], None, lower_intensity, upper_intensity, cv2.NORM_MINMAX)
            channels.append(channel_normalized)
        # Merge normalized channels back into a color image
        return cv2.merge(channels)

def large_log(image: np.ndarray, factor: int = 2) -> np.ndarray:  
    """
    Applies an exaggerated logarithmic transformation to the input image, 
    enhancing the contrast of dark regions by overly brightening them.

    Parameters:
        - image (np.ndarray): The input image to perform contrast stretching.
        - factor (int)      : The integer value to indicate the factor of log transformation. Default is 2.
    Returns:
        - Dict[str, np.ndarray] : Image with exaggerated logarithmic transformation.
    
    """
    if image is None:
        raise ValueError("The input image is empty") 
       
    #prevent divide by zero error
    epsilon = 1e-9

    # Apply normalization
    
    c: float = 255 / np.log(1 + np.max(image)/factor)
    log_stretched = np.array(c * (np.log(1 + image + epsilon) ** 3), dtype='uint8')    
    return log_stretched

def gamma_correction(image: np.ndarray, gamma: float = 0.01) -> np.ndarray:
    """
    Performs gamma correction on the input image to adjust contrast based 
    on the specified gamma value.

    Parameters:
        - image (np.ndarray): The input image to perform contrast stretching.
        - gamma (float): The gamma value for the correction. Values less than 1 
                   will brighten the image, and values greater than 1 will 
                   darken it.

    Returns:
    numpy.ndarray: Image after gamma correction.
    """
   
    if image is None:
        raise ValueError("The input image is empty") 
      
    gamma_stretched = np.array(255 * (image / 255) ** gamma, dtype='uint8')    
    return gamma_stretched

def random_stretch(image: np.ndarray) -> np.ndarray:
    """
    Applies random contrast stretching by selectively boosting or 
    suppressing pixel intensities based on a random mask.
    Parameters:
        - image (np.ndarray): The input image to perform contrast stretching.
    Returns:
        - numpy.ndarray: Image with random contrast stretching.
    """
    
    if image is None:
        raise ValueError("The input image is empty") 
   
    # create a mask with random values.
    mask = np.random.choice([0, 1], size=image.shape, p=[0.50, 0.50]).astype(np.uint8)        
    # randomly masking image 
    random_stretched = np.clip(np.where(mask == 1, image * 2, image // 2), 0,  255).astype(np.uint8)

    return random_stretched

def reversed_stretching(image: np.ndarray, 
                        lower_intensity: int = 255, 
                        upper_intensity: int = 0) -> np.ndarray:  
    """
    Applies random contrast stretching by selectively boosting or 
    suppressing pixel intensities based on a random mask.
    Parameters:
        - image (np.ndarray): The input image to perform contrast stretching.
    Returns:
        - numpy.ndarray: Image with random contrast stretching.
    """
    if image is None:
        raise ValueError("The input image is empty") 
       
    return minmax_normalization(image, lower_intensity, upper_intensity)

def compare_contrast_stretching(image:np.ndarray,
                                lower_intensity: int = 0, 
                                upper_intensity: int = 255,
                                factor: int = 2, 
                                gamma: float = 0.01, 
                                extremeLow: float = 51,
                                extremeHigh: float = 90) -> np.ndarray:   
    """
    Applies various stretching methods on the given image like MINMAX, REVERSED,LARGELOG,GAMMA, RANDOM, EXTREME
    
    Parameters:
        - image (np.ndarray): The input image to perform contrast stretching.
    Returns:
        - Dict[str,numpy.ndarray]: Dictionary of transformed images.
    """
    if image is None:
        raise ValueError("The input image is empty") 
    
    normalized_images = {}
    
    if upper_intensity > 255 or lower_intensity < 0:
        raise ValueError("The Lower and Upper intensity values are not in range 0 and 255.") 
       
    for normalize in Normalization:
        if normalize == Normalization.MINMAX:  
            normalized_images["minmax"] = minmax_normalization(image, lower_intensity, upper_intensity)

        if normalize == Normalization.REVERSED:  
            normalized_images["reversed"] = reversed_stretching(image,upper_intensity,lower_intensity)

        if normalize == Normalization.LARGELOG:  
            normalized_images["large_log"] = large_log(image,factor)

        # if normalize == Normalization.GAMMA:  
        #     normalized_images["gamma"] = reversed_stretching(image,gamma)

        if normalize == Normalization.EXTREME:  
            normalized_images["extreme"] = minmax_normalization(image, extremeLow, extremeHigh)

        if normalize == Normalization.RANDOM:  
            normalized_images["random"] = random_stretch(image)

    return normalized_images

def rotate_image_by_pixel(image:np.ndarray, angle_degrees :int = 30) -> np.ndarray:
    """
    Rotates the given image by the theta value specified. 
    R(x,y)=(x⋅cos(θ)−y⋅sin(θ),x⋅sin(θ)+y⋅cos(θ))

    Parameters:
        - image (np.ndarray): The input image to rotate.
        - angle_degrees (int): The angle in degrees by which to rotate the image.
    Returns:
        - numpy.ndarray         : Rotated image
    """
    if image is None:
        raise ValueError("The input image is empty.")

    # Convert angle from degrees to radians
    theta = np.radians(angle_degrees)
    
    # Get the dimensions of the image
    h, w = image.shape[:2]
    
    # Calculate the center of the image
    center_x, center_y = w // 2, h // 2
    
    # Calculate new width and height
    new_w = int(abs(w * np.cos(theta)) + abs(h * np.sin(theta)))
    new_h = int(abs(h * np.cos(theta)) + abs(w * np.sin(theta)))

    # Create an output image with the new size
    rotated_image = np.zeros((new_h, new_w, image.shape[2]), dtype=image.dtype)
    
    # Calculate the new center
    new_center_x, new_center_y = new_w // 2, new_h // 2

    # Apply rotation to each pixel
    for y in range(h):
        for x in range(w):
            # Translate pixel position to center (for rotation)
            x_centered = x - center_x
            y_centered = y - center_y
            
            # Apply the rotation matrix
            new_x = int(x_centered * np.cos(theta) - y_centered * np.sin(theta)) + new_center_x
            new_y = int(x_centered * np.sin(theta) + y_centered * np.cos(theta)) + new_center_y
            
            # Check if new coordinates are within image bounds
            if 0 <= new_x < new_w and 0 <= new_y < new_h:
                rotated_image[new_y, new_x] = image[y, x]

    return rotated_image

def rotate_image(image: np.ndarray, angle_degrees: int) -> np.ndarray:
    """
    Rotates the given image by the specified angle using OpenCV's built wrapAffine function.
    
    Parameters:
        - image (np.ndarray): The input image to rotate.
        - angle_degrees (int): The angle in degrees to rotate the image.
    
    Returns:
        np.ndarray: The rotated image.
    """
    if image is None:
        raise ValueError("The input image is empty")
    
    # Get the image dimensions
    (h, w) = image.shape[:2]
    
    # Calculate the center of the image
    center = (w // 2, h // 2)
    
    # Create the rotation matrix
    M = cv2.getRotationMatrix2D(center, angle_degrees, 1.0)
    
    # Perform the rotation
    rotated_image = cv2.warpAffine(image, M, (w, h))
    
    return rotated_image
