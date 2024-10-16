"""
utils.py

This module provides utility functions for common image processing tasks. 
It supports loading, saving, resizing, converting to grayscale, and plotting 
images, along with basic image manipulations. These utility functions are 
designed to simplify and streamline the image processing workflows in the 
Pixel-Visions project.

"""
import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from enum import Enum, auto
from typing import Dict, Tuple

class ImageLoadType(Enum):
    COLOR = cv2.IMREAD_COLOR
    GRAYSCALE = cv2.IMREAD_GRAYSCALE
    UNCHANGED = cv2.IMREAD_UNCHANGED

def load_image(image_path:str, image_load_type: ImageLoadType = ImageLoadType.UNCHANGED) -> np.ndarray:
    """
    Loads the image from the image file path and returns a numpy array represenation of the image. 

    Parameters:
        - image_path(str)                   : The path to the image file.
        - image_load_type(ImageLoadType)    : The type of image loading (COLOR, GRAYSCALE, UNCHANGED)
    
    Returns:
        - np.ndarray        : The numpy array representation of the image. 
    """
    image = cv2.imread(image_path, image_load_type.value)

    if image is None:
        raise FileNotFoundError(f"Image not found at path: {image_path}")
    return image

def save_image(image: np.ndarray, output_path: str) -> None:
    """
    Saves an image (NumPy array) to the specified file path.
    Parameters:
        - image(np.ndarray)     : The image to save.
        - output_path(str)        : The image file path to save the given image. 
    """
    try:
        success = cv2.imwrite(output_path, image)

        if not success:
            raise IOError(f"Failed to write the image to {output_path}. Check the path and permissions.")
        print(f"Image successfully saved to {output_path}")

    except Exception as e:
        print(f"An error occurred while saving the image: {e}")

def plot_images(images: Dict[str, np.ndarray], outputfile: str = None, title: str = "", axis: str = "off") -> None:
    """
    Display the images with the given title.

    Parameters: 
        - images (Dict[str, np.ndarray]) : Dictionary of images to plot
        - title (str)                    : Text to display as title of the plot
        - outputfile (str)               : The output file path. 
        - axis (str)                     : String to display the axis in the plot. Default is OFF.
    """

    if images is None or not images:
        raise ValueError("ERROR: Images not available or empty.")

    for key, image in images.items():
        if not isinstance(image, np.ndarray):
            raise ValueError(f"ERROR: Image '{key}' is not a valid NumPy array.")

    num_cols = 2 if len(images) > 1 else 1
    num_rows = (len(images) + num_cols - 1) // num_cols

    plt.figure(figsize=(20 , 7 * num_rows))

    for i, (key, image) in enumerate(images.items()):
        plt.subplot(num_rows, num_cols, i + 1)

        # Check for binary images (values 0 or 1)
        if image.ndim == 2:  # Grayscale or binary image
            if image.max() <= 1:  # Binary image
                image = (image * 255).astype(np.uint8)  # Convert to 0-255 range
            plt.imshow(image, cmap="gray")
        elif image.ndim == 3:  # Colored image
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        else:
            raise ValueError(f"ERROR: Image '{key}' has unsupported shape: {image.shape}")

        plt.axis(axis)
        plt.title(key, fontsize=10)

    plt.suptitle(title, fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    if outputfile:
        plt.savefig(outputfile)
    plt.show()

def display_image_statistics(image: np.ndarray) -> None:
    """
    Calculate statistics for the given image.

    Args:
        image (np.ndarray): The input image as a NumPy array.

    Returns:
        Dict[str, Tuple[float, float]]: A dictionary containing mean, standard deviation, 
                                         size, maximum pixel value, and mean pixel value.
    """

    if image is None or image.size == 0:
        raise ValueError(f"The input image is empty") 
    
    #Display image size
    height, width, channels = get_image_size(image)
    print(f"Image dimensions: {width} x {height} (Width x Height)")
    print(f"Total number of pixels: {height * width}")
    print(f"Number of color channels: {channels}")

    # Display pixel statistics
    stats = get_pixel_statistics(image)

    if len(stats) > 1:
        print(f"Statistics for the Colored iamge.")
    else:
        print(f"Statistics for the Grayscale iamge.")
    
    for idx,stat in enumerate(stats):
        if len(stats) > 1:
            print(f"~~ Stat for channel : {idx}")
            print(f"\t Min pixel value : {stat[0]}")
            print(f"\t Max pixel value : {stat[1]}")
            print(f"\t Mean pixel value : {stat[2]}")
            print(f"\t Standard deviation pixel value : {stat[3]}")
        else:
            print(f"Min pixel value : {stat[0]}")
            print(f"Max pixel value : {stat[1]}")
            print(f"Mean pixel value : {stat[2]}")
            print(f"Standard deviation pixel value : {stat[3]}")
    
def get_image_size(image: np.ndarray) -> Tuple[int, int, int]:
    """
        Get the dimensions of the given image.

        Parameters:
            image (np.ndarray): The input image as a NumPy array.

        Returns:
            Tuple[int, int, int]: A tuple containing the height, width, and number of color channels of the image.
    """

    if image is None or image.size == 0:
        raise ValueError("The input image is empty.")
    #Colored Image
    if len(image.shape) > 2:
        height, width, channels = image.shape

    # Grayscale Image
    else:
        height, width = image.shape
        channels = 0    
    return height, width, channels

def get_pixel_statistics(image: np.ndarray) -> Tuple[Tuple[float, float, float, float],...]:
    """
        Get the dimensions of the given image.

        Parameters:
            image (np.ndarray): The input image as a NumPy array.
        Returns:
            Tuple[int, int, int]: A tuple containing the min pixel value, max pixel value, mean pixel value, standard deviation.
    """
    if image is None or image.size == 0:
        raise ValueError("The input image is empty.")
    
    stats = []
    
    #Colored Image
    if len(image.shape) > 2:
        for channel in range(image.shape[2]):
            pixels = image[:,:,channel]
            min_pixel = pixels.min()
            max_pixel = pixels.max()
            mean_pixel = pixels.mean()
            std_deviation_pixel = pixels.std()
            stats.append([min_pixel, max_pixel, mean_pixel,std_deviation_pixel])  
            # print(f"~~~~~~~~~ Stats for channel {channel} are {min_pixel}, {max_pixel}, {mean_pixel},{std_deviation_pixel}") 

    # Grayscale Image
    else:
        min_pixel = image.min()
        max_pixel = image.max()
        mean_pixel = image.mean()
        std_deviation_pixel = image.std()
        stats.append([min_pixel, max_pixel, mean_pixel, std_deviation_pixel])    
    return stats
    
def threshold_image(image: np.ndarray, threshold: Tuple[float, ...]) -> np.ndarray:
    """
    Creates a binary image based on the provided threshold values for each channel. 
    Pixels with intensity less than or equal to the corresponding threshold are set to 0, 
    while pixels with intensity greater than the threshold are set to 1.

    Parameters:
        image (np.ndarray): The input image as a NumPy array.
        threshold (Tuple[float, ...]): A tuple of threshold values for each channel of the image.

    Returns:
        np.ndarray: The resultant binary image, where pixel intensities are either 0 or 1.
    """

    if image is None or image.size == 0:
        raise ValueError("The input image is empty.")
    
    if threshold is None:
        raise ValueError("The threshold value is None (not provided).")

    if len(threshold) != (3 if len(image.shape) == 3 else 1):
        raise ValueError(f"Expected threshold length {3 if len(image.shape) == 3 else 1}, but got {len(threshold)}.")
    
    # Colored Image
    if len(image.shape) > 2:
        binary_image = np.zeros(image.shape[:2], dtype=np.uint8)
        for i in range(len(image.shape)):
            binary_image += np.where(image[:, :, i] > threshold[i], 1, 0).astype(np.uint8)
    #Grayscale image
    else:
        binary_image = np.where(image<= threshold[0], 0, 1).astype(np.uint8)
    
    return binary_image

def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    Normalize an image to the range [0, 1].

    Parameters:
        image (np.ndarray): The input image as a NumPy array.

    Returns:
        np.ndarray: The normalized image.
    """
    return image.astype(np.float32) / 255.0

def subtract_images(image1: np.ndarray, image2: np.ndarray) -> np.ndarray:
    """
    Subtracts two images by calculating the absolute difference between them.

    Args:
        original (np.ndarray): The original image.
        modified (np.ndarray): The image that has been modified (e.g., resized or restored).

    Returns:
        np.ndarray: The result of subtracting the modified image from the original image, showing pixel differences.
    """
    if image1 is None or image2 is None:
        raise ValueError("Image 1 or Image 2 doesnt exists")
    
    if image1.shape != image2.shape:
        raise ValueError("The two images must have the same dimensions for subtraction.")

    # Use OpenCV's absdiff to calculate the absolute pixel-wise difference between the images
    subtracted_image = cv2.absdiff(image1, image2)
    # Normalize to the range [0, 255]
    cv2.normalize(subtracted_image, subtracted_image, 0, 255, cv2.NORM_MINMAX)

    return subtracted_image

def reduce_gray_levels(image: np.ndarray, scale: float = 0.1) -> np.ndarray:
    """
    Reduces the number of gray levels in a grayscale image.

    Parameters:
        - image (np.ndarray): Input grayscale image.
        - scale (float): Desired scale for the number of gray levels (can be a non-integer). 
                            Default value = 0.1

    Returns:
        - np.ndarray: Image with reduced gray levels.
    """
    max_intensity = 255
    scale_factor = max_intensity / (scale - 0.5)

    # Reduce the gray levels
    reduced_image = np.floor(image / scale_factor).astype(np.uint8) * scale_factor

    return reduced_image

