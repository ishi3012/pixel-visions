# Refactor the ENUM with dynamic function call as did in spatial_filtering function. 

import cv2
import numpy as np
from typing import Tuple
import matplotlib.pyplot as plt
from enum import Enum, auto
import pixel_visions.utils as u
from typing import Dict, Tuple, List

class Normalization(Enum):
    MINMAX = auto()
    REVERSED = auto()
    LARGELOG = auto()
    GAMMA    = auto()
    EXTREME  = auto()
    RANDOM  = auto()

class LocalHistogramEqualizationMethod(Enum):
    ADAPTIVE_HISTOGRAM_EQUALIZATION = "Adaptive Histogram Equalization (AHE)"
    CONTRAST_LIMITED_ADAPTIVE_HISTOGRAM_EQUALIZATION = "Contrast Limited Adaptive Histogram Equalization (CLAHE)"
    MULTI_SCALE_HISTOGRAM_EQUALIZATION = "Multi-scale Histogram Equalization"
    SLIDING_WINDOW_HISTOGRAM_EQUALIZATION = "Sliding Window Histogram Equalization"
    BLOCK_BASED_HISTOGRAM_EQUALIZATION = "Block-based Histogram Equalization"

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

def display_image_histogram(image: np.ndarray, outputfile: str, title: str = "Image Histogram") -> None:
    """
    Calculates and displays the intensity histogram of the image alongside the image itself.

    Parameters:
        - image (np.ndarray): The input image to calculate the histogram.
        - outputfile (str): File location to save the plot.
        - title (str): Title of the plot. Default is "Image Histogram".   
    """
    if image is None:
        raise ValueError("The input image is empty")

    # Create a figure with 2 subplots: one for the image and one for the histogram
    plt.figure(figsize=(12, 5))

    # Subplot 1: Display the image
    plt.subplot(1, 2, 1)  # 1 row, 2 columns, 1st subplot
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for correct color representation
    plt.axis('off')  # Hide axis
    plt.title("Histogram Image")
    hist_data = []
    # Subplot 2: Display the histogram
    plt.subplot(1, 2, 2)  # 1 row, 2 columns, 2nd subplot
    if len(image.shape) == 2:  # Grayscale image
        hist, _ = np.histogram(image.ravel(), bins=256, range=[0, 256])
        hist_data.append(hist)
        plt.hist(image.ravel(), bins=256, range=[0, 256], color="black")
        plt.title(f"{title} (Grayscale)")
        plt.xlabel("Pixel Intensity")
        plt.ylabel("Frequency")
        
    
    elif len(image.shape) == 3:  # Colored image
        colors = ('blue', 'green', 'red')  # OpenCV uses BGR format
        
        for i, color in enumerate(colors):
            hist = cv2.calcHist([image], [i], None, [256], [0, 256])
            hist_data.append(hist)
            plt.plot(hist, color=color)
        plt.title(f"{title} (Color)")
        plt.xlabel("Pixel Intensity")
        plt.ylabel("Frequency")
        plt.xlim([0, 256])
        
    else:
        raise ValueError("Invalid image shape. Image must be either 2D (grayscale) or 3D (color).")

    if outputfile:
        plt.savefig(outputfile)    

    plt.show()
    plt.close()

    return hist_data

def display_equalize_histogram(image: np.ndarray, outputfile:str, title:str="Equalized image histogram") -> np.ndarray:
    """
    Enhances the contrast of an image using histogram equalization and displays the result.

    Parameters:
        - image (np.ndarray): The input grayscale image to equalize.
        - outputfile(str)   : File location to save the histogram plot.
        - title(str)        : Title of the plot. Default is "Equalized image histogram".
    Returns:
        np.ndarray: The histogram-equalized image.
    """
    equalized_images = []
    equalized_histograms = []
    

     # Check if the image is grayscale or colored
    if len(image.shape) == 2:  # Grayscale image
        # Perform histogram equalization
        equalized_image = cv2.equalizeHist(image)

        # Calculate histogram
        histogram, bins = np.histogram(image.flatten(), 256, [0, 256])
        equalized_histogram, _ = np.histogram(equalized_image.flatten(), 256, [0, 256])

        # Display the original and equalized images with their histograms
        plt.figure(figsize=(12, 6))

        # Original Image
        plt.subplot(2, 2, 1)
        plt.imshow(image, cmap='gray')
        plt.title('Original Grayscale Image')
        plt.axis('off')

        plt.subplot(2, 2, 2)
        plt.plot(histogram, color='black')
        plt.title('Histogram of Original Image')
        plt.xlim([0, 256])

        # Equalized Image
        plt.subplot(2, 2, 3)
        plt.imshow(equalized_image, cmap='gray')
        plt.title('Equalized Image')
        plt.axis('off')

        plt.subplot(2, 2, 4)
        plt.plot(equalized_histogram, color='black')
        plt.title('Histogram of Equalized Image')
        plt.xlim([0, 256])

    elif len(image.shape) == 3:  # Colored image
        # Split the channels
        channels = cv2.split(image)
        colors = ('b', 'g', 'r')  # OpenCV uses BGR format

        # Initialize a list for equalized images and their histograms
        equalized_images = []
        equalized_histograms = []

        plt.figure(figsize=(12, 6))

        # Display original image and histogram
        plt.subplot(2, 3, 1)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title('Original Color Image')
        plt.axis('off')

        for i, color in enumerate(colors):
            # Calculate histogram for each channel
            histogram = cv2.calcHist([channels[i]], [0], None, [256], [0, 256])
            plt.subplot(2, 3, i + 2)
            plt.plot(histogram, color=color)
            plt.title(f'Histogram of {color.upper()} Channel')
            plt.xlim([0, 256])

            # Perform histogram equalization for each channel
            equalized_image = cv2.equalizeHist(channels[i])
            equalized_images.append(equalized_image)
            equalized_histogram = cv2.calcHist([equalized_image], [0], None, [256], [0, 256])
            equalized_histograms.append(equalized_histogram)

        # Merge equalized channels back into an image
        equalized_image = cv2.merge(equalized_images)

        # Display equalized image
        plt.subplot(2, 3, 5)
        plt.imshow(cv2.cvtColor(equalized_image, cv2.COLOR_BGR2RGB))
        plt.title('Equalized Color Image')
        plt.axis('off')

        # Display histograms of equalized channels
        plt.subplot(2, 3, 6)
        for i, color in enumerate(colors):
            plt.plot(equalized_histograms[i], color=color)
        plt.title('Histogram of Equalized Image')
        plt.xlim([0, 256])

    else:
        raise ValueError("Invalid image shape. Image must be either 2D (grayscale) or 3D (color).")

    plt.tight_layout()

    if outputfile:
        plt.savefig(outputfile)   
    plt.show()

    return equalized_image

def adaptive_histogram_equalization(image):
    """
    Apply Adaptive Histogram Equalization (AHE) to the image.

    Args:
        image (numpy.ndarray): Input image (grayscale or color).

    Returns:
        numpy.ndarray: Enhanced image after applying AHE.
    """
    if len(image.shape) == 3:
        channels = cv2.split(image)
        ahe_channels = [cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(ch) for ch in channels]
        return cv2.merge(ahe_channels)
    else:
        ahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return ahe.apply(image)

def contrast_limited_adaptive_histogram_equalization(image: np.ndarray) -> np.ndarray:
    """
    Apply Contrast Limited Adaptive Histogram Equalization (CLAHE) to enhance image contrast.

    Args:
        image (numpy.ndarray): Input image (grayscale).

    Returns:
        numpy.ndarray: Enhanced image after applying CLAHE.
    """
    # Ensure the image is grayscale
    if len(image.shape) != 2:
        raise ValueError("CLAHE can only be applied to grayscale images.")

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(image)

def multi_scale_histogram_equalization(image: np.ndarray) -> np.ndarray:
    """
    Apply multi-scale histogram equalization to the image by averaging results 
    from multiple scales.

    Args:
        image (numpy.ndarray): Input image (grayscale or color).

    Returns:
        numpy.ndarray: Enhanced image after applying multi-scale histogram equalization.
    """
    scales = [1, 2, 4]
    enhanced_images = []

    if len(image.shape) == 2:  # Grayscale image
        for scale in scales:
            # Resize the image for the current scale
            resized_image = cv2.resize(image, (image.shape[1] // scale, image.shape[0] // scale))
            enhanced_image = contrast_limited_adaptive_histogram_equalization(resized_image)
            enhanced_images.append(cv2.resize(enhanced_image, (image.shape[1], image.shape[0])))

        return np.mean(enhanced_images, axis=0).astype(np.uint8)

    elif len(image.shape) == 3:  # Color image
        # Split the color channels
        channels = cv2.split(image)
        enhanced_channels = []

        for ch in channels:
            for scale in scales:
                # Resize the channel for the current scale
                resized_channel = cv2.resize(ch, (ch.shape[1] // scale, ch.shape[0] // scale))
                # Ensure the channel is grayscale before applying CLAHE
                enhanced_channel = contrast_limited_adaptive_histogram_equalization(resized_channel)
                enhanced_channels.append(cv2.resize(enhanced_channel, (ch.shape[1], ch.shape[0])))

        # Merge the enhanced channels back into a color image
        return cv2.merge(enhanced_channels).astype(np.uint8)

    else:
        raise ValueError("Input image must be either grayscale (2D) or color (3D).")

def sliding_window_histogram_equalization(image, window_size=(50, 50)):
    """
    Apply sliding window histogram equalization to the image.

    Args:
        image (numpy.ndarray): Input image (grayscale or color).
        window_size (tuple): Size of the sliding window (height, width).

    Returns:
        numpy.ndarray: Enhanced image after applying sliding window histogram equalization.
    """
    if len(image.shape) == 3:
        channels = cv2.split(image)
        enhanced_channels = [apply_sliding_window_histogram_equalization(ch, window_size) for ch in channels]
        return cv2.merge(enhanced_channels)
    else:
        return apply_sliding_window_histogram_equalization(image, window_size)

def apply_sliding_window_histogram_equalization(image: np.ndarray, window_size: tuple):
    """
    Apply the sliding window histogram equalization method to the image.

    Parameters:
        image (numpy.ndarray): Input image (grayscale or color).
        window_size (tuple): Size of the sliding window (height, width).

    Returns:
        numpy.ndarray: Enhanced image after applying sliding window histogram equalization.
    """
    # Check if the image is grayscale or color
    if len(image.shape) == 2:  # Grayscale image
        height, width = image.shape
        enhanced_image = np.zeros_like(image)

        for i in range(0, height, window_size[0]):
            for j in range(0, width, window_size[1]):
                window = image[i:i+window_size[0], j:j+window_size[1]]
                hist, _ = np.histogram(window.flatten(), 256, [0, 256])
                cdf = hist.cumsum()
                cdf_normalized = (cdf - cdf.min()) * 255 / (cdf.max() - cdf.min())
                enhanced_window = np.interp(window.flatten(), np.arange(256), cdf_normalized).reshape(window.shape)
                enhanced_image[i:i+window_size[0], j:j+window_size[1]] = enhanced_window

        return enhanced_image.astype(np.uint8)

    elif len(image.shape) == 3:  # Color image
        # Split the color channels
        channels = cv2.split(image)
        enhanced_channels = []

        for ch in channels:
            height, width = ch.shape
            enhanced_channel = np.zeros_like(ch)

            for i in range(0, height, window_size[0]):
                for j in range(0, width, window_size[1]):
                    window = ch[i:i+window_size[0], j:j+window_size[1]]
                    hist, _ = np.histogram(window.flatten(), 256, [0, 256])
                    cdf = hist.cumsum()
                    cdf_normalized = (cdf - cdf.min()) * 255 / (cdf.max() - cdf.min())
                    enhanced_window = np.interp(window.flatten(), np.arange(256), cdf_normalized).reshape(window.shape)
                    enhanced_channel[i:i+window_size[0], j:j+window_size[1]] = enhanced_window

            enhanced_channels.append(enhanced_channel)

        return cv2.merge(enhanced_channels).astype(np.uint8)

    else:
        raise ValueError("Input image must be either grayscale (2D) or color (3D).")

def block_based_histogram_equalization(image: np.ndarray, block_size: tuple=(50, 50)):
    """
    Apply block-based histogram equalization to the image.

    Parameters:
        image (numpy.ndarray): Input image (grayscale or color).
        block_size (tuple): Size of the block for histogram equalization (height, width).

    Returns:
        numpy.ndarray: Enhanced image after applying block-based histogram equalization.
    """
    if len(image.shape) == 3:
        channels = cv2.split(image)
        enhanced_channels = [apply_block_based_histogram_equalization(ch, block_size) for ch in channels]
        return cv2.merge(enhanced_channels)
    else:
        return apply_block_based_histogram_equalization(image, block_size)

def apply_block_based_histogram_equalization(image: np.ndarray, block_size: tuple):
    """
    Apply the block-based histogram equalization method to the image.

    Parameters:
        image (numpy.ndarray): Input image (grayscale or color).
        block_size (tuple): Size of the block for histogram equalization (height, width).

    Returns:
        numpy.ndarray: Enhanced image after applying block-based histogram equalization.
    """
    # Check if the image is grayscale or color
    if len(image.shape) == 2:  # Grayscale image
        height, width = image.shape
        enhanced_image = np.zeros_like(image)

        for i in range(0, height, block_size[0]):
            for j in range(0, width, block_size[1]):
                block = image[i:i+block_size[0], j:j+block_size[1]]
                hist, _ = np.histogram(block.flatten(), 256, [0, 256])
                cdf = hist.cumsum()
                cdf_normalized = (cdf - cdf.min()) * 255 / (cdf.max() - cdf.min())
                enhanced_block = np.interp(block.flatten(), np.arange(256), cdf_normalized).reshape(block.shape)
                enhanced_image[i:i+block_size[0], j:j+block_size[1]] = enhanced_block

        return enhanced_image.astype(np.uint8)

    elif len(image.shape) == 3:  # Color image
        # Split the color channels
        channels = cv2.split(image)
        enhanced_channels = []

        for ch in channels:
            height, width = ch.shape
            enhanced_channel = np.zeros_like(ch)

            for i in range(0, height, block_size[0]):
                for j in range(0, width, block_size[1]):
                    block = ch[i:i+block_size[0], j:j+block_size[1]]
                    hist, _ = np.histogram(block.flatten(), 256, [0, 256])
                    cdf = hist.cumsum()
                    cdf_normalized = (cdf - cdf.min()) * 255 / (cdf.max() - cdf.min())
                    enhanced_block = np.interp(block.flatten(), np.arange(256), cdf_normalized).reshape(block.shape)
                    enhanced_channel[i:i+block_size[0], j:j+block_size[1]] = enhanced_block

            enhanced_channels.append(enhanced_channel)

        return cv2.merge(enhanced_channels).astype(np.uint8)

    else:
        raise ValueError("Input image must be either grayscale (2D) or color (3D).")

def apply_local_histogram_equalization(image: np.ndarray, method: LocalHistogramEqualizationMethod):
    """
    Apply the specified local histogram equalization method to the image.

    Parameters:
        image (numpy.ndarray): Input image (grayscale or color).
        method (LocalHistogramEqualizationMethod): The local histogram equalization method to apply.

    Returns:
        numpy.ndarray: Enhanced image after applying the selected method.
    """
    if method == LocalHistogramEqualizationMethod.ADAPTIVE_HISTOGRAM_EQUALIZATION:
        return adaptive_histogram_equalization(image)
    elif method == LocalHistogramEqualizationMethod.CONTRAST_LIMITED_ADAPTIVE_HISTOGRAM_EQUALIZATION:
        return contrast_limited_adaptive_histogram_equalization(image)
    elif method == LocalHistogramEqualizationMethod.MULTI_SCALE_HISTOGRAM_EQUALIZATION:
        return multi_scale_histogram_equalization(image)
    elif method == LocalHistogramEqualizationMethod.SLIDING_WINDOW_HISTOGRAM_EQUALIZATION:
        return sliding_window_histogram_equalization(image)
    elif method == LocalHistogramEqualizationMethod.BLOCK_BASED_HISTOGRAM_EQUALIZATION:
        return block_based_histogram_equalization(image)
    else:
        raise ValueError("Unsupported method.")
    
def compute_histogram(image: np.ndarray) -> Dict[str, List[np.ndarray]]:
    """
    Compute and return the histogram for a given image. If the image is grayscale,
    the function returns a single histogram. For a colored image, it returns histograms
    for each channel (B, G, R).
    
    Parameters:
    -----------
    image : np.ndarray
        The input image for which the histogram needs to be computed.
        
        Flag indicating whether the image is grayscale. If True, a single histogram
        will be returned. If False, separate histograms for each color channel will be returned.
        Default is False.
    
    Returns:
    --------
    Dict[str, List[np.ndarray]]
        A dictionary containing the computed histograms. 
        - For a grayscale image, the key is 'grayscale' with the corresponding histogram.
        - For a colored image, keys are 'blue', 'green', and 'red' with their respective histograms.
    """
    
    histograms = {}

    # Grayscale image
    if len(image.shape)==2:
        hist = cv2.calcHist([image], [0], None, [256], [0, 256])
        histograms['grayscale'] = hist
    # Colored image
    elif len(image.shape)==3:
        # Compute histogram for each color channel (B, G, R)
        hist_b = cv2.calcHist([image], [0], None, [256], [0, 256])
        hist_g = cv2.calcHist([image], [1], None, [256], [0, 256])
        hist_r = cv2.calcHist([image], [2], None, [256], [0, 256])
        
        histograms['blue'] = hist_b
        histograms['green'] = hist_g
        histograms['red'] = hist_r
    else:
        raise ValueError(f"Unsupported Image format.")

    return histograms

def plot_and_save_histogram(image: np.ndarray, histogram: Dict[str, List[np.ndarray]], filename: str) -> None:
    """
    Plot the original image alongside its histogram and save the plot as an image file.
    
    Parameters:
    -----------
    image : np.ndarray
        The original image to be displayed.
    histogram : Dict[str, List[np.ndarray]]
        The histogram data computed by the `compute_histogram` function.
        Should contain either 'grayscale' or 'blue', 'green', and 'red' histograms.
    filename : str
        The file path where the output plot should be saved.
    
    Returns:
    --------
    None
        This function saves the resulting plot to a file and returns nothing.
    """
    
    # Create a figure for displaying the image and its histogram
    plt.figure(figsize=(10, 5))

    # Plot the original image
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for display
    plt.title('Original Image')
    plt.axis('off')

    # Plot the histogram(s)
    plt.subplot(1, 2, 2)
    if 'grayscale' in histogram:
        plt.hist(image.ravel(), bins=256, range=[0, 256], color="black")
        plt.title('Grayscale Histogram')
        plt.xlabel("Pixel Intensity")
        plt.ylabel("Frequency")
    else:
        plt.plot(histogram['blue'], color='b', label='Blue')
        plt.plot(histogram['green'], color='g', label='Green')
        plt.plot(histogram['red'], color='r', label='Red')
        plt.title('Color Histogram')
        plt.legend()

    # Adjust layout and save the plot
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

    # print(f"Histogram and image saved to {filename}")