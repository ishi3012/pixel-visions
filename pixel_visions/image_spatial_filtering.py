"""
Filtering in image processing refers to the process of modifying or enhancing an image 
by manipulating its pixel values based on specific algorithms or mathematical operations. 
It is widely used to improve image quality, extract features, or prepare images for further analysis.
Types of Filtering
Spatial Filtering:
Involves manipulating the pixel values based on their neighboring pixels within a defined area or window.
    Low-Pass Filters (Smoothing): Reduce noise and blur the image (e.g., Gaussian filter, average filter).
    High-Pass Filters (Sharpening): Enhance edges and fine details (e.g., Laplacian filter, Sobel filter).
Frequency Domain Filtering:
    Involves transforming the image into the frequency domain using techniques like the Fast Fourier Transform (FFT).
    Filters are applied in the frequency domain, and the result is transformed back to the spatial domain.
    Common frequency filters include:
        High-Pass Filters: Emphasize high-frequency components (edges).
        Low-Pass Filters: Emphasize low-frequency components (smooth areas).
"""


import cv2
import numpy as np
from typing import Tuple
import matplotlib.pyplot as plt
from enum import Enum, auto
import pixel_visions.utils as u
from typing import Dict, Tuple, List, Any, Callable

def apply_filter(image: np.ndarray, kernel: List[List[int]]) -> np.ndarray:
    """
    Apply a filter to the input image pixel by pixel and explicitly calculate the sum of products.
    
    Assumptions:
   
        - For pixels on the borders, zero-padding is assumed (i.e., pixels outside the border are treated as zero).
        - The kernel size is odd (e.g., 3x3, 5x5), so the kernel has a well-defined center. 

    Parameters:
    -----------
    image : np.ndarray
        2D numpy array representing the grayscale image.
    filter_kernel : List[List[int]]
        2D list representing the 3x3 filter to apply.

    Returns:
    --------
    np.ndarray
        2D numpy array of the same shape as the input image after applying the filter.
    """
    if image is None:
        raise ValueError("The input image is empty")
    
    # Verify the kernel shape
    kernel_height, kernel_width = kernel.shape
    assert kernel_height == kernel_width, "Kernel must be square."
    assert kernel_height % 2 == 1, "Kernel size must be odd." 

    if len(image.shape) == 2:
       return _apply_filter_grayscale(image, kernel)
    
    elif len(image.shape) == 3:
        return _apply_filter_color(image, kernel)
    
    else:
        raise ValueError("Unsupported image type.")

def _apply_filter_grayscale(grayscale_image: np.ndarray, kernel: List[List[int]]) -> np.ndarray:
    """
    Apply a filter to the input image pixel by pixel and explicitly calculate the sum of products.
    
    Assumptions:
   
        - For pixels on the borders, zero-padding is assumed (i.e., pixels outside the border are treated as zero).
        - The kernel size is odd (e.g., 3x3, 5x5), so the kernel has a well-defined center. 

    Parameters:
    -----------
    grayscale_image : np.ndarray
        2D numpy array representing the grayscale image.
    filter_kernel : List[List[int]]
        2D list representing the 3x3 filter to apply.

    Returns:
    --------
    np.ndarray
        2D numpy array of the same shape as the input image after applying the filter.
    """
    if grayscale_image is None:
        raise ValueError("The input image is empty")
    
    if len(grayscale_image.shape) != 2:
        raise ValueError("The image is not grayscale.")
    
    image_height, image_width = grayscale_image.shape
    kernel_height, kernel_width = kernel.shape
    pad = kernel_height // 2

    padded_image = np.pad(grayscale_image, pad, mode = "constant", constant_values = 0)
    filtered_image = np.zeros_like(grayscale_image)

    # Iterate over each pixel in the original image
    for i in range(image_height):
        for j in range(image_width):
            # Get the padded image region
            region = padded_image[i:i+kernel_height, j:j+kernel_width]
            filtered_value = np.sum(region * kernel)
            filtered_image[i,j] = filtered_value

    return filtered_image

def _apply_filter_color(image: np.ndarray, kernel: List[List[int]]) -> np.ndarray:
    """
    Apply a filter to a color image (RGB) pixel by pixel and explicitly calculate the sum of products.
    
    Assumptions:
    - The input image is a 3D color image (numpy array) with shape (height, width, 3).
    - The filter is a 3x3 kernel.
    - For pixels on the borders, zero-padding is assumed (i.e., pixels outside the border are treated as zero).

    Parameters:
    -----------
        - image : np.ndarray : 3D numpy array representing the color image (RGB).
        - filter_kernel : List[List[int]] : 2D list representing the 3x3 filter to apply.      

    Returns:
    --------
        - np.ndarray : Filtered Colored image
    """
    image_height, image_width, num_channels = image.shape
    kernel_height, kernel_width = kernel.shape
    pad = kernel_height // 2

    filtered_image = np.zeros((image_height, image_width, num_channels), dtype = np.float32)
    
    padded_image = np.pad(image, pad_width=((pad, pad), (pad, pad), (0, 0)), mode='constant', constant_values=0)

    # Iterate over each channel (e.g., R, G, B)
    for channel in range(num_channels):
        # Iterate over each pixel in the original image
        for i in range(image_height):
            for j in range(image_width):
                # Get the padded image region
                region = padded_image[i:i+kernel_height, j:j+kernel_width, channel]
                filtered_value = np.sum(region * kernel)
                filtered_image[i,j, channel] = filtered_value

    # Clip values to the valid range and convert to uint8
    filtered_image = np.clip(filtered_image, 0, 255)  # Ensure values are within [0, 255]
    filtered_image = filtered_image.astype(np.uint8)  # Convert to uint8

    return filtered_image

class SpatialFilter(Enum):
    GAUSSIAN = (cv2.GaussianBlur, "'Gaussian") # Smooth the image using a Gaussian kernel.
    AVERAGE = (cv2.blur, "Average") # Smooth the image.
    MEDIAN = (cv2.medianBlur, "Median") # Reduces salt-and-pepper noise
    BILATERAL = (cv2.bilateralFilter, "Bilateral") # Smoothes the image while preserving edges
    BOX = (cv2.boxFilter, "Box") # Apply a box filter (similar to the average filter but with more flexibility)
    def __init__(self, function: Any, name: str) -> None:
        """
        Initializes the spatial filter enum member.
        Args:
            function: The OpenCV function associated with the filter.
            name: The name of the filter as a string.
        """
        self._function = function
        self._name = name
    @property
    def function(self) -> Any:
        """Get the OpenCV function associated with the filter."""
        return self._function
    @property
    def name(self) -> str:
        """Get the name of the filter."""
        return self._name  
    def apply(self, image: np.ndarray, *args: Tuple[Any], **kwargs: Any) -> np.ndarray:
        """
        Apply the filter to the given image with additional arguments.
        Args:
            image (np.ndarray): The input image to be filtered.
            *args: Additional positional arguments to pass to the filter function.
            **kwargs: Additional keyword arguments to pass to the filter function.
        Returns:
            np.ndarray: The filtered image.
        """
        return self.function(image, *args, **kwargs)
    
class EdgeDetection(Enum):
    SOBEL   = ("Sobel", lambda image, **kwargs: sobel_filter(image, **kwargs))
    CANNY   = ("Canny", lambda image, **kwargs: canny_filter(image, **kwargs))
    PREWITT = ("Prewitt", lambda image, **kwargs: prewitt_filter(image, **kwargs))
    ROBERTS = ("Roberts", lambda image, **kwargs: roberts_filter(image, **kwargs))
    GAUSSIAN_DERIVATIVE = ("Gaussian_derivative", lambda image, **kwargs:apply_custom_sobel_filter(image, **kwargs))


    def __init__(self, name: str, function: Callable):
        """
        Initializes the edge detection enum member.

        Args:
            name: The name of the Edge detection filter as a string.
            function: The function associated with the edge detection filter.
        """
        self._name = name
        self._function = function

    @property
    def name(self) -> str:
        """ Get the name of the Edge Detection filter."""
        return self._name

    def apply(self, image: np.ndarray, **kwargs: Any) -> np.ndarray:
        """
        Apply the Edge detection filter to the given image with additional arguments.

        Args:
            image (np.ndarray): The input image to be filtered.
            **kwargs: Additional keyword arguments to pass to the filter function.

        Returns:
            np.ndarray: The filtered image.
        """
        parameters_string = f"Applying {self.name} with the following arguments:\n"
        parameters_string+=f"    Image: {image.shape}"
        # print(f"Applying {self.name} with the following arguments:")
        # print(f"    Image: {image.shape}")
        if kwargs:
            # print(f"    Filter Parameters: {kwargs}")
            parameters_string+=f"    Filter Parameters: {kwargs}"
        
        return self._function(image, **kwargs), parameters_string

# def sobel_filter(image: np.ndarray, ddepth: int = cv2.CV_64F, dx: int = 1, dy: int = 0, ksize: int = 3, scale: int = 1, delta: int = 0, borderType: int = cv2.BORDER_DEFAULT) -> np.ndarray:
def sobel_filter(image: np.ndarray, **kwargs: Any) -> np.ndarray:
    """
    Apply the Sobel filter to the given image. 
    Args:
        image (np.ndarray): The input image to be filtered.
        ddepth (int): Desired depth of the output image (default is cv2.CV_64F).
        dx (int): Order of the derivative in x (default is 1).
        dy (int): Order of the derivative in y (default is 0).
        ksize (int): Size of the Sobel kernel (default is 3).
        scale (int): Optional scaling factor for the computed derivative (default is 1).
        delta (int): Optional offset added to the results (default is 0).
        borderType (int): Pixel extrapolation method (default is cv2.BORDER_DEFAULT).
    Returns:
        np.ndarray: The filtered image.
    """
    # Check if the image is grayscale
    if len(image.shape) != 2:
        raise ValueError("Input image must be a grayscale image.")    
    # Default values
    ddepth = kwargs.get('ddepth', cv2.CV_64F)
    dx = kwargs.get('dx', 1)
    dy = kwargs.get('dy', 0)
    ksize = kwargs.get('ksize', 3)
    scale = kwargs.get('scale', 1)
    delta = kwargs.get('delta', 0)
    borderType = kwargs.get('borderType', cv2.BORDER_DEFAULT)
    # Apply Sobel filter in x and y directions
    sobel_x = cv2.Sobel(image, ddepth=ddepth, dx=dx, dy=0, ksize=ksize, scale=scale, delta=delta, borderType=borderType)
    sobel_y = cv2.Sobel(image, ddepth=ddepth, dx=0, dy=dy, ksize=ksize, scale=scale, delta=delta, borderType=borderType)
    # Combine gradients
    sobel_edges = cv2.magnitude(sobel_x, sobel_y)
    # Normalize to the range [0, 255]
    sobel_edges = cv2.normalize(sobel_edges, None, 0, 255, cv2.NORM_MINMAX)
    # Convert to 8-bit image
    return np.uint8(sobel_edges)

def canny_filter(image: np.ndarray, **kwargs: Any) -> np.ndarray:
    """
    Apply the Canny edge detection filter to the given image.

    Args:
        image (np.ndarray): The input image to be filtered.
        threshold1 (int): The lower threshold for hysteresis.
        threshold2 (int): The upper threshold for hysteresis.
    Returns:
        np.ndarray: The filtered image.
    """
    # Default values
    threshold1      = kwargs.get('threshold1', 100)
    threshold2      = kwargs.get('threshold2', 200)
    apertureSize    = kwargs.get('apertureSize', 3)
    L2gradient      = kwargs.get('L2gradient', False)

    # Check if the image is grayscale
    if len(image.shape) != 2:
        raise ValueError("Input image must be a grayscale image.")
    return cv2.Canny(image, threshold1, threshold2, apertureSize=apertureSize, L2gradient=L2gradient)

def prewitt_filter(image: np.ndarray, **kwargs) -> np.ndarray:
    """
    Apply the Prewitt filter to the given image.
    Args:
        image (np.ndarray): The input image to be filtered.
        **kwargs: Additional parameters for the filter (not used in this simple implementation).

    Returns:
        np.ndarray: The filtered image.
    """
    # Default values
    threshold  = kwargs.get('threshold', 100)
    scale      = kwargs.get('scale', 1)
    # Define Prewitt kernels
    kernel_x = np.array([[1, 0, -1],
                          [1, 0, -1],
                          [1, 0, -1]], dtype=np.float32)    
    kernel_y = np.array([[1, 1, 1],
                          [0, 0, 0],
                          [-1, -1, -1]], dtype=np.float32)
    # Apply the kernels to the image
    grad_x = cv2.filter2D(image, cv2.CV_32F, kernel_x)
    grad_y = cv2.filter2D(image, cv2.CV_32F, kernel_y)
    # Compute the magnitude of the gradient
    filtered_image = cv2.magnitude(grad_x, grad_y)
    # Scale the output
    filtered_image *= scale
    # Apply thresholding
    _, filtered_image = cv2.threshold(filtered_image, threshold, 255, cv2.THRESH_BINARY)

    return filtered_image

def roberts_filter(image: np.ndarray, scale=1, threshold=0, **kwargs) -> np.ndarray:
    """
    Apply the Roberts filter to the given image.
    Args:
        image (np.ndarray): The input image to be filtered.
        scale (float): Scaling factor for the output.
        threshold (int): Threshold for edge detection.
        **kwargs: Additional parameters for the filter.
    Returns:
        np.ndarray: The filtered image.
    """
    # Default values
    threshold  = kwargs.get('threshold', 100)
    scale      = kwargs.get('scale', 1)
    # Roberts kernels
    kernel_x = np.array([[1, 0],
                         [0, -1]], dtype=np.float32)
    kernel_y = np.array([[0, 1],
                         [-1, 0]], dtype=np.float32)
    # Apply the kernels to the image
    grad_x = cv2.filter2D(image, cv2.CV_32F, kernel_x)
    grad_y = cv2.filter2D(image, cv2.CV_32F, kernel_y)
    # Compute the magnitude of the gradient
    filtered_image = cv2.magnitude(grad_x, grad_y)
    # Scale the output
    filtered_image *= scale
    # Apply thresholding if needed
    if threshold > 0:
        _, filtered_image = cv2.threshold(filtered_image, threshold, 255, cv2.THRESH_BINARY)

    return filtered_image

def apply_custom_sobel_filter(image: np.ndarray, ksize = 7, sigma = 1, **kwargs) -> np.ndarray:
    """
    Apply custom Sobel filters to detect edges in both x and y directions.

    Parameters:
        image (np.ndarray): Input grayscale image.
        sobel_x (np.ndarray): Sobel-like filter in x direction.
        sobel_y (np.ndarray): Sobel-like filter in y direction.

    Returns:
        edges (np.ndarray): Combined edges detected from both directions.
    """

    k = ksize // 2
    x, y = np.meshgrid(np.arange(-k, k+1), np.arange(-k, k+1))

    sobel_x = -x / (2 * np.pi * sigma ** 4) * np.exp(-(x ** 2 + y ** 2)/(2 * sigma ** 2))
    sobel_y = -y / (2 * np.pi * sigma ** 4) * np.exp(-(x ** 2 + y ** 2)/(2 * sigma ** 2))
    grad_x = cv2.filter2D(image, -1, sobel_x).astype(np.float32)
    grad_y = cv2.filter2D(image, -1, sobel_y).astype(np.float32)
    sobel_edges = cv2.magnitude(grad_x, grad_y)
    sobel_edges = cv2.normalize(sobel_edges, None, 0, 255, cv2.NORM_MINMAX)
    # Convert to 8-bit image
    return np.uint8(sobel_edges)




