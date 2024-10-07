"""
Assignment 3: Advanced Image Processing Techniques

This module implements various advanced image processing techniques, allowing users to work with multiple images at once by passing a dictionary of images.

- Histogram Equalization: Enhance the contrast of an image by equalizing its intensity histogram.
- Local Histogram Enhancement: Apply local contrast enhancement using different window sizes to optimize the result.
- Filtering: Perform filtering on images using filters such as Prewitt, Sobel, Point, and Blurring.
- Custom Filtering Function: Implement a pixel-by-pixel filtering method (for advanced students).
- Bit Plane Splicing: Analyze the frequency distribution of pixel intensities by splitting the image into its bit planes.
- Image Reconstruction from Bit Planes: Reconstruct the original image by progressively adding the most significant bit planes.

Each method processes a dictionary of images, where each key is the image name and the value is a NumPy array representing the image. 
This allows users to perform operations on multiple images if desired.

Usage:
Import this module and call the relevant method for the desired image operations on a dictionary of images.
"""


import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from enum import Enum, auto
from typing import Dict
import utils.image_processing as utils

def display_histogram(images: Dict[str, np.ndarray], outputfile: str)-> None:
    """
        Displays the histogram of image. 

        Parameters:
            - images (Dict[str, np.ndarray])    : A dictiionary of graysscale images. 
            - outputfiledirectory (str)         : A file directory to save the hisstogram image file. 
    """

    if images is None:
        raise ValueError(f"Images are not available.")  

    num_images = len(images)
    num_cols = 1
    num_rows = (num_images + num_cols -1 ) // num_cols
    plt.figure(figsize = (10, 5 * num_rows))

    for i, (key, image) in enumerate(images.items()):
        histogram, bin_edges = np.histogram(image, bins = 256,range = (0, 256))        
        # Create a subplot
        plt.subplot(num_rows, num_cols, i + 1)
        plt.title(f"Histogram of image - {key}")
        plt.xlabel("Pixel Intensity")
        plt.ylabel("Number of Pixels")
        plt.xlim([0,255])
        plt.bar(bin_edges[0:-1], histogram, width = 1, edgecolor = "black")    
    plt.tight_layout()
    plt.savefig(outputfile)
    plt.show()

def histogram_equalizer(images: Dict[str, np.ndarray], outputfile : str) -> Dict[str, np.ndarray]:
    """
    Calculates and Displays equilized images. 
    Parameters:
        - images (Dict[str, np.ndarray])    : A dictiionary of graysscale images. 
        - outputfiledirectory (str)         : A file directory to save the hisstogram image file. 
    Returns:
        - Dict[str, np.ndarray]             : Returns a dictionary of equalized images. 
    """
    if images is None:
        raise ValueError(f"Images are not available.")
    resultant_images = {}
    for key, image in images.items():
        resultant_images[key] = image
        resultant_images[key+"_equalized_Image"] = cv2.equalizeHist(image)

        # Display histogram for equalized Images 
        equalized_Images = dict(filter(lambda item: "_equalized_Image" in item[0], resultant_images.items()))
        display_histogram(equalized_Images, outputfile)
    # utils.plot_images(resultant_images, outputfile)

    return resultant_images

def create_clahe_image(grayscale_image: np.ndarray, clip_limit:float = 2.0, grid_size:(int, int) = (3,3)) -> np.ndarray:
    """
    Applies CLAHE (Contrast Limited Adaptive Histogram Equalization).
    Parameters:
        - grayscale_image (np.ndarray)      : The input grayscale image to apply the CLAHE equalization. 
        - clip_limit (float)                : Threshold for contrast limiting. Default is 2.0.
        - grid_size(int, int)               : Size of the grid for the histogram equalization. Default is (3,3)
    Returns:
        - np.ndarray                        : Returns the image after applying CLAHE in grayscale format. 
    """
    if len(grayscale_image.shape) == 3:
        grayscale_image = cv2.cvtColor(grayscale_image, cv2.COLOR_BGR2GRAY)

    clahe_grid = cv2.createCLAHE(clipLimit = clip_limit, tileGridSize = grid_size)
    
    return clahe_grid.apply(grayscale_image)

def apply_clahe(images: Dict[str, np.ndarray], outputfiledirectory: str) -> None:
    """
    Applies CLAHE enhancement technique on the dictionary of the input grayscale images and saves the results as a 
    video. The video shows the effect of applying varying window sizes for each image.
    Parameters:
        - images (Dict[str, np.ndarray])    : A dictiionary of graysscale images. 
        - outputfiledirectory (str)         : A file directory to save the hisstogram image file.     
    """
    for key, image in images.items():
        if len(image.shape) == 3:  # Skip if not a grayscale image
            continue        
        height, width = image.shape
        output = cv2.VideoWriter(outputfiledirectory + key + '.mp4', cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (width, height))

        # Apply CLAHE with varying window sizes.
        for window_size in range(1, min(height, width) + 1):
            clahe_image = create_clahe_image(image, grid_size=(window_size, window_size))

            # Normalize the CLAHE image to ensure pixel values are in the range [0, 255]
            clahe_image_normalized = utils.normalize_image(clahe_image)

            # Convert to BGR format for video writing
            clahe_image_bgr = cv2.cvtColor(clahe_image_normalized.astype(np.uint8), cv2.COLOR_GRAY2BGR)

            # Write the enhanced image with the current window size to the video writer.
            output.write(clahe_image_bgr)

        output.release()  # Release the video writer after processing each image

def create_filtered_image(grayscale_image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Apply a filter to an image using pixel-by-pixel convolution.

    This function applies a given kernel (filter) to each pixel in the image by calculating the
    sum of the products of the kernel values with the corresponding pixel values in the image.

    Assumptions:
        - For pixels on the border of the image, we extend the border pixels (padding) by replicating the nearest
        edge values (edge padding). This ensures the kernel can be applied even at the borders.
        - The input image is a grayscale image (2D array).
        - The kernel size is odd (e.g., 3x3, 5x5), so the kernel has a well-defined center. 
    Parameters:    
        - grayscale_image (np.ndarray)    : A 2D numpy array representing the grayscale image to be filtered.
        - kernel (np.ndarray)   : A 2D numpy array representing the kernel/filter to be applied. 
                                It must be a square matrix with odd dimensions.
    Returns:    
        - np.ndarray            : A 2D numpy array representing the filtered image.
    """
    # Get the dimensions of the input image and kernel
    img_height, img_width = grayscale_image.shape
    kernel_height, kernel_width = kernel.shape

    # Verify the kernel shape
    assert kernel_height == kernel_width, "Kernel must be square."
    assert kernel_height % 2 == 1, "Kernel size must be odd." 

    # Create a padded image
    padded_image = np.pad(grayscale_image, kernel_height // 2, mode = "edge")

    # Create an array to store values of the filtered image
    filtered_image = np.zeros_like(grayscale_image)

    for i in range(img_height):
        for j in range(img_width):
            region = padded_image[i:i + kernel_height, j:j + kernel_width]
            filtered_value = np.sum(region * kernel)
            filtered_image[i, j] = filtered_value
        
    return filtered_image

def apply_sobel_filter(image: np.ndarray) -> np.ndarray:
    """
    Applies filter specified in the kernel input matrix.

    Parameters:
        - images (Dict[str, np.ndarray])    : A dictiionary of graysscale images. 
        - kernel (np.ndarray)               : A 2D numpy array representing the kernel/filter to be applied. 
                                              It must be a square matrix with odd dimensions.
        - outputfile (str)                  : A file directory to save the filtered image file. 
    Returns:
        - Dict[str, np.ndarray]             : A dictionary of filtered images.
    """
   
    filtered_image = create_filtered_image(image, kernel)
    
    # plot the filtered images.
    utils.plot_images(filtered_images, outputfile)

    # return dictionary  of filtered images. 
    return filtered_images

def apply_prewitt_filter(image: np.ndarray) -> np.ndarray:
    """
    Applies the Prewitt filter to an image to detect edges.    
    Parameters:
        - image (np.ndarray): The input grayscale image to apply the filter on.
    
    Returns:
        - np.ndarray: The image with Prewitt filter applied.
    """
    # Define Prewitt kernels
    kernel_x = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]], dtype=np.float32)
    kernel_y = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=np.float32)
    
    # Apply filters
    prewitt_x = cv2.filter2D(image, cv2.CV_32F, kernel_x)  # Convert to float32
    prewitt_y = cv2.filter2D(image, cv2.CV_32F, kernel_y)  # Convert to float32
    
    # Combine results using cv2.magnitude
    prewitt = cv2.magnitude(prewitt_x, prewitt_y)
    
    # Normalize the result to 0-255 and convert back to uint8
    prewitt = cv2.normalize(prewitt, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    return prewitt

def apply_point_filter(image: np.ndarray, point_value: float = 1.5) -> np.ndarray:
    """
    Applies a simple point filter by multiplying each pixel by a point value.
    
    Parameters:
        - image (np.ndarray)  : The input grayscale image to apply the filter on.
        - point_value (float) : The value to multiply each pixel by. Default is 1.5.
    
    Returns:
        np.ndarray          : The image after applying the point filter.
    """
    return np.clip(image * point_value, 0, 255).astype(np.uint8)

def apply_blur_filter(image: np.ndarray, ksize: tuple[int, int] = (5, 5)) -> np.ndarray:
    """
    Applies a blurring filter to an image.
    
    Parameters:
        - image (np.ndarray): The input grayscale image to apply the filter on.
        - ksize (Tuple[int, int]): The size of the kernel to use for blurring. Default is (5, 5).
    
    Returns:
        np.ndarray: The image after applying the blurring filter.
    """
    return cv2.GaussianBlur(image, ksize, 0)

def apply_filters(images: Dict[str, np.ndarray], outputfile: str) -> Dict[str, np.ndarray]:
    """
    Applies Prewitt filter, a Sobel filter, a point filter, and the blurring filter using OpenCV to the dictionary of images. 

    Parameters:
        - images (Dict[str, np.ndarray])    : A dictiionary of graysscale images.         
        - kernel_size (int)                 : represents the size of the kernel.
        - outputfile (str)                  : A file directory to save the filtered image file. 
    Returns:
        - Dict[str, np.ndarray]             : A dictionary of filtered images.
    """
    if images is None:
        raise ValueError(f"Grayscale images are not available.")
    
    resultant_images = {}
    for key, image in images.items():
        resultant_images[key] = image
        # Apply filters
        resultant_images[key+"_prewitt"] = apply_prewitt_filter(image)
        resultant_images[key+"_sobel"]   = apply_sobel_filter(image)
        resultant_images[key+"_point"] = apply_point_filter(image)
        resultant_images[key+"_blur"] = apply_blur_filter(image)

    return resultant_images

def apply_filter_pixel_by_pixel(images: Dict[str, np.ndarray], kernel: int, outputfile: str) -> Dict[str, np.ndarray]:
    """
    Applies Prewitt filter, a Sobel filter, a point filter, and the blurring filter using OpenCV to the dictionary of images. 

    Parameters:
        - images (Dict[str, np.ndarray])    : A dictiionary of graysscale images.         
        - kernel_size (int)                 : represents the size of the kernel.
        - outputfile (str)                  : A file directory to save the filtered image file. 
    Returns:
        - Dict[str, np.ndarray]             : A dictionary of filtered images.
    """
    if images is None:
        raise ValueError(f"Grayscale images are not available.")
    filtered_images = {}
    for key, image in images.items():
        filtered_images[key] = create_filtered_image(image, kernel)
    
    utils.plot_images(filtered_images, outputfile)

    return filtered_images

def bit_plane_slicing(image):
    """
    Perform bit-plane slicing on the given grayscale image.
    Args:
        image (ndarray): Input image in grayscale format (0-255).
    Returns:
        tuple: A tuple containing the original image and an array of the 8 bit planes.
    """
    # Ensure the image is in grayscale
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Create an array to hold the bit planes
    bit_planes = np.zeros((8, *image.shape), dtype=np.uint8)

    # Extract each bit plane
    for i in range(8):
        bit_planes[i] = (image >> i) & 1  # Get the i-th bit plane
        # Debug: Print the shape of each bit plane
        print(f'Bit plane {i} shape: {bit_planes[i].shape}')

    return image, bit_planes

def assemble_image(bit_planes):
    """
    Assemble the original image from the extracted bit planes.

    Args:
        bit_planes (ndarray): Array containing the 8 bit planes.
    Returns:
        list: A list of assembled images at each stage of combination.
    """
    assembled_images = []
    assembled_image = np.zeros(bit_planes[0].shape, dtype=np.uint8)  

    # Add bit planes progressively
    for i in range(8):
        # Shift the current bit plane and combine it
        assembled_image |= (bit_planes[i] << i)  
        assembled_images.append(assembled_image.copy())  

    return assembled_images

def display_original_image(original_image: np.ndarray, image_title: str, outputfile: str):
    """
    Display the original image and save it as a JPEG file.

    Args:
        original_image (ndarray): The original grayscale image.
        image_title (str): Title of the image to display.
        outputfile (str): Path to save the original image plot as a JPEG file.
    """
    plt.figure(figsize=(12, 12))
    plt.imshow(original_image, cmap='gray')
    plt.title(f'Original Image: {image_title}')
    plt.axis('on')
    plt.savefig(outputfile)  
    plt.show()

def display_bit_planes(bit_planes: np.ndarray, outputfile: str):
    """
    Display all bit planes in a single plot and save it as a JPEG file.

    Args:
        bit_planes (ndarray): Array containing the 8 bit planes.
        outputfile (str): Path to save the bit planes plot as a JPEG file.
    """
    num_images = len(bit_planes)  
    plt.figure(figsize=(30, 15))  
    for i in range(num_images):
        plt.subplot(2, 4, i + 1)  
        plt.imshow(bit_planes[i] * 255, cmap='gray', aspect='auto')  
        plt.title(f'Bit Plane {i}')
        plt.axis('off')

    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, wspace=0.1, hspace=0.1)  
    plt.savefig(outputfile)  
    plt.show()  

def display_assembled_images(assembled_images: list, outputfile: str):
    """
    Display all assembled images in a single plot and save it as a JPEG file.

    Args:
        assembled_images (list): List of images assembled from the bit planes.
        outputfile (str): Path to save the assembled images plot as a JPEG file.
    """
    num_images = len(assembled_images)
    cols = 2  
    rows = (num_images + cols - 1) // cols  

    plt.figure(figsize=(20, 10 * rows))  
    for i in range(num_images):
        plt.subplot(rows, cols, i + 1)        
        plt.imshow(assembled_images[i], cmap='gray', aspect='auto')  
        plt.title(f'Assembled Image (1-{i + 1})')
        plt.axis('off')

    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, wspace=0.05, hspace=0.05)  
    plt.savefig(outputfile)  
    plt.show()  

def display_results(original_image: np.ndarray, bit_planes: np.ndarray, assembled_images: list, image_title: str, outputfile:str):
    """
    Display the original image, bit planes, and assembled images, saving each as a JPEG file.

    Args:
        original_image (ndarray): The original grayscale image.
        bit_planes (ndarray): Array containing the 8 bit planes.
        assembled_images (list): List of images assembled from the bit planes.
        image_title (str): Title of the image to display.
    """
    display_original_image(original_image, image_title, outputfile+"original_image.jpeg")  
    display_bit_planes(bit_planes, outputfile+"bit_planes.jpeg")                            
    display_assembled_images(assembled_images, outputfile+"assembled_images.jpeg")

def process_images(image_dict, outputfile:str):
    """
    Process each image in the provided dictionary to perform bit-plane slicing.

    Args:
        image_dict (dict): Dictionary where keys are image titles and values are image file paths.
        save_dir (str): Directory to save the processed images.
    """
    for title, image in image_dict.items():     
        
        # Perform bit-plane slicing
        original_image, bit_planes = bit_plane_slicing(image)
        
        # Assemble the image
        assembled_images = assemble_image(bit_planes)        
        
        # Display and save the results
        display_results(original_image, bit_planes, assembled_images, title, outputfile)



def main():
    input_image_paths = ["assignments/assignment3/Butterflies.jpg"]
    images =utils.read_images(input_image_paths)
    gray_scale_images = dict(filter(lambda item: "GS_" in item[0],images.items()))

    output_images_directory = "assignments/assignment3/Output/OutputImages/"
    output_plot_directory = "assignments/assignment3/Output/"

    # Plot images
    utils.plot_images(images, title = "Display input images", outputfile = output_plot_directory+"Input_Images.jpeg")

   

if __name__ == "__main__":
    main()