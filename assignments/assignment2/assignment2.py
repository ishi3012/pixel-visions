"""
Assignment 2: Image Processing Operations

This module implements various image processing techniques:

- Shrinking: Reduce image dimensions by a specified factor.
- Zooming: Resize a shrunk image back to its original dimensions using pixel replication or other techniques.
- Contrast Stretching: Apply different methods to enhance image contrast.
- Image Negation: Generate the negative of an image.
- Piecewise Stretching: Apply piecewise linear transformations to enhance contrast in specific regions.
- Image Transformations: Perform other transformations such as rotation, scaling, and translation.

Each method processes a dictionary of images, where each key is the image name and the value is a NumPy array representing the image. 
This allows users to perform operations on multiple images if desired.

Usage:
Import this module and call the relevant method for the desired image operation.
"""

import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from typing import Dict, Type
from enum import Enum, auto


# Set the backend to Agg for non-interactive plotting
matplotlib.use('Agg')

class ImageType(Enum):
    ORIGINAL    = auto()
    REDUCED     = auto()
    ZOOMED      = auto()
    SUBTRACTED  = auto()
    NEGATIVE  = auto()

class ContrastType(Enum):
    CONTRAST_STRETCH_BASIC  = auto()
    CONTRAST_STRETCH_REVERSED = auto()
    CONTRAST_STRETCH_LARGELOG = auto()
    CONTRAST_STRETCH_GAMMA    = auto()
    CONTRAST_STRETCH_EXTREME  = auto()
    CONTRAST_STRETCH_RANDOM  = auto()


def read_images(image_paths: list) -> Dict[str, np.ndarray]:
    """
        Reads image files from a list of file path locations using OpenCV.

        Parameters:
            - image_paths (List[str]): A list of file paths pointing to the image files to be processed.
        Returns:
            - Dict[str, np.ndarray]: A dictionary where the key is the image name (without extension) and the value is the image data as a NumPy array.
    
    """

    if image_paths is None:
        raise ValueError("The list of image filepaths is empty. ")    

    images = {} 
    image_count = 0   
    for path in image_paths:
        if path is None:
            continue
        image_name = path.split("/")[-1].split(".")[0]
        images[image_name] = cv2.imread(path)
        image_count += 1 
        # Convert and save the colored image as grayscale as well in the dictionary.
        if len(images[image_name].shape) == 3:
            images["GS_"+image_name] = cv2.cvtColor(images[image_name], cv2.COLOR_BGR2GRAY)    
            
    print(f" Total images loaded = {image_count} out of {len(image_paths)}")
    return images


def save_image(images: Dict[str, np.ndarray], fileDirectory = "assignments/assignment2/Output/OutputImages/", imageType : str = "Original") -> None:
    """
    Saves the image file.

    Parameters:
    - images: Dict[str, np.ndarray]   : Dictionary of images to save.
    - fileDirectory(str)              : The path of the directory to save the images.
    - imageType (str)                 : Type of image to save.
    """
    if images is None:
        print(f"ERROR: Images not available.")
        return
    
    for i, (key, image) in enumerate(images.items()):
        filename = fileDirectory + str(imageType) + "_" + key + ".jpeg"
        success = cv2.imwrite(filename, image)

        if success:
            print(f"Image saved successfully as {filename}.\n")
        else:
            print(f"Failed to save the image.\n")

def plot_images(images: Dict[str, np.ndarray] , title: str = "Display images", outputfile:str = "assignments/assignment2/Output/Output.jpeg") -> None:
    """
    Display the images with the given title.

    Parameters: 
        - images (Dict[str, np.ndarray]) : Dictionary of images to plot
        - title (str)                    : Text to display as title of the plot
        - outputfile(str)                : The output file path. Default is "assignments/assignment2/Output/Output.jpeg"

    """
    if images is None:
        print(f"ERROR: Images not available.")
        return

    num_cols = 2
    num_rows = (len(images) + num_cols - 1) // num_cols

    plt.figure(figsize=(10, 5 * num_rows))

    for i, (key, image) in enumerate(images.items()):
        # print(f"Displaying Image: {key} ")

        plt.subplot(num_rows, num_cols, i + 1) 

        # Display Color image
        if len(image.shape) == 3:
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        # Display grayscale image
        else:
            plt.imshow(image, cmap='gray')
        plt.axis('on')
        #plt.title(key, fontsize=10, pad=2) 
        plt.title(key, fontsize=10) 

        # Set a fixed size for the display
        # plt.gca().set_aspect('equal', adjustable='box')

    plt.suptitle(title)
    # Use tight_layout to minimize extra space
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for suptitle
    plt.savefig(outputfile)
    plt.close()

def scale_image(images: Dict[str, np.ndarray] , scale_factor: int = 1) -> None:
    """
    Scales the image by the given scaling factor. 

    
    Parameters:
        - images: Dict[str, np.ndarray] : Dictionary of images to shrink
        - scale_factor(int)             : The scaling factor by which the input image should be scaled.          
    """   
    if images is None:
        print(f"ERROR: Images not available.")
        return {}
    if not isinstance(scale_factor, int):
        print(f"ERROR: Scale factro must be an integer.")
        return {}


    if scale_factor < 0:
        return shrink_image(images, abs(scale_factor))
    else:
        return zoom_image(images, scale_factor)

def shrink_image(images: Dict[str, np.ndarray] , scale_factor: int = 1) -> Dict[str, np.ndarray]:
    """
    Shrinks the image by the given scaling factor. 
    Parameters:
        - images: Dict[str, np.ndarray] : Dictionary of images to shrink
        - scale_factor(int)             : The scaling factor by which the input image should be shrunk.
    Returns:
        - np.ndarray    : Returns the dictionary of redcued images.
    """
    if images is None:
        print(f"ERROR: Images not available.")
        return {}

    reduced_images = {}

    for i, (key, image) in enumerate(images.items()):        
        height, width = image.shape[:2]
        # Use slicing to reduce the image
        if len(image.shape) == 3:  # Color image
            reduced_images[key] = image[::scale_factor, ::scale_factor]
        else:  # Grayscale image
            reduced_images[key] = image[::scale_factor, ::scale_factor]
    return reduced_images  

def zoom_image(images: Dict[str, np.ndarray], scale_factor: int = 1) -> Dict[str, np.ndarray]:
    """
    Zooms the images by the given scaling factor. 
    Parameters:
        - images: Dict[str, np.ndarray]: Dictionary of images to zoom
        - scale_factor (int): The scaling factor by which the input image should be zoomed.
        
    Returns:
        - Dict[str, np.ndarray]: Returns the dictionary of zoomed images.
    """

    if images is None:
        print(f"ERROR: Images not available.")
        return {}

    zoomed_images = {}
    for key, image in images.items():
        height, width = image.shape[:2]  # Get height and width

        # Calculate new dimensions
        increased_height, increased_width = height * scale_factor, width * scale_factor
        
        # Initialize a zoomed image
        if len(image.shape) == 3:
            zoomed_image = np.zeros((increased_height, increased_width, image.shape[2]), dtype=np.uint8)
        else:
            zoomed_image = np.zeros((increased_height, increased_width), dtype=np.uint8)
        # Populate the zoomed image using pixel replication
        for i in range(increased_height):
            for j in range(increased_width):
                original_i = i // scale_factor
                original_j = j // scale_factor                
                # Ensure original indices are within bounds
                original_i = min(original_i, height - 1)
                original_j = min(original_j, width - 1)                
                zoomed_image[i, j] = image[original_i, original_j]        
        # Add zoomed image to the dictionary
        zoomed_images[key] = zoomed_image 
    return zoomed_images

def calculate_negatives(images: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """
    Calculates the negative of the images (Color and Grayscale).

    Parameters:
        - images(Dict[str, np.ndarray])    : The disctionary of images to calculate negatives.  
        
    Returns:
        - Dict[str, np.ndarray]     : Returns the dictionary of negative images.
    """
    if images is None:
        print(f"ERROR : Images not available.")
        return {}

    negative_images = {}
    for i, (key, image) in enumerate(images.items()):
        negative_images[key] = 255 - image   

    return negative_images  

def contrast_streching_minmax(images: Dict[str, np.ndarray], 
                                lower_intensity: int = 0, 
                                upper_intensity: int =  255) -> Dict[str, np.ndarray]:      
    """
    Performs the minmax contrast stretching of the images in the input dictionary by using lower_intensity 
    and upper_intensity as the bounds. 
    Parameters:
        - images(Dict[str, np.ndarray])    : The disctionary of images to calculate perform contrast streetching.
        - lower_intensity (int)                : The lower bound of the desired output intensity range. Default is 0.
        - upper_intensity (int)                : The upper bound of the desired output intensity range. Default is 255.
    Returns:
        - Dict[str, np.ndarray]     : Returns the dictionary of stretched images.
    """
    stretched_images = {}
    if images is None:
        print(f"ERROR : Images are not available.")
        return {}

    if not isinstance(lower_intensity, int)  or not isinstance(upper_intensity, int):
        print(f"ERROR : Intensity range vvules should be integer")
        return {}
    if lower_intensity < 0 or upper_intensity > 255:
        print(f"ERROR : Intensity range is not correct.")
        return {}

    for i, (key, image) in enumerate(images.items()):
        stretched_images[key] = cv2.normalize(image, None, lower_intensity, upper_intensity, cv2.NORM_MINMAX)
    
    return stretched_images

def large_log(images: Dict[str, np.ndarray], factor: int = 2) -> Dict[str, np.ndarray]:
    """
    Applies an exaggerated logarithmic transformation to the input image, 
    enhancing the contrast of dark regions by overly brightening them.

    Parameters:
        - images(Dict[str, np.ndarray])    : The disctionary of images to calculate perform contrast stretching.
        - factor (int)                     : The integer value to indicate the factor of log transformation. Default is 2.
    Returns:
    - Dict[str, np.ndarray]                      : Image with exaggerated logarithmic transformation.
    
    """
    stretched_images = {}
    if images is None:
        print(f"ERROR : Images are not available.")
        return {}    
    #prevent divide by zero error
    epsilon = 1e-9
    for i, (key, image) in enumerate(images.items()):
        c: float = 255 / np.log(1 + np.max(image)/factor)
        stretched_images[key] = np.array(c * (np.log(1 + image + epsilon) ** 3), dtype='uint8')    
    return stretched_images

def gamma_correction(images: Dict[str, np.ndarray], gamma: float) -> np.ndarray:
    """
    Performs gamma correction on the input image to adjust contrast based 
    on the specified gamma value.

    Parameters:
        - images(Dict[str, np.ndarray])    : The disctionary of images to calculate perform contrast streetching.
        - gamma (float): The gamma value for the correction. Values less than 1 
                   will brighten the image, and values greater than 1 will 
                   darken it.

    Returns:
    numpy.ndarray: Image after gamma correction.
    """
    stretched_images = {}
    if images is None:
        print(f"ERROR : Images are not available.")
        return {}

    for i, (key, image) in enumerate(images.items()):        
        stretched_images[key] = np.array(255 * (image / 255) ** gamma, dtype='uint8')    
    return stretched_images

def random_stretch(images: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Applies random contrast stretching by selectively boosting or 
    suppressing pixel intensities based on a random mask.
    Parameters:
        - images(Dict[str, np.ndarray])    : The disctionary of images to calculate perform contrast streetching.
    Returns:
        - numpy.ndarray: Image with random contrast stretching.
    """
    stretched_images = {}
    if images is None:
        print(f"ERROR : Images are not available.")
        return {}  
    for i, (key, image) in enumerate(images.items()): 
        # create a mask with random values.
        mask = np.random.choice([0, 1], size=image.shape, p=[0.50, 0.50]).astype(np.uint8)        
        # randomly masking image 
        stretched_images[key] = np.clip(np.where(mask == 1, image * 2, image // 2), 0,  255).astype(np.uint8)
    
    return stretched_images

def contrast_stretching_type(contrast_algo : ContrastType=ContrastType.CONTRAST_STRETCH_BASIC) -> None:

    plot_filename = str(contrast_algo.name)
    plot_subtracted_filename = plot_filename+"_Subtracted"
    stretched_images = {}
    subtracted_images = {}

    if contrast_algo == ContrastType.CONTRAST_STRETCH_BASIC:    

        stretched_images = contrast_streching_minmax(images, 30, 201)
        subtracted_images = subtract_images(images, stretched_images)

        plot_images(stretched_images, title = plot_filename, outputfile = output_plot_directory + plot_filename)
        # save_image(stretched_images, fileDirectory = output_images_directory, imageType = plot_filename)

        
        plot_images(subtracted_images, title = plot_subtracted_filename, outputfile = output_plot_directory + plot_subtracted_filename)
        # save_image(subtracted_images, fileDirectory = output_images_directory, imageType = plot_subtracted_filename)

    if contrast_algo == ContrastType.CONTRAST_STRETCH_REVERSED:    
           
        stretched_images = contrast_streching_minmax(images, 255, 0)
        subtracted_images = subtract_images(images, stretched_images)

        plot_images(stretched_images, title = plot_filename, outputfile = output_plot_directory + plot_filename)
        # save_image(stretched_images, fileDirectory = output_images_directory, imageType = plot_filename)

        
        plot_images(subtracted_images, title = plot_subtracted_filename, outputfile = output_plot_directory + plot_subtracted_filename)
        # save_image(subtracted_images, fileDirectory = output_images_directory, imageType = plot_subtracted_filename)
    
    if contrast_algo == ContrastType.CONTRAST_STRETCH_LARGELOG:    
           
        stretched_images = large_log(images, 3)
        subtracted_images = subtract_images(images, stretched_images)

        plot_images(stretched_images, title = plot_filename, outputfile = output_plot_directory + plot_filename)
        # save_image(stretched_images, fileDirectory = output_images_directory, imageType = plot_filename)

        
        plot_images(subtracted_images, title = plot_subtracted_filename, outputfile = output_plot_directory + plot_subtracted_filename)
        # save_image(subtracted_images, fileDirectory = output_images_directory, imageType = plot_subtracted_filename)
    
    if contrast_algo == ContrastType.CONTRAST_STRETCH_GAMMA:    
           
        stretched_images = gamma_correction(images, 0.01)
        subtracted_images = subtract_images(images, stretched_images)

        plot_images(stretched_images, title = plot_filename, outputfile = output_plot_directory + plot_filename)
        # save_image(stretched_images, fileDirectory = output_images_directory, imageType = plot_filename)

        
        plot_images(subtracted_images, title = plot_subtracted_filename, outputfile = output_plot_directory + plot_subtracted_filename)
        # save_image(subtracted_images, fileDirectory = output_images_directory, imageType = plot_subtracted_filename)
    
    if contrast_algo == ContrastType.CONTRAST_STRETCH_EXTREME:    
           
        stretched_images =  contrast_streching_minmax(images, 51, 90)
        subtracted_images = subtract_images(images, stretched_images)

        plot_images(stretched_images, title = plot_filename, outputfile = output_plot_directory + plot_filename)
        # save_image(stretched_images, fileDirectory = output_images_directory, imageType = plot_filename)

        
        plot_images(subtracted_images, title = plot_subtracted_filename, outputfile = output_plot_directory + plot_subtracted_filename)
        # save_image(subtracted_images, fileDirectory = output_images_directory, imageType = plot_subtracted_filename)

    if contrast_algo == ContrastType.CONTRAST_STRETCH_RANDOM:    
           
        stretched_images =  random_stretch(images)
        subtracted_images = subtract_images(images, stretched_images)

        plot_images(stretched_images, title = plot_filename, outputfile = output_plot_directory + plot_filename)
        # save_image(stretched_images, fileDirectory = output_images_directory, imageType = plot_filename)

        
        plot_images(subtracted_images, title = plot_subtracted_filename, outputfile = output_plot_directory + plot_subtracted_filename)
        # save_image(subtracted_images, fileDirectory = output_images_directory, imageType = plot_subtracted_filename)
    
import numpy as np

def interpolate_pixel(image, x, y):
    """
    Performs bilinear interpolation to get the pixel value at fractional coordinates (x, y).
    
    Args:
    - image (np.ndarray): The input image array.
    - x (float): The x-coordinate of the pixel.
    - y (float): The y-coordinate of the pixel.
    
    Returns:
    - float: The interpolated pixel value.
    """
    # Get the integer parts and the fractional parts of x and y
    x1, y1 = int(np.floor(x)), int(np.floor(y))
    x2, y2 = x1 + 1, y1 + 1
    
    # Ensure the coordinates are within the image boundaries
    if x1 < 0 or y1 < 0 or x2 >= image.shape[1] or y2 >= image.shape[0]:
        return 0  

    # Get the fractional part of x and y
    dx, dy = x - x1, y - y1
    
    # Get the four surrounding pixel values
    Q11 = image[y1, x1]  # Top-left
    Q21 = image[y1, x2]  # Top-right
    Q12 = image[y2, x1]  # Bottom-left
    Q22 = image[y2, x2]  # Bottom-right
    
    # Perform the bilinear interpolation
    R1 = (1 - dx) * Q11 + dx * Q21  
    R2 = (1 - dx) * Q12 + dx * Q22  
    
    # Interpolate in the y-direction
    P = (1 - dy) * R1 + dy * R2
    
    return P

def rotate_image(images: Dict[str, np.ndarray], angle: float = 0.0) -> Dict[str, np.ndarray]:
    """
    Rotates images based on the input angle value.

    Parameters:
        - images (Dict[str, np.ndarray]): The dictionary of input images.
        - angle (float)                 : The angle by which to rotate the images, in degrees.
    Returns:
        - Dict[str, np.ndarray]         : The dictionary of rotated images.
    """
    if images is None:
        print(f"ERROR : Images are not available.")
        return {}

    rotated_images = {}

    # Convert angle to radians
    radians = np.radians(angle)

    # Calculate the rotation matrix
    rotation_matrix = np.array([[np.cos(radians), -np.sin(radians)],
                                [np.sin(radians), np.cos(radians)]])

    for key, image in images.items():
        height, width = image.shape[:2]
        rotated_image = np.zeros_like(image)

        # Center of the image
        center_x, center_y = width // 2, height // 2

        # Iterate over every pixel in the output (rotated) image
        for i in range(height):
            for j in range(width):
                # Shift the current coordinates to the center (relative to the origin)
                x_shifted = j - center_x
                y_shifted = i - center_y

                # Apply the inverse of the rotation matrix to the coordinates
                original_coords = np.dot(rotation_matrix.T, [x_shifted, y_shifted])

                # Shift the coordinates back to their original position
                orig_x = original_coords[0] + center_x
                orig_y = original_coords[1] + center_y

                # Nearest neighbor interpolation: round to the nearest integer
                orig_x = int(round(orig_x))
                orig_y = int(round(orig_y))

                # Check if the computed coordinates are within the bounds of the input image
                if 0 <= orig_x < width and 0 <= orig_y < height:
                    rotated_image[i, j] = image[orig_y, orig_x]

        rotated_images[key] = rotated_image

    return rotated_images


if __name__ == "__main__":

    input_image_paths = ["assignments/assignment2/Flower.jpg",
                        "assignments/assignment2/Buddha.jpg"]
    images = read_images(input_image_paths)

    output_images_directory = "assignments/assignment2/Output/OutputImages/"
    output_plot_directory = "assignments/assignment2/Output/"

    # Display input images.
    plot_images(images, title = "Display input images", outputfile = output_plot_directory+"Input_Images.jpeg")

    # Shrink input images
    reduced_images = scale_image(images, scale_factor = -8)
    plot_images(reduced_images, title = "Display Reduced images", outputfile = output_plot_directory+"Reduced_Images.jpeg")
    save_image(reduced_images, fileDirectory = output_images_directory, imageType = str(ImageType.REDUCED.name))
    
    # Zoom reduced images
    zoomed_images = scale_image(reduced_images, scale_factor = 8)
    plot_images(zoomed_images, title = "Display Zoomed images", outputfile = output_plot_directory+"Zoomed_Images.jpeg")
    save_image(zoomed_images, fileDirectory = output_images_directory, imageType = str(ImageType.ZOOMED.name))

    # Subtracted Images 
    subtracted_images = subtract_images(images, zoomed_images)
    plot_images(subtracted_images, title = "Display Subtracted images", outputfile = output_plot_directory+"Subtracted_Images.jpeg")
    save_image(subtracted_images, fileDirectory = output_images_directory, imageType = str(ImageType.SUBTRACTED.name))
    
    # Negative Images 
    negative_images = calculate_negatives(images)
    plot_images(negative_images, title = "Display Negative images", outputfile = output_plot_directory+"Negative_Images.jpeg")
    save_image(negative_images, fileDirectory = output_images_directory, imageType = str(ImageType.NEGATIVE.name))

    # Contrast Stretching Examples

    for contrast_algo in ContrastType:
        contrast_stretching_type(contrast_algo)
        
    # Rotate images
    rotated_images = rotate_image(images, angle = 40.0)
    plot_images(rotated_images, title = "Display Rotated images", outputfile = output_plot_directory+"Rotated_Images.jpeg")
