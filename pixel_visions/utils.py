import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from enum import Enum, auto
from typing import Dict

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
        # image_name = "Original_"+image_name
        images[image_name] = cv2.imread(path)
        image_count += 1 
        # Convert and save the colored image as grayscale as well in the dictionary.
        if len(images[image_name].shape) == 3:
            images["GS_"+image_name] = cv2.cvtColor(images[image_name], cv2.COLOR_BGR2GRAY)    
            
    print(f" Total images loaded = {image_count} out of {len(image_paths)}")
    return images

def plot_images(images: Dict[str, np.ndarray], outputfile: str, title: str = "") -> None:
    """
    Display the images with the given title.

    Parameters: 
        - images (Dict[str, np.ndarray]) : Dictionary of images to plot
        - title (str)                    : Text to display as title of the plot
        - outputfile(str)                : The output file path. 
    """

    if images is None:
        raise ValueError(f"ERROR: Images not available.")

    if len(images) > 1:
        num_cols = 2
    else:
        num_cols = 1
    num_rows = (len(images) + num_cols - 1) // num_cols

    plt.figure(figsize = (10, 5 * num_rows))

    for i, (key, image) in enumerate(images.items()):
        plt.subplot(num_rows, num_cols, i+1)

        if len(image.shape) == 3:
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        else:
            plt.imshow(image, cmap = "gray")
        
        plt.axis("on")
        plt.title(key, fontsize = 10)

    plt.suptitle(title, fontsize = 10)
    plt.tight_layout(rect = [0, 0, 1, 0.95])
    plt.savefig(outputfile)
    plt.show()

def save_images(images: Dict[str, np.ndarray], fileDirectory: str, imageType : str = "Original") -> None:
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

def subtract_images(images1: Dict[str, np.ndarray] ,images2: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """
    Subtracts two images (image1 - image2) and save the resultant imag2. 

    Parameters:
        - images1(Dict[str, np.ndarray])    : The disctionary of images from which the other image will be subtracted. 
        - images2(Dict[str, np.ndarray])    : The disctionary of images to subtract from first image.
    Returns:
        - Dict[str, np.ndarray]    : Returns the dictionary of subtracted images.

    """ 
    subtracted_images = {}

    if images1 is None or images2 is None:
        raise ValueError("One or both of the image dictionaries are not available.")
    
    for key in images1.keys():
        if key not in images2:
            print(f"WARNING: Key '{key}' not found in images2. Skipping.")
            continue
        
        # Convert images to float to avoid clipping issues
        image1 = images1[key].astype(np.float32)
        image2 = images2[key].astype(np.float32)

        # Subtract images
        # subtracted_image = cv2.subtract(image1, image2)
        subtracted_image = cv2.absdiff(image1, image2)

        # Normalize to the range [0, 255]
        cv2.normalize(subtracted_image, subtracted_image, 0, 255, cv2.NORM_MINMAX)

        # Convert back to uint8
        subtracted_images[key] = subtracted_image.astype(np.uint8)

    return subtracted_images

    


def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    Normalize the pixel values of an image to the range [0, 255].

    Parameters:
    -----------
    image : np.ndarray
        A 2D numpy array representing the grayscale image.

    Returns:
    --------
    np.ndarray
        A 2D numpy array representing the normalized image.
    """
    # Find the minimum and maximum pixel values
    min_val = np.min(image)
    max_val = np.max(image)

    # Avoid division by zero if all pixel values are the same
    if max_val - min_val == 0:
        return np.zeros_like(image, dtype=np.uint8)

    # Normalize the pixel values to the range [0, 255]
    normalized_image = (image - min_val) / (max_val - min_val) * 255.0

    # Convert to uint8
    return normalized_image.astype(np.uint8)