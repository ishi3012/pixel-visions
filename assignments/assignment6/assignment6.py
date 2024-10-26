import pixel_visions.utils as u
import pixel_visions.image_scaling as i
import pixel_visions.image_transformations as t
import pixel_visions.image_spatial_filtering as sf
import pixel_visions.image_segmentation as s
import numpy as np
import cv2
from itertools import product
from typing import Dict

# Define image file paths
assignment_folder = "assignments/assignment6/Output/"
input_image_path = ""
output_image_path = ""
plot_image_path = ""
output_images = {}
# grayscale_image = None
# Original_Image = None

def display_input_image(foldername: str = ""):
    global assignment_folder, input_image_path, output_image_path, plot_image_path

    if foldername != "":
        # Folder path
        assignment_folder = assignment_folder + foldername+"/"
    else:
        assignment_folder = assignment_folder


    # Image file paths
    # input_image_path = "Input_Images/How-To-Master-Pet-Photography-940x529.png"
    input_image_path = "Input_Images/Tulip1.jpg"
    output_image_path = assignment_folder + "Tulip1.jpg"
    plot_image_path = assignment_folder + "Tulip Image.jpg"

    # Define dictionary of output images    
    output_images['Original_Image'] = u.load_image(input_image_path, image_load_type=u.ImageLoadType.UNCHANGED)
    # output_images['Color_Image']    = u.load_image(input_image_path, image_load_type=u.ImageLoadType.COLOR)
    output_images['Grayscale_Image'] =u.load_image(input_image_path, image_load_type=u.ImageLoadType.GRAYSCALE)

    u.plot_images(output_images,plot_image_path, title = "Original Image")
    return output_images

def binary_thresholding(image: np.ndarray) -> np.ndarray:
    
    assert image is not None and image.size > 0, "Input image is empty"

    _, binary_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return binary_image

def apply_erosion_dilation(binary_image: np.ndarray, iteration: int = 2) -> np.ndarray:

    assert binary_image is not None and binary_image.size > 0, "Input image is empty"
    
    structuring_element = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # Apply erosion twice
    eroded = cv2.erode(binary_image, structuring_element, iterations=iteration)
    # Apply dilation twice
    result = cv2.dilate(eroded, structuring_element, iterations=iteration)
    return result

def apply_dilation_erosion(binary_image: np.ndarray, iterations: int = 2) -> np.ndarray:

    assert binary_image is not None and binary_image.size > 0, "Input image is empty"

    structuring_element = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    
    dilated = cv2.dilate(binary_image, structuring_element, iterations)
    
    result = cv2.erode(dilated, structuring_element, iterations)

    return result

def problem1_erosion_and_dilation(image: np.ndarray, output_folderpath:str = "") -> None:

    print("Problem 1: Erosion and Dilation")
    assert image is not None and image.size > 0, "Input image is empty"

    output_images = {}
    output_images["Original_Image"] = image
    
    if output_folderpath == "":
        output_folderpath = "assignments/assignment6/Output/"

    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    output_images["Grayscale_image"] = grayscale_image
    binary_image = binary_thresholding(grayscale_image)
    output_images["Binary_image"] = binary_image

    erosion_dilation_result = apply_erosion_dilation(binary_image)    
    output_images["Erosion_dilation_result"] = erosion_dilation_result
    dilation_erosion_result = apply_dilation_erosion(binary_image)
    output_images["Dilation_Erosion_result"] = dilation_erosion_result
    u.plot_images(output_images,output_folderpath+"1_ErosionAndDilation.jpg", title = "Problem 1 : Erosion and Dilation ")

def apply_opening_closing(binary_image: np.ndarray, iteration:int = 2) -> Dict[str, np.ndarray]:

    structuring_element = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    
    # Dictionary to store results
    output_images = {}

    # Apply opening twice, then closing twice
    opened = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, structuring_element, iterations=iteration)
    opened_then_closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, structuring_element, iterations=iteration)
    output_images["Opened_then_Closed"] = opened_then_closed

    # Apply closing twice, then opening twice
    closed = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, structuring_element, iterations=iteration)
    closed_then_opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, structuring_element, iterations=iteration)
    output_images["Closed_then_Opened"] = closed_then_opened

    return output_images

def problem2_opening_and_closing(image: np.ndarray, output_folderpath:str = "") -> None:
    print("Problem 2: Opening and Closing ")
    
    assert image is not None and image.size > 0, "Input image is empty"

    output_images = {}
    # output_images["Original_Image"] = image
    
    if output_folderpath == "":
        output_folderpath = "assignments/assignment6/Output/"

    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    output_images["Grayscale_image"] = grayscale_image

    binary_image = binary_thresholding(grayscale_image)
    output_images["Binary_image"] = binary_image

    output_images.update(apply_opening_closing(binary_image))
   
    u.plot_images(output_images,output_folderpath+"2_OpeningAndClosing.jpg", title = "Problem 2: Opening and Closing")

def extract_boundaries(binary_image: np.ndarray) -> np.ndarray:
    
    structuring_element = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    eroded_image = cv2.erode(binary_image, structuring_element)
    boundary_image = cv2.subtract(binary_image, eroded_image)    
    return boundary_image

def extract_canny_edges(grayscale_image: np.ndarray, low_threshold: int = 100, high_threshold: int = 200) -> np.ndarray:
    
    edges = cv2.Canny(grayscale_image, low_threshold, high_threshold)    
    return edges


def problem3_boundary_extraction (image: np.ndarray, output_folderpath:str = "") -> None:
    print("Problem 3: Boundary Extraction ")
    
    assert image is not None and image.size > 0, "Input image is empty"

    output_images = {}
    # output_images["Original_Image"] = image
    
    if output_folderpath == "":
        output_folderpath = "assignments/assignment6/Output/"

    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    output_images["Grayscale_image"] = grayscale_image

    binary_image = binary_thresholding(grayscale_image)
    output_images["Binary_image"] = binary_image

    boundary_image = extract_boundaries(binary_image)
    output_images["Boundary_image"] = boundary_image
    canny_edges = extract_canny_edges(grayscale_image)
    output_images["Canny_edges"] = canny_edges
   
    u.plot_images(output_images,output_folderpath+"3_BoundaryExtraction.jpg", title = "Problem 3: Boundary Extraction ")

if __name__ == "__main__":

   
    output_images = display_input_image()
    
    # problem1_images_path = 'assignments/assignment6/Output/Problem1/'
    
    problem1_erosion_and_dilation(output_images["Original_Image"])
    problem2_opening_and_closing(output_images["Original_Image"])
    problem3_boundary_extraction(output_images["Original_Image"])


















