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
    input_image_path = "Input_Images/Chicago_River1.jpg"
    output_image_path = assignment_folder + "Input_Image.jpg"
    plot_image_path = assignment_folder + "Input_Images.jpg"

    # Define dictionary of output images    
    output_images['Original_Image'] = u.load_image(input_image_path, image_load_type=u.ImageLoadType.UNCHANGED)
    # output_images['Color_Image']    = u.load_image(input_image_path, image_load_type=u.ImageLoadType.COLOR)
    output_images['Grayscale_Image'] =u.load_image(input_image_path, image_load_type=u.ImageLoadType.GRAYSCALE)

    u.plot_images(output_images,plot_image_path, title = "Original Image")
    return output_images

def problem1_erosion_and_dilation(image: np.ndarray, output_folderpath:str = "") -> None:

    print("Problem 1: Erosion and Dilation")
    assert image is not None and image.size > 0, "Input image is empty"

    output_images = {}
    # output_images["Original_Image"] = image   
    
    if output_folderpath == "":
        output_folderpath = "assignments/assignment6/Output/Problem1/"

    # print the original image
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    output_images["Grayscale_image"] = grayscale_image

    _, binary_image = cv2.threshold(grayscale_image, np.mean(grayscale_image), 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    output_images["Binary_image"] = binary_image

    u.plot_images(output_images,output_folderpath+"1a_ErosionAndDilation.jpg", title = "Problem 1 : Erosion and Dilation ")

    # output_images.pop("Original_Image")
    output_images.pop("Grayscale_image")
    output_images.pop("Binary_image")

    # print the Eroded and Dilated
    structuring_element = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    eroded = cv2.erode(binary_image, structuring_element, iterations=2)
    output_images["Eroded"] = eroded

    dilated = cv2.dilate(binary_image, structuring_element, iterations=2)  
    output_images["Dilated"] = dilated

    u.plot_images(output_images,output_folderpath+"1b_ErosionAndDilation.jpg", title = "Problem 1 : Erosion and Dilation ")
    output_images.pop("Eroded")
    output_images.pop("Dilated")

    dilation_after_erosion = cv2.dilate(eroded, structuring_element, iterations=2)
    output_images["Dilation_after_erosion"] = dilation_after_erosion

    erosion_after_dilation = cv2.erode(dilated, structuring_element, iterations=2)
    output_images["Erosion_after_dilation"] = erosion_after_dilation

    u.plot_images(output_images,output_folderpath+"1c_ErosionAndDilation.jpg", title = "Problem 1 : Erosion and Dilation ")

def problem2_opening_and_closing(image: np.ndarray, output_folderpath:str = "") -> None:
    print("Problem 2: Opening and Closing ")
    
    assert image is not None and image.size > 0, "Input image is empty"
    output_images = {}
    # output_images["Original_Image"] = image  
    
    if output_folderpath == "":
        output_folderpath = "assignments/assignment6/Output/Problem2/"

    # print the original image
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    output_images["Grayscale_image"] = grayscale_image

    _, binary_image = cv2.threshold(grayscale_image, np.mean(grayscale_image), 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    output_images["Binary_image"] = binary_image

    u.plot_images(output_images,output_folderpath+"2a_OpeningAndClosing.jpg", title = "Problem 2: Opening and Closing")

    # output_images.pop("Original_Image")
    output_images.pop("Grayscale_image")
    output_images.pop("Binary_image")

    # Print opening and closing
    structuring_element = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    opened_image_1  = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, structuring_element)
    output_images["Opened_image_1"] = opened_image_1
    opened_image_2 = cv2.morphologyEx(opened_image_1, cv2.MORPH_OPEN, structuring_element)
    output_images["Opened_image_2"] = opened_image_2

    u.plot_images(output_images,output_folderpath+"2b_OpeningAndClosing.jpg", title = "Problem 2: Opening and Closing")

    # output_images.pop("Original_Image")
    output_images.pop("Opened_image_1")
    output_images.pop("Opened_image_2")

    closed_image_1  = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, structuring_element)
    output_images["Closed_image_1"] = closed_image_1
    closed_image_2 = cv2.morphologyEx(closed_image_1, cv2.MORPH_CLOSE, structuring_element)
    output_images["Closed_image_2"] = closed_image_2

    u.plot_images(output_images,output_folderpath+"2c_OpeningAndClosing.jpg", title = "Problem 2: Opening and Closing")
    output_images.pop("Closed_image_1")
    output_images.pop("Closed_image_2")

    opened_after_closing_1 = cv2.morphologyEx(closed_image_2, cv2.MORPH_OPEN, structuring_element)
    output_images["Opened_after_closing_1"] = opened_after_closing_1
    opened_after_closing_2 = cv2.morphologyEx(opened_after_closing_1, cv2.MORPH_OPEN, structuring_element)
    output_images["Opened_after_closing_2"] = opened_after_closing_2
   
    u.plot_images(output_images,output_folderpath+"2d_OpeningAndClosing.jpg", title = "Problem 2: Opening and Closing")

def problem3_boundary_extraction (image: np.ndarray, output_folderpath:str = "") -> None:
    print("Problem 3: Boundary Extraction ")
    
    assert image is not None and image.size > 0, "Input image is empty"
    output_images = {}
    # output_images["Original_Image"] = image  
    
    if output_folderpath == "":
        output_folderpath = "assignments/assignment6/Output/Problem3/"

    # print the original image
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    output_images["Grayscale_image"] = grayscale_image

    _, binary_image = cv2.threshold(grayscale_image, np.mean(grayscale_image), 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    output_images["Binary_image"] = binary_image

    u.plot_images(output_images,output_folderpath+"3a_BoundaryExtraction.jpg", title = "Problem 3: Boundary Extraction")

    # output_images.pop("Original_Image")
    output_images.pop("Grayscale_image")
    output_images.pop("Binary_image")

    # Print opening and closing
    structuring_element = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    eroded = cv2.erode(binary_image, structuring_element)
    output_images["Eroded"] = eroded

    boundary = cv2.subtract(binary_image, eroded)
    output_images["Boundary"] = boundary

    u.plot_images(output_images,output_folderpath+"3b_BoundaryExtraction.jpg", title = "Problem 3: Boundary Extraction")
        
    # output_images.pop("Original_Image")
    output_images.pop("Eroded")
    output_images.pop("Boundary")

    canny_edges = cv2.Canny(image, 100, 200)
    output_images["Canny_edges"] = canny_edges
    
    u.plot_images(output_images,output_folderpath+"3c_BoundaryExtraction.jpg", title = "Problem 3: Boundary Extraction")

if __name__ == "__main__":
   
    output_images = display_input_image()    
    problem1_erosion_and_dilation(output_images["Original_Image"])
    problem2_opening_and_closing(output_images["Original_Image"])
    problem3_boundary_extraction(output_images["Original_Image"])


















