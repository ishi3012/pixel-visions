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
assignment_folder = "assignments/assignment5/Output/"
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

def problem1_edge_detection_with_color(image: np.ndarray, output_folderpath:str = "") -> None:
     
    print("Problem 1: Edge Detection with Color")

    assert image is not None and image.size > 0, "Input image is empty"
    
    grayscale_images = {}
    hsi_images = {}

    grayscale_images["Original_Image"] = image
    hsi_images["Original_Image"] = image

    if output_folderpath == "":
        output_folderpath = "assignments/assignment5/Output/Problem1/"

    # Convert an image from BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Convert image to grayscale by averaging the color channels 
    grayscale_image = np.mean(image_rgb, axis=2).astype(np.uint8)
    grayscale_images["Grayscale_image"] = grayscale_image

    # Apply Sobel edge detection
    sobel_x = cv2.Sobel(grayscale_image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(grayscale_image, cv2.CV_64F, 0, 1, ksize=3)

    sobel_edges = np.hypot(sobel_x, sobel_y)

    # Normalize to 0-255 and convert to uint8
    sobel_edges = np.uint8(sobel_edges / sobel_edges.max() * 255)  
    grayscale_images["Sobel_edges_GrayScale"] = sobel_edges

    u.plot_images(grayscale_images,output_folderpath+"GrayScale_SobelEdgeDetection.jpg", title = "Problem 1 GrayScale: Sobel Edge Detection")
    
    # Convert original image to HSI image
    
    # hsi_image = convert_rgb_to_HSI(image_rgb)
    # hsi_image = (hsi_image * 255).astype(np.uint8)  # Scale HSI image to 0-255 and convert to uint8
    hsi_image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    
    hsi_images["HSI_Image"] = hsi_image

    # get H, S , and I components 
    H, S, I = hsi_image[:, :, 0], hsi_image[:, :, 1], hsi_image[:, :, 2]

    # Apply sobel edge detection on the Intensitty component
    sobel_x_I = cv2.Sobel(I, cv2.CV_64F, 1, 0, ksize=3)  
    sobel_y_I = cv2.Sobel(I, cv2.CV_64F, 0, 1, ksize=3)  
    sobel_edges_I = np.hypot(sobel_x_I, sobel_y_I)
    sobel_edges_I = np.uint8(sobel_edges_I / sobel_edges_I.max() * 255)  
    hsi_images["HSI_Sobel_Intensity"] = sobel_edges_I

    # Apply sobel edge detection on the Hue component
    sobel_x_H = cv2.Sobel(H, cv2.CV_64F, 1, 0, ksize=3)  
    sobel_y_H = cv2.Sobel(H, cv2.CV_64F, 0, 1, ksize=3)  
    sobel_edges_H = np.hypot(sobel_x_H, sobel_y_H)
    sobel_edges_H = np.uint8(sobel_edges_H / sobel_edges_H.max() * 255)  
    hsi_images["HSI_Sobel_Hue"] = sobel_edges_H
    
    
    u.plot_images(hsi_images,output_folderpath+"HSI_SobelEdgeDetection.jpg", title = "Problem 1 HSI: Sobel Edge Detection")

def problem2a_color_segmentation(image: np.ndarray, output_folderpath:str = "") -> None:
    
    print("Problem 2a: Color Segmentation")

    assert image is not None and image.size > 0, "Input image is empty"
    
    output_images = {}
    hsi_images = {}

    output_images["Original_Image"] = image
    
    if output_folderpath == "":
        output_folderpath = "assignments/assignment5/Output/Problem2/"

    # Convert image to grayscale by averaging the color channels 
    grayscale_image = np.mean(image, axis=2).astype(np.uint8)
    output_images["Grayscale_image"] = grayscale_image

    _, thresholded_image = cv2.threshold(grayscale_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)   
    segmented_a = image.copy()

    # Green for background
    segmented_a[thresholded_image == 0] = [0, 255, 0]  
    output_images["Segmented_Image_2a"] = segmented_a 

    u.plot_images(output_images,output_folderpath+"2b_Color_Segmentation.jpg", title = "Problem 2 : Color Segmentation")

def problem2b_color_segmentation(image: np.ndarray,output_folderpath:str = "") -> None:
    
    print("Problem 2b: Color Segmentation")

    assert image is not None and image.size > 0, "Input image is empty"
    
    output_images = {}
    output_images["Original_Image"] = image
    
    channels = cv2.split(image)
    channel_names = ["Blue_Channel", "Green_Channel", "Red_Channel"]
    masks = []

    # Apply Otsu's thresholding to each channel and "blue out" the object regions
    for i, channel in enumerate(channels):
        _, channel_mask = cv2.threshold(channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        masks.append(channel_mask)

        # Apply the mask to "blue out" detected regions in this channel
        channel_blued = image.copy()
        channel_blued[channel_mask == 255] = [255, 0, 0]  # Apply blue color where mask is true
        output_images[f"{channel_names[i]}_Blued_Out"] = channel_blued

    
    combined_mask = cv2.bitwise_and(masks[0], cv2.bitwise_and(masks[1], masks[2]))
    segmented_combined = image.copy()
    segmented_combined[combined_mask == 255] = [255, 0, 0]
    output_images["Combined_Segmented_Image"] = segmented_combined

    # Plot images, assuming a utility function `u.plot_images` is available
    u.plot_images(output_images, output_folderpath + "2b_Color_Segmentation.jpg", title="Problem 2: Color Segmentation")
    
def problem2c_color_segmentation(image: np.ndarray,output_folderpath:str = "") -> None:
    
    print("Problem 2c: Color Segmentation")

    assert image is not None and image.size > 0, "Input image is empty"
    
    output_images = {}
    output_images["Original_Image"] = image

     # Convert image to HSV (HSI) color space
    hsi_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    output_images["HSI_Image"] = hsi_image
    
    # Define thresholds for hue to isolate red tones (with wrap-around at 0-10 and 160-179)
    lower_hue = np.array([0, 50, 50])    
    upper_hue = np.array([10, 255, 255])   
    lower_hue2 = np.array([160, 50, 50])   
    upper_hue2 = np.array([179, 255, 255])

    mask_hue1 = cv2.inRange(hsi_image, lower_hue, upper_hue)
    mask_hue2 = cv2.inRange(hsi_image, lower_hue2, upper_hue2)
    mask_combined = cv2.bitwise_or(mask_hue1, mask_hue2)
    # Create separate masks and blued-out images for R, G, B channels based on the HSV mask
    for i, color in enumerate(["Blue_Channel", "Green_Channel", "Red_Channel"]):
        
        channel_blued = image.copy()
        channel_mask = mask_combined if color == "Red_Channel" else np.zeros_like(mask_combined)
        
        if color == "Green_Channel":
            lower_green, upper_green = np.array([35, 50, 50]), np.array([85, 255, 255])
            channel_mask = cv2.inRange(hsi_image, lower_green, upper_green)
        elif color == "Blue_Channel":
            lower_blue, upper_blue = np.array([100, 50, 50]), np.array([140, 255, 255])
            channel_mask = cv2.inRange(hsi_image, lower_blue, upper_blue)
        
        channel_blued[channel_mask > 0] = [255, 0, 0]
        output_images[f"{color}_Blued_Out"] = channel_blued
    # Final combined blued-out image based on combined mask
    segmented_c = image.copy()
    segmented_c[mask_combined > 0] = [255, 0, 0]
    output_images["Combined_Segmented_Image"] = segmented_c
    # Plot images, assuming a utility function `u.plot_images` is available
    u.plot_images(output_images, output_folderpath + "2c_Color_Segmentation.jpg", title="Problem 2: Color Segmentation")
    
if __name__ == "__main__":
   
    output_images = display_input_image()
    
    problem1_images_path = 'assignments/assignment5/Output/Problem1/'
    
    problem1_edge_detection_with_color(output_images["Original_Image"], problem1_images_path)

    problem2_images_path = 'assignments/assignment5/Output/Problem2/'
    
    problem2a_color_segmentation(output_images["Original_Image"],problem2_images_path)

    problem2b_color_segmentation(output_images["Original_Image"],problem2_images_path)

    problem2c_color_segmentation(output_images["Original_Image"],problem2_images_path)















