import pixel_visions.utils as u
import pixel_visions.image_scaling as i
import pixel_visions.image_transformations as t
import pixel_visions.image_spatial_filtering as sf
import pixel_visions.image_segmentation as s
import numpy as np
import cv2
from itertools import product

# Define image file paths
assignment_folder = "assignments/assignment4/Output/"
input_image_path = ""
output_image_path = ""
plot_image_path = ""
output_images = {}
# grayscale_image = None
# Original_Image = None

def display_dog1_image(foldername: str = ""):

    if foldername != "":
        # Folder path
        assignment_folder = "assignments/assignment4/Output/"+foldername+"/"
    else:
        # Folder path
        assignment_folder = "assignments/assignment4/Output/"

    # Image file paths
    # input_image_path = "Input_Images/How-To-Master-Pet-Photography-940x529.png"
    input_image_path = "Input_Images/Dog1.jpg"
    output_image_path = assignment_folder + "Dog1.jpg"
    plot_image_path = assignment_folder + "Dog1 Images.jpg"

    # Define dictionary of output images    
    output_images['Original_Image'] = u.load_image(input_image_path, image_load_type=u.ImageLoadType.UNCHANGED)
    output_images['Color_Image']    = u.load_image(input_image_path, image_load_type=u.ImageLoadType.COLOR)
    output_images['Grayscale_Image'] =u.load_image(input_image_path, image_load_type=u.ImageLoadType.GRAYSCALE)
    # grayscale_image = output_images['Grayscale_Image']
    # Original_Image = output_images['Original_Image']

    u.plot_images(output_images,plot_image_path, title = "Original Image")

def display_dog2_image(foldername: str = ""):

    if foldername != "":
        # Folder path
        assignment_folder = "assignments/assignment4/Output/"+foldername+"/"
    else:
        # Folder path
        assignment_folder = "assignments/assignment4/Output/"

    input_image_path = "Input_Images/Dog2.png"

    output_image_path = assignment_folder + "Dog2.jpg"
    plot_image_path = assignment_folder + "Dog2_plot.jpg"

    # Define dictionary of output images    
    output_images['Original_Image'] = u.load_image(input_image_path, image_load_type=u.ImageLoadType.UNCHANGED)
    output_images['Color_Image']    = u.load_image(input_image_path, image_load_type=u.ImageLoadType.COLOR)
    output_images['Grayscale_Image'] =u.load_image(input_image_path, image_load_type=u.ImageLoadType.GRAYSCALE)

    # grayscale_image = output_images['Grayscale_Image']
    # Original_Image = output_images['Original_Image']

    u.plot_images(output_images,plot_image_path, title = "Original Image")

def problem1_EdgeDetection():
    display_dog1_image(foldername = "Problem1")
    assignment_folder = "assignments/assignment4/Output/Problem1/"
    print(f"~~~~~~~~~~~~Problem 1 : Edge Detection (10)~~~~~~~~~~~~~")
    apply_sobel(assignment_folder)
    apply_canny(assignment_folder)
    apply_prewitt(assignment_folder)
    apply_roberts(assignment_folder)

def apply_sobel(assignment_folder:str = ""):
    # Apply Sobel filter
    print(f"~~~~~~~~~~~~~ Sobel filter ~~~~~~~~~~~~~~~~~~~~~")
    
    sobel_filtered_images = {}
    sobel_filtered_images['Original_Image'] = output_images['Grayscale_Image']

    # Define parameter ranges
    ddepths = [cv2.CV_32F, cv2.CV_64F]
    dx_values = [1]
    dy_values = [1]
    ksizes = [3, 5, 7]
    scales = [1]
    deltas = [0]
    border_types = [cv2.BORDER_DEFAULT, cv2.BORDER_REFLECT]

    # Generate all combinations of parameters
    param_combinations = product(ddepths, dx_values, dy_values, ksizes, scales, deltas, border_types)
    for i, (ddepth, dx, dy, ksize, scale, delta, borderType) in enumerate(param_combinations):
        sobel_filtered_images["Sobel_Filtered_"+str(i)], parameters = sf.EdgeDetection.SOBEL.apply(output_images['Grayscale_Image'], ddepth=ddepth, dx=dx, dy=dy, 
                                       ksize=ksize, scale=scale, delta=delta, 
                                       borderType=borderType)
        sobel_filtered_images_path = assignment_folder + "1_Sobel_filtered_images_"+str(i)+".jpg"
        u.plot_images(sobel_filtered_images,sobel_filtered_images_path, title = parameters)
        sobel_filtered_images.pop("Sobel_Filtered_"+str(i))

def apply_canny(assignment_folder:str = ""):
    print(f"~~~~~~~~~~~~~ Canny filter ~~~~~~~~~~~~~~~~~~~~~")

    canny_filtered_images = {}
    canny_filtered_images['Original_Image'] = output_images['Grayscale_Image']

    # Define parameter ranges
    threshold1_values = [100]
    threshold2_values = [200, 250]
    aperture_sizes = [3, 5] 
    l2_gradients = [False] 

    # Generate all combinations of parameters
    param_combinations = product(threshold1_values, threshold2_values, aperture_sizes, l2_gradients)
    for i, (threshold1, threshold2, apertureSize, L2gradient) in enumerate(param_combinations):
        canny_filtered_images["Canny_filtered_"+str(i)], parameters = sf.EdgeDetection.CANNY.apply(output_images['Grayscale_Image'], threshold1=threshold1, threshold2=threshold2, 
                                    apertureSize=apertureSize, L2gradient=L2gradient)
        canny_filtered_images_path = assignment_folder + "1_Canny_filtered_images_"+str(i)+".jpg"
        u.plot_images(canny_filtered_images,canny_filtered_images_path, title = parameters)
        canny_filtered_images.pop("Canny_filtered_"+str(i))

def apply_prewitt(assignment_folder:str = ""):
    
    print(f"~~~~~~~~~~~~~ Prewitt filter ~~~~~~~~~~~~~~~~~~~~~")

    Prewitt_filtered_images = {}
    Prewitt_filtered_images["Original_Image"] = output_images['Grayscale_Image']

    ## Define parameter ranges
    scales = [1, 2]            
    thresholds = [0, 50, 100]  

    # # Generate all combinations of parameters
    param_combinations = product(scales, thresholds)
    # param_combinations = product(threshold1_values, threshold2_values, aperture_sizes, l2_gradients)
    for i, (scale, threshold) in enumerate(param_combinations):
        Prewitt_filtered_images["Prewitt_filtered_"+str(i)], parameters = sf.EdgeDetection.PREWITT.apply(output_images['Grayscale_Image'], scale=scale, threshold=threshold)
        Prewitt_filtered_images_path = assignment_folder + "1_Prewitt_filtered_images_"+str(i)+".jpg"
        u.plot_images(Prewitt_filtered_images,Prewitt_filtered_images_path, title = parameters)
        Prewitt_filtered_images.pop("Prewitt_filtered_"+str(i))

def apply_roberts(assignment_folder:str = ""):
    
    print(f"~~~~~~~~~~~~~ Roberts Filter ~~~~~~~~~~~~~~~~~~~~~")
    ## Apply Roberts filter
    Roberts_filtered_images = {}
    Roberts_filtered_images['Original_Image'] = output_images['Grayscale_Image']

    ## Define parameter ranges
    scales = [1, 2]            
    thresholds = [0, 50, 100]  

    # # Generate all combinations of parameters
    param_combinations = product(scales, thresholds)
    # param_combinations = product(threshold1_values, threshold2_values, aperture_sizes, l2_gradients)
    for i, (scale, threshold) in enumerate(param_combinations):
        Roberts_filtered_images["Roberts_filtered_"+str(i)], parameters = sf.EdgeDetection.ROBERTS.apply(output_images['Grayscale_Image'], scale=scale, threshold=threshold)
        Roberts_filtered_images_path = assignment_folder + "1_Roberts_filtered_images_"+str(i)+".jpg"
        u.plot_images(Roberts_filtered_images,Roberts_filtered_images_path, title = parameters)
        Roberts_filtered_images.pop("Roberts_filtered_"+str(i))

def problem2_Edge_Filter():
    display_dog1_image(foldername = "Problem1")
    assignment_folder = "assignments/assignment4/Output/Problem2/"
    # Apply Sobel filter
    print(f"~~~~~~~~~~~~~ Custome Sobel filter ~~~~~~~~~~~~~~~~~~~~~")
    sobel_filtered_images = {}
    sobel_filtered_images['Original_Image'] = output_images['Grayscale_Image']
    sobel_filtered_images_path = assignment_folder + "2_Sobel_filtered_images.jpg"
    sobel_filtered_images["Gaussian_Sobel_Filtered"], _ = sf.EdgeDetection.GAUSSIAN_DERIVATIVE.apply(output_images['Grayscale_Image'], ksize = 7, sigma = 1)
    u.plot_images(sobel_filtered_images,sobel_filtered_images_path, title = "Custom Sobel Filter")



def problem3_Histogram_based_segmentation():
    display_dog1_image(foldername = "Problem3")
    assignment_folder = "assignments/assignment4/Output/Problem3/"
    
    print(f"~~~~~~~~~~~~ Problem 3 : Histogram-based segmentation ~~~~~~~~~~~~")
    ## Histogram-based segmentation (20)
    Histogram_Segmented_Images = {}

    # plot the histogram of the grayscale image.
    Histogram_Plot_path = assignment_folder + "3_Histogram_Plot.jpg"
    t.plot_and_save_histogram(output_images['Grayscale_Image'], t.compute_histogram(output_images['Grayscale_Image']), filename=Histogram_Plot_path)

    #Compute the histogram-based segmentation and display binary segmentations

    # intensity_ranges = [(0,30), (60, 100), (120, 180), (250 ,256)]
    intensity_ranges = [(0,35), (80, 120), (160, 180), (220 ,245), (250 ,256)]
    Histogram_Segmented_Images = s.histogram_based_segmentation(output_images['Grayscale_Image'],
                                                                Histogram_Plot_path, 
                                                                ranges = intensity_ranges)
    images_to_plot = {}

    items = list(Histogram_Segmented_Images.items())
    count = 1
    for i in range(0, len(items), 1):
        images_to_plot['Original Image'] = output_images['Grayscale_Image']
        images_to_plot.update(dict(items[i:i + 1]))
        Histogram_Segmented_Images_path = assignment_folder + "3_Histogram_Segmented_Images_"+str(count)+".jpg"
        u.plot_images(images_to_plot,Histogram_Segmented_Images_path, title = "Histogram segmented Images")
        count+=1
        images_to_plot.clear()

def problem4_Noise_reduction():
    display_dog2_image(foldername = "Problem4")
    assignment_folder = "assignments/assignment4/Output/Problem4/"

    print("f~~~~~~~~~~~~~~~~~~~Noise reduction (20/0)~~~~~~~~~~~~")

    # Create Noisy images using Gaussian and Salt and Pepper methods.

    Noisy_Images = {}
    Noisy_Images['Original Image'] = output_images['Grayscale_Image']
    Noisy_Images_path = assignment_folder + "4_Noisy_Images.jpg"

    Noisy_Images["Gaussian"],parameters = u.add_noise(output_images['Grayscale_Image'], noise_type=u.NoiseType.GAUSSIAN, mean=0.0, sigma=0.05)
    Noisy_Images["SaltAndPepper"],parameters = u.add_noise(output_images['Grayscale_Image'], noise_type=u.NoiseType.SALT_AND_PEPPER, salt_prob=0.02, pepper_prob=0.02)
    u.plot_images(Noisy_Images,Noisy_Images_path, title = "Noisy Images")

    # Create averaged filtered images using the noisy images.

    Averaged_gaussian_images = {}
    Median_blurred_gaussian_images = {}
    Averaged_gaussian_images['Noisy Image'] = Noisy_Images["Gaussian"]
    Median_blurred_gaussian_images['Noisy Image'] =  Noisy_Images["Gaussian"]
    Averaged_gaussian_images_path = assignment_folder + "4_Averaged_gaussian_images.jpg"
    Median_blurred_gaussian_images_path = assignment_folder + "4_Median_blurred_gaussian_images.jpg"

    Averaged_sandp_images = {}
    Median_blurred_sandp_images = {}
    Averaged_sandp_images['Noisy Image'] = Noisy_Images["SaltAndPepper"]
    Median_blurred_sandp_images['Noisy Image'] =  Noisy_Images["SaltAndPepper"]
    Averaged_sandp_images_path = assignment_folder + "4_Averaged_sandp_images.jpg"
    Median_blurred_sandp_images_path = assignment_folder + "4_Median_blurred_sandp_images.jpg"

    for k in [3,5,7]:
        # Apply Average filter
        Averaged_gaussian_images[sf.SpatialFilter.AVERAGE.name+"_"+str(k)] = sf.SpatialFilter.AVERAGE.apply(Noisy_Images["Gaussian"], (k, k))
        Averaged_sandp_images[sf.SpatialFilter.AVERAGE.name+"_"+str(k)] = sf.SpatialFilter.AVERAGE.apply(Noisy_Images["SaltAndPepper"], (k, k))
        # Apply Median filter
        Median_blurred_gaussian_images[sf.SpatialFilter.MEDIAN.name+"_"+str(k)] = sf.SpatialFilter.MEDIAN.apply(Noisy_Images["Gaussian"], k)
        Median_blurred_sandp_images[sf.SpatialFilter.MEDIAN.name+"_"+str(k)] = sf.SpatialFilter.MEDIAN.apply(Noisy_Images["SaltAndPepper"], k)
    u.plot_images(Averaged_gaussian_images,Averaged_gaussian_images_path, title = "Averaged Gaussian Images")
    u.plot_images(Averaged_sandp_images,Averaged_sandp_images_path, title = "Averaged Salt and Pepper Images")

    u.plot_images(Median_blurred_gaussian_images,Median_blurred_gaussian_images_path, title = "Averaged Gaussian Images")
    u.plot_images(Median_blurred_sandp_images,Median_blurred_sandp_images_path, title = "Averaged Salt and Pepper Images")
    

if __name__ == "__main__":
    # problem1_EdgeDetection()
    # problem3_Histogram_based_segmentation()
    # problem4_Noise_reduction()
    problem2_Edge_Filter()

















