import pixel_visions.utils as u
import pixel_visions.image_scaling as i
import pixel_visions.image_transformations as t

# Load, save and plot the input image

# Image file paths
input_image_path = "Input_Images/Nature1.jpg"
output_image_path = "assignments/assignment2/Output/Nature1.jpg"
plot_image_path = "assignments/assignment2/Output/Nature1_plot.jpg"


# Define dictionary of output images
output_images = {}


original_image = u.load_image(input_image_path, image_load_type=u.ImageLoadType.UNCHANGED)
output_images['Original_Image'] = original_image

color_image = u.load_image(input_image_path, image_load_type=u.ImageLoadType.COLOR)
output_images['Color_Image'] = color_image

grayscale_image = u.load_image(input_image_path, image_load_type=u.ImageLoadType.GRAYSCALE)
output_images['Grayscale_Image'] = grayscale_image

u.plot_images(output_images,plot_image_path)

# Problem 1: Images Scaling by Pixel Replication (15/15)

# Scale image by a factor of 8
scaled_images = {}
scaled_image_path = "assignments/assignment2/Output/1_Image_Scaling_plot.jpg"

scaled_images['Original_Image'] = original_image
scaled_images['Shrunk_image'] = i.scale_image(original_image, scale = 0.125)
scaled_images['Zoomed_image'] = i.scale_image(scaled_images['Shrunk_image'], scale = 8, original_dims=original_image.shape[:2])
scaled_images['Subtracted_image'] = u.subtract_images(original_image, scaled_images['Zoomed_image'])
u.plot_images(scaled_images, scaled_image_path, title = "Image Scaling", axis="on")

# Problem 2: Basic Grey Level Transformations (5/5)
Transformed_images = {}
Transformed_image_path = "assignments/assignment2/Output/2_Transformed_image_plot.jpg"
Transformed_images['Original_Image'] = original_image
Transformed_images["Negative_Image"] = t.negative_image(original_image)
u.plot_images(Transformed_images, Transformed_image_path, title = "Transformed Images", axis="on")

# Perform contrast stretching
streched_images = {}
streched_images_path = "assignments/assignment2/Output/2_Contrast_Stretching_plot.jpg"

streched_images['Original_Image'] = original_image
streched_images.update(t.compare_contrast_stretching(original_image, 
                                                    lower_intensity=0, 
                                                    upper_intensity=255, 
                                                    factor=2,
                                                    gamma = 0.001,
                                                    extremeLow = 51,
                                                    extremeHigh = 90 ))
u.plot_images(streched_images, streched_images_path, title = "Contrast Stretching")



# Problem 3: Image Transformations (10/0) (481 only)

rotated_image = {}
rotated_image_path = "assignments/assignment2/Output/3_Rotated_Image_plot.jpg"
rotated_image['Original_Image'] = original_image
rotated_image['Rotated Image_pixel_operation'] = (t.rotate_image_by_pixel(original_image, 30))
rotated_image['Rotated Image_using_OpenCV'] = (t.rotate_image(original_image, -30))
u.plot_images(rotated_image, rotated_image_path, title = "Rotated Images")