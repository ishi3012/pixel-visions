import pixel_visions.utils as u
import pixel_visions.image_scaling as i
import pixel_visions.image_transformations as t
import pixel_visions.image_spatial_filtering as sf
import numpy as np

# Load, save and plot the input image

# Folder path
assignment_folder = "assignments/assignment3/Output/"

# Image file paths
input_image_path = "Input_Images/KASHMIR.jpg"
output_image_path = assignment_folder + "Nature1.jpg"
plot_image_path = assignment_folder + "Nature1_plot.jpg"


# Define dictionary of output images
output_images = {}
original_image = u.load_image(input_image_path, image_load_type=u.ImageLoadType.UNCHANGED)
output_images['Original_Image'] = original_image

color_image = u.load_image(input_image_path, image_load_type=u.ImageLoadType.COLOR)
output_images['Color_Image'] = color_image

grayscale_image = u.load_image(input_image_path, image_load_type=u.ImageLoadType.GRAYSCALE)
output_images['Grayscale_Image'] = grayscale_image

# u.plot_images(output_images,plot_image_path, title = "Original Image")

# # 1. Histogram Equalization (20/10) 

# image_histogram = {}
# image_histogram_path = assignment_folder + "1_Histogram_plot.jpg"
# image_equalized_histogram_path = assignment_folder + "1_Equalized_Histogram_plot.jpg"
# image_histogram_image_path = assignment_folder + "1_Histogram_Image_plot.jpg"

# image_histogram["Original Image"] = grayscale_image
# t.display_image_histogram(grayscale_image, image_histogram_path)
# image_histogram["Equalized_Image"] = t.display_equalize_histogram(grayscale_image, image_equalized_histogram_path)

# ### Apply a local enhancement approach 
# local_enhanced_images = {}
# local_enhanced_images["Original Image"] = grayscale_image
# local_enhanced_images_path = assignment_folder + "2_Local Enhanced Images.jpg"

# for method in t.LocalHistogramEqualizationMethod:
#     local_enhanced_images[method] = t.apply_local_histogram_equalization(grayscale_image, method)

# u.plot_images(local_enhanced_images,local_enhanced_images_path, title = "Local Enhanced Images")

# # Filtering (25/15) 

# filtered_images = {}
# filtered_images["Original Image"] = grayscale_image #original_image
# filtered_grayscale_image_plot = assignment_folder + "3_Filtered_GrayScale_Image.jpg"
# filtered_color_image_plot = assignment_folder + "3_Filtered_Color_Image.jpg"

# kernel = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])

# # Apply filter to grayscale image
# filtered_images["Filtered Image"] = sf.apply_filter(grayscale_image, kernel)
# u.plot_images(filtered_images,filtered_grayscale_image_plot, title = "Filtered GrayScale Image")
# filtered_images.clear()

# # Apply filter to grayscale image
# filtered_images["Original Image"] = original_image
# filtered_images["Filtered Image"] = sf.apply_filter(original_image, kernel)
# u.plot_images(filtered_images,filtered_color_image_plot, title = "Filtered GrayScale Image")

filtered_images = {}
filtered_images["Original Image"] = original_image
filtered_color_image_plot = assignment_folder + "3_Filtered_Color_Image.jpg"

# Apply Gaussian filter
filtered_images[sf.SpatialFilter.GAUSSIAN.name] = sf.SpatialFilter.GAUSSIAN.apply(original_image, (5, 5), 0)


# Apply Average filter
filtered_images[sf.SpatialFilter.AVERAGE.name] = sf.SpatialFilter.AVERAGE.apply(original_image, (5, 5))


# Apply Median filter
filtered_images[sf.SpatialFilter.MEDIAN.name] = sf.SpatialFilter.MEDIAN.apply(original_image, 5)


# Apply Bilateral filter
filtered_images[sf.SpatialFilter.BILATERAL.name] = sf.SpatialFilter.BILATERAL.apply(original_image, 9, 75, 75)


# Apply Box filter
filtered_images[sf.SpatialFilter.BOX.name] = sf.SpatialFilter.BOX.apply(original_image, -1, (5, 5), normalize=True)
u.plot_images(filtered_images,filtered_color_image_plot, title = "Filtered GrayScale Image")
