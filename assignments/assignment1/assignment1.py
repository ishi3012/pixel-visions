import pixel_visions.utils as u
import pixel_visions.interpolation as i

# Image file paths
input_image_path = "Input_Images/Nature1.jpg"
output_image_path = "assignments/assignment1/Output/Nature1.jpg"
plot_image_path = "assignments/assignment1/Output/Nature1_plot.jpg"


# Define a dictionaries of output images
output_images = {}


original_image = u.load_image(input_image_path, image_load_type=u.ImageLoadType.UNCHANGED)
output_images['Original_Image'] = original_image

color_image = u.load_image(input_image_path, image_load_type=u.ImageLoadType.COLOR)
output_images['Color_Image'] = color_image

grayscale_image = u.load_image(input_image_path, image_load_type=u.ImageLoadType.GRAYSCALE)
output_images['Grayscale_Image'] = grayscale_image

u.plot_images(output_images,plot_image_path)

# 1. Getting familiar with image manipulation in your favorite programming language 

u.display_image_statistics(original_image)

pixel_stats = u.get_pixel_statistics(original_image)

pixel_means = [stat[2] for stat in pixel_stats]

binary_image = u.threshold_image(original_image, pixel_means)

# images["BinaryImage"] = binary_image

# 2. Image Interpolation (5/5)

resized_images = {}
scale = 0.1
width, height = original_image.shape[1], original_image.shape[0]
resized_image_path = "assignments/assignment1/Output/2_Resized_Images_plot.jpg"
restored_image_path = "assignments/assignment1/Output/2_Restored_Images_plot.jpg"
subtracted_image_path = "assignments/assignment1/Output/2_Subtracted_Images_plot.jpg"
grayscale_reduceded_image_path = "assignments/assignment1/Output/3_GrayScale_Reduced_Images_plot.jpg"

resized_images['Original_Image'] = original_image
# resized_images_dict = i.compare_interpolations(original_image,scale = 0.1)
resized_images.update(i.compare_interpolations(original_image,scale = scale))
u.plot_images(resized_images,resized_image_path, title = "RESIZED IMAGES")

restored_images = {}
restored_images['Original_Image'] = original_image
width, height = original_image.shape[1], original_image.shape[0]

for key, image in resized_images.items():
    if key == 'NEAREST_NEIGHBOR':
        restored_images[key] = i.restore_image(resized_images[key], (width, height), i.InterpolationMethod.NEAREST_NEIGHBOR)
    elif key == 'BILINEAR':
        restored_images[key] = i.restore_image(resized_images[key], (width, height), i.InterpolationMethod.BILINEAR)
    elif key == 'BICUBIC':
        restored_images[key] = i.restore_image(resized_images[key], (width, height), i.InterpolationMethod.BICUBIC)

u.plot_images(restored_images,restored_image_path, title = "RESTORED IMAGES")

subtracted_images = {}
subtracted_images['Original_Image'] = original_image

for key, image in restored_images.items():
    subtracted_images[key] = u.subtract_images(original_image, restored_images[key])
u.plot_images(subtracted_images,subtracted_image_path, title = "SUBTRACTED IMAGES")

# 3. Reducing the Number of Gray Levels in an Image 
reduced_images = {}
reduced_images['GraySclae_Image'] = grayscale_image
for i in range(7, 0, -1):
    scale = 2 ** i 
    reduced_images['Reduced_'+str(scale)] = u.reduce_gray_levels(grayscale_image,scale=scale)
u.plot_images(reduced_images,grayscale_reduceded_image_path, title = "GrayScale Reduced IMAGES")







