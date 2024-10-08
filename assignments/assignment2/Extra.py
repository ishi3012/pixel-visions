import cv2
import numpy as np
from typing import List



def __init__(self, image_path: str, output_image_directory: str = "assignments/assignment2/Output/" ) -> None:
    """
    Initializes the class by loading the input image from the specified path.

    Parameters:
        - image_path(str)               : The string that represents the path of the input image file.  
        - output_image_directory(str)   : The string to define the output image file directory.
    """
    self.image = cv2.imread(image_path)
    self.output_directory = output_image_directory        

    if self.image is None:
        print(f"Image not found at: {image_path}")
        raise FileNotFoundError
    else:
        # Convert the color image to grayscale image.
        if len(self.image.shape) == 3:
            self.gray_scale_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        else:
            self.gray_scale_image = self.image

        # Save grayscale image
        success = cv2.imwrite(self.output_directory+"gray_scale_image.jpg", self.gray_scale_image)

        if success:
            print(f"Image saved successfully as {self.output_directory+"gray_scale_image.jpg"}.\n")
        else:
            print(f"Failed to save the image - gray_scale_image.jpg.\n")
            
        print(f"The image is successfully loaded.")

def save_image(self, image:np.ndarray, filename = "output.jpg") -> None:
    """
    Saves the image file.

    Parameters:
    - image(np.ndarray)        : The image to be saved.
    - filename(str)            : The name of the output image file. Default is "output.jpg".
    """
    if image is None:
        print(f"Image not found.")
    else:
        # Save shrunk color output image
        success = cv2.imwrite(self.output_directory + filename, image)

        if success:
            print(f"Image saved successfully as {self.output_directory + filename}.\n")
        else:
            print(f"Failed to save the image.\n")

def plot_images(self, title: str, images: list):
    """
    Display the images with the given title.
    """
    plt.figure(figsize=(15, 5))
    for i, img in enumerate(images):
        plt.subplot(1, len(images), i + 1)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.axis('off')
    plt.suptitle(title)
    plt.show()

def shrink_image(self, scale_factor: int = 1) -> None:
    """
    Shrinks the image by the given scaling factor. 

    Parameters:
    - scale_factor(int)         :   The scaling factor by which the input image should be shrunk.
    """
    for image, filename in zip([self.image, self.gray_scale_image], ["Reduced_color_image.jpg","Reduced_grayscale_image.jpg"]):
        
        height, width = image.shape[0], image.shape[1]
        reduced_height, reduced_width = height // scale_factor, width // scale_factor

        if len(image.shape) > 2:
            self.reduced_color_image = np.zeros((reduced_height, reduced_width, image.shape[2]))
            self.reduced_color_image = image[::scale_factor, ::scale_factor]
            self.save_image(self.reduced_color_image, filename)
        else:
            self.reduced_grayscale_image = np.zeros((reduced_height, reduced_width))
            self.reduced_grayscale_image = image[::scale_factor, ::scale_factor] 
            self.save_image(self.reduced_grayscale_image, filename)

def zoom_image(self, scale_factor: int = 1) -> None:
    """
    Zooms the reduced images by the given scaling factor. 

    Parameters:
    - scale_factor(int)         :   The scaling factor by which the input image should be zoomed.
    """
    for image, filename in zip([self.reduced_color_image, self.reduced_grayscale_image], ["Zoomed_color_image.jpg","Zoomed_grayscale_image.jpg"]):
        height, width = image.shape[0], image.shape[1]
        increased_height, increased_width = height * scale_factor, width * scale_factor
        
        if len(image.shape)>2:
            
            # Initialize a zoomed image
            self.zoomed_color_image = np.zeros((increased_height, increased_width, image.shape[2]), dtype=np.uint8)
            
            # Populate the zoomed image using pixel replication
            for i in range(increased_height):
                for j in range(increased_width):                    
                    original_i = i // scale_factor
                    original_j = j // scale_factor
                    self.zoomed_color_image[i, j] = image[original_i, original_j].astype(np.uint8)   
            
            self.save_image(self.zoomed_color_image, filename)                    
        else:
            
            # Initialize a zoomed image
            self.zoomed_grayscale_image = np.zeros((increased_height, increased_width), dtype=np.uint8)

            # Populate the zoomed image using pixel replication
            for i in range(increased_height):
                for j in range(increased_width):                    
                    original_i = i // scale_factor
                    original_j = j // scale_factor
                    self.zoomed_grayscale_image[i, j] = image[original_i, original_j].astype(np.uint8)   
            
            self.save_image(self.zoomed_grayscale_image, filename) 

def scale_image(self, scale_factor: int = 1) -> None:
    """
    Scales the image by the given scaling factor. 

    Parameters:
    - scale_factor(int)         :   The scaling factor by which the input image should be shrunk or zoomed based on the value
                                    If value is positive the image will be zoomed and if the value is negative the image will be shrunk.
                                    Default is 1.                
    """

    if self.image is None:
        print(f"File not found error.")            
    else:

        if scale_factor < 0:
            self.shrink_image(abs(scale_factor))
        else:
            self.zoom_image(scale_factor)
            
def compare_images(self) -> None:
    """
    Subtracts the zoomed images (color and grayscale) from the original image.      

    """
    for original_image,zoomed_image, filename in zip([self.image,self.gray_scale_image],[self.zoomed_color_image, self.zoomed_grayscale_image], ["Subtracted_Color_Image.jpg", "Subtracted_Grayscale_Image.jpg"]):
        self.subtract_images(original_image,zoomed_image, filename)
        

def subtract_images(self, image1: np.ndarray, image2: np.ndarray, filename: str = "output.jpg") -> None:
    """
    Subtracts two images (image1 - image2) and save the resultant imag2. 

    Parameters:
        - image1(np.ndarray)    : The first image from which the other image will be subtracted. 
        - image2(np.ndarray)    : The second image to subtract from first image.
        - filename(str)         : Saves the subtracted image with the given filename. 

    """        
    if image1 is None or image2 is None:
        print(f"Images are not availabe to compare.")

    else:
        subtracted_image = np.clip(np.abs(image1.astype(np.int16) - image2.astype(np.int16)), 0, 255).astype(np.uint8)
        #subtracted_image = np.abs(image1 - image2).astype(np.uint8)
        self.save_image(subtracted_image, filename)

def calculate_negatives(self) -> None:
    """
    Calculates the negative of the images (Color and Grayscale).
    """
    for image, filename in zip([self.image, self.gray_scale_image], ["Negative_color_image.jpg","Negative_grayscale_image.jpg"]):
        negative_image = 255 - image            
        self.save_image(negative_image, filename)

def contrast_streching_minmax(self, image_path: str, lower_intensity: int, upper_intensity: int ) -> None:
    """
    Performs basic contrast stretching by mapping the intensity range of the input image to a specified range [lower_intensity, upperintensity].

    Parameters:
        - image_path (str)          : Path to the image file.
        - lower_intensity (int)     : The lower bound of the intensity value in the output image.
        - upper_intensity (int)     : The upper bound of the intensity value in the output image.             
    """
    pass
    
                        
                


            
if __name__ == "__main__":

    input_image_path = "assignments/assignment2/Flower.jpg"

    a2 = Assignment2(input_image_path)

    # Shrink the image by a factor of 8
    a2.scale_image(-8)

    # zoom the reduced image by factor of 8
    a2.scale_image(8)

    # subtract original image and the zoomed image
    a2.compare_images()
    # a2.subtract_images(a2.image, a2.zoomed_grayscale_image)

    a2.calculate_negatives()
