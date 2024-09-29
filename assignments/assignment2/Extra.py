import cv2
import numpy as np


class Assignment2:
    """
    A class to implements image processing tasks:

    1. Shrinking and Zooming an input image.
    2. Basic Grey Level Transformations.
    3. Image Transformations.
    
    """
    

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

            self.reduced_image = np.zeros((reduced_height, reduced_width))
            self.reduced_image = image[::scale_factor, ::scale_factor]

            self.save_image(self.reduced_image, filename)

    def zoom_image(self, scale_factor: int = 1) -> None:
        """
        Zooms the image by the given scaling factor. 

        Parameters:
        - scale_factor(int)         :   The scaling factor by which the input image should be zoomed.
        """
        for image, filename in zip([self.image, self.gray_scale_image], ["Zoomed_color_image.jpg","Zoomed_grayscale_image.jpg"]):
            height, width = image.shape[0], image.shape[1]
            increased_height, increased_width = height * scale_factor, width * scale_factor

            if len(image.shape)>2:
                # Initialize a zoomed image
                self.zoomed_image = np.zeros((increased_height, increased_width, image.shape[2]), dtype=np.uint8)
            else:
                # Initialize a zoomed image
                self.zoomed_image = np.zeros((increased_height, increased_width), dtype=np.uint8)
            
            # Populate the zoomed image using pixel replication
            for i in range(increased_height):
                for j in range(increased_width):
                    
                    original_i = i // scale_factor
                    original_j = j // scale_factor
                    self.zoomed_image[i, j] = image[original_i, original_j].astype(np.uint8)   


        self.save_image(self.zoomed_image, filename)





        

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
            
if __name__ == "__main__":

    input_image_path = "assignments/assignment2/Flower.jpg"

    a2 = Assignment2(input_image_path)
    a2.scale_image(-8)
    a2.scale_image(8)
