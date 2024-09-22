import cv2
import numpy as np 
import matplotlib.pyplot as plt 
from typing import Tuple

class Assignment1:
    """
    A class for performing the basic image processing tasks on a given image.

    Attributes:
        - image (numpy.ndarray) : An array representing the loaded image. The operations will be performed on this image.
    """
    def __init__(self, image_file_path) -> None:
        """
        Initialized Assignment1 class by loading an image from the specified path.

        Parameters: 
            - image_file_path (str) : The path of the image file. 

        """
        self.image = self.load_image(image_file_path)

    def load_image(self, path: str) -> np.ndarray:
        """
        Loads the image from the specified file path. 

        Parameter:
            - path (str) : A string value containing path to the image file. 

        Returns: 
            - numpy.ndarray : Loaded image in RGB format. 
        """
        image = cv2.imread(path)

        if image is None:
            raise ValueError("The image not found at the specified location or could not be loaded.")
        
        return image

    def save_output_image(self, image: np.ndarray, title="TestImage", output_file_path: str="assignments/assignment1_Outputs/output_image.png") -> None:
        """
        Displays the image with the title provided by user.

        Parameter:
            - image (numpy.ndarray) : The image to be displayed.
            - title (str)           : The title of the image. Default = TestImage.
        """        
        # Save the output image
        success = cv2.imwrite(output_file_path, image)

        if success:
            print(f"Image saved successfully as {output_file_path}.\n")
        else:
            print(f"Failed to save the image.\n")
    
    def to_grayscale(self, output_file_path: str = "assignments/assignment1_Outputs/Grayscale_output_image.png") -> None:
        """
        Converts the given image into grayscale image and saves the grayscale image at specified file loaction.

        parameter:
            - output_file_path(str) : The binary file output path. Default is Binary_output_image.png
       
        """
        if self.image is None:
            raise ValueError("The image is not provided.") 

        self.image_grayscale = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.save_output_image(self.image_grayscale, title="GrayScaleImage", output_file_path=output_file_path)              
    
    def get_image_statistics(self) -> Tuple[int, int, float]:
        """
        Calculates metrics for the input image. 

        Returns:
            - Tuple[int, int, float]: Total number of pixels, maximum pixel value and the mean pixel value
        """
        if self.image_grayscale is None:
            self.to_grayscale()

        self.total_pixels = cv2.countNonZero(self.image_grayscale)
        self.max_pixel_value = cv2.minMaxLoc(self.image_grayscale)[1]
        self.mean_pixel_value = cv2.mean(self.image_grayscale)[0]
        
        print(f" Image size (total number of pixels) = {self.total_pixels} \n Maximum Pixel Value = {self.max_pixel_value} \n Mean Pixel value = {self.mean_pixel_value}\n")
          
    
    def image_thresholding_by_mean(self, output_file_path: str = "assignments/assignment1_Outputs/Binary_output_image.png") -> np.ndarray:
        """
        Converts the grayscale image to binary image using mean pixel value as threshold. 
        All pixels’ values less than the average calculated at (d) will be equal to 0 and all the others will be equal to 1.

        parameter:
            - output_file_path(str) : The binary file output path. Default is Binary_output_image.png

        Returns: 
            - numpy.ndarray : Binary image.
        """
        
        if self.image_grayscale is None:
            self.to_grayscale()
        
        if self.mean_pixel_value is None:
            self.get_image_statistics()
        
        self.binary_image = (self.image_grayscale >= self.mean_pixel_value).astype(np.uint8)*255
        
        self.save_output_image(self.binary_image, output_file_path=output_file_path)   





if __name__ == "__main__":
    
    a1 = Assignment1("assignments/Image.png")
    # 1(a) Read an image, convert it to grayscale if it isn’t already, and display the converted image
    a1.to_grayscale()

    # 1(b)(c)(d)Calculate and report the size (total number of pixels) , the maximum pixel value , mean pixel value
    a1.get_image_statistics()

    # # 1(e) Change the grayscale image to binary using mean_pixel_value as threshold
    a1.image_thresholding_by_mean()
    
    

    
