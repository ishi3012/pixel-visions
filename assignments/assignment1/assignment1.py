import cv2
import numpy as np 
import matplotlib.pyplot as plt 
from typing import Tuple

interpolations = {
    'nearest_neighbor':cv2.INTER_NEAREST,
    'bilinear': cv2.INTER_LINEAR,
    'bicubic': cv2.INTER_CUBIC
}

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
        self.image_grayscale = None

        # Image statistics
        self.total_pixels = None
        self.max_pixel_value = None
        self.mean_pixel_value = None
        
        # Statistics for the channel
        self.channel_total_pixels = None
        self.channel_max_pixel_value = None
        self.channel_mean_pixel_value = None

        self.binary_image = None
        self.color_binary_image = None

        # Reduced images dictionary
        self.reduced_images = {}

        # Restored images dictionary
        self.restored_images = {}

        # Subtracted images dictionary
        self.subtracted_images = {}

        # Reduced Gray Levels images dictionary
        self.reduced_gary_levels_images = {}

    def load_image(self, path: str) -> np.ndarray:
        """
        Loads the image from the specified file path. 

        Parameters:
            - path (str) : A string value containing path to the image file. 

        Returns: 
            - numpy.ndarray : Loaded image in RGB format. 
        """
        image = cv2.imread(path)

        if image is None:
            raise ValueError("The image not found at the specified location or could not be loaded.")
        
        return image

    def save_output_image(self, image: np.ndarray, 
    output_file_path: str="assignments/assignment1_Outputs/output_image.png") -> None:
        """
        Displays the image with the title provided by user.

        Parameters:
            - image (numpy.ndarray) : The image to be displayed.
            - title (str)           : The title of the image. Default = TestImage.
        """        
        # Save the output image
        success = cv2.imwrite(output_file_path, image)

        if success:
            print(f"Image saved successfully as {output_file_path}.\n")
        else:
            print(f"Failed to save the image.\n")

    def display_image(image: np.ndarray, title: str = 'Image') -> None:
        """
        Display an image using OpenCV.

        Parameters:
            image (np.ndarray): The image to display.
            title (str): The title of the displayed window.
        """
        
        cv2.imshow(title, image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    def to_grayscale(self, output_file_path: str = "assignments/assignment1_Outputs/Grayscale_Input_image.png") -> None:
        """
        Converts the given image into grayscale image and saves the grayscale image at specified file loaction.

        parameters:
            - output_file_path(str) : The binary file output path. Default is Binary_output_image.png
       
        """
        if self.image is None:
            raise ValueError("The image is not provided.") 

        self.image_grayscale = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.save_output_image(self.image_grayscale, output_file_path=output_file_path)              
    
    def get_image_statistics(self, channel:np.ndarray = None) -> None:
        """
        Calculates metrics for the input image. 

        Parameters:
            - channel (np.ndarray): A 2D array representing the image channel for which the statistics are to be calculated.
        """
        if self.image_grayscale is None:
            self.to_grayscale()

        if channel is None:
            self.total_pixels = cv2.countNonZero(self.image_grayscale)
            self.max_pixel_value = cv2.minMaxLoc(self.image_grayscale)[1]
            self.mean_pixel_value = cv2.mean(self.image_grayscale)[0]
            print(f" Image size (total number of pixels) = {self.total_pixels} \n Maximum Pixel Value = {self.max_pixel_value} \n Mean Pixel value = {self.mean_pixel_value}\n")
        else:
            # Statistics for the channel
            self.channel_total_pixels = cv2.countNonZero(channel)
            self.channel_max_pixel_value = cv2.minMaxLoc(channel)[1]
            self.channel_mean_pixel_value = cv2.mean(channel)[0]
            print(f" Image channel size (total number of pixels) = {self.channel_total_pixels} \n Maximum Pixel Value = {self.channel_max_pixel_value} \n Mean Pixel value = {self.channel_mean_pixel_value}\n")
        
    def threshold_by_mean(self, 
    output_file_path: str = "assignments/assignment1_Outputs/GrayScale_binary_image.png") -> np.ndarray:
        """
        Converts the grayscale image to binary image using mean pixel value as threshold. 
        All pixels’ values less than the average calculated at (d) 
        will be equal to 0 and all the others will be equal to 1.

        parameters:
            - output_file_path(str) : The binary file output path. Default is GrayScale_binary_image.png

        Returns: 
            - numpy.ndarray : Binary image.
        """
        
        if self.image_grayscale is None:
            self.to_grayscale()
        
        if self.mean_pixel_value is None:
            self.get_image_statistics()
       
       # Threshold the image based on mean pixel value. 
        _,self.binary_image = cv2.threshold(self.image_grayscale, self.mean_pixel_value, 1, cv2.THRESH_BINARY)
        self.binary_image = self.binary_image *255

        self.save_output_image(self.binary_image, output_file_path=output_file_path)  
    
    def threshold_by_channel(self, channel: np.ndarray) -> np.ndarray:
        """
        Thresholds teh pixel values of the channel based on its mean pixel value

        Parameters:
            - channel (numpy.ndarray) : A single color channel from the input colored image. 

        Returns:
            - np.ndarray: A binary version of the channel where pixel values greater than or equal 
                            to the mean are set to 255 (white) and those less than the mean are set to 0 (black).
        """

        if self.channel_mean_pixel_value is None:
            self.get_image_statistics(channel = channel)

        _ , binary_channel = cv2.threshold(channel, self.channel_mean_pixel_value, 1, cv2.THRESH_BINARY)
        binary_channel = binary_channel * 255

        return binary_channel
    
    def threshold_color_image(self) -> np.ndarray:
        """
        Applies thresholding to each color channel (Red, Green, Blue) 
        separately and combines the processed images to create a color binary image.

        Returns:
            - np.ndarray - The combined thresholded image where each 
                            channel is thresholded based on  its mean value. 
        """

        blue_channel, green_channel, red_channel = cv2.split(self.image)

        blue_channel_processed = self.threshold_by_channel(blue_channel)
        green_channel_processed = self.threshold_by_channel(green_channel)
        red_channel_processed = self.threshold_by_channel(red_channel)

        self.color_binary_image = cv2.merge((blue_channel_processed, 
                                                green_channel_processed, 
                                                red_channel_processed))

        # Save the colored binary image
        self.save_output_image(self.color_binary_image, 
                                output_file_path = "assignments/assignment1_Outputs/Combined_color_binary_image.png")
        self.save_output_image(blue_channel_processed, 
                                output_file_path = "assignments/assignment1_Outputs/Blue_channel_binary_image.png")
        self.save_output_image(green_channel_processed, 
                                output_file_path = "assignments/assignment1_Outputs/Green_channel_binary_image.png")
        self.save_output_image(red_channel_processed, 
                                output_file_path = "assignments/assignment1_Outputs/Red_channel_binary_image.png")

    def reduce_and_restore_image(self, scale:float = 0.1) -> None:
        """
        Reduces the spatial resolution of the input image based on the scale and restores the image to 
        the original resolution. The method uses Nearest Neighbor, Bilinear and Bicubic interpolation method. 

        Parameters: 
            - scale (float) : The factor by which the image's spatial resolution should be adjusted.
        """
        scaled_height = int(self.image.shape[0] * scale)
        scaled_width = int(self.image.shape[1] * scale)

        for key, interpolation in interpolations.items():
            self.reduced_images[key] = cv2.resize(self.image, 
                                                    (scaled_width,scaled_height), 
                                                    interpolation = interpolation)

            self.restored_images[key] = cv2.resize(self.reduced_images[key], 
                                                    (self.image.shape[1], self.image.shape[0]), 
                                                    interpolation = interpolation)

            self.subtracted_images[key] = cv2.subtract(self.image, 
                                                        self.restored_images[key])
            # Save both reduced and restored images.
            self.save_output_image(self.reduced_images[key], 
                                    output_file_path = f"assignments/assignment1_Outputs/Reduced_{key}_image.png")
            self.save_output_image(self.restored_images[key], 
                                    output_file_path = f"assignments/assignment1_Outputs/Restored_{key}_image.png")
            self.save_output_image(self.subtracted_images[key], 
                                    output_file_path = f"assignments/assignment1_Outputs/Subtracted_{key}_image.png")
    
    def reduce_gray_levels(self, scale: float = 0.1) -> np.ndarray:
        """
        Reduces number of gray levels in a grayscale.

        Parameters:
            - scale (float): Desired number of gray levels (can be a non-integer). 
                             Default value = 0.1

        Returns:
            - (numpy.ndarray): Image with reduced gray levels.

        """ 
        max_intensity = 255 
        scale = max_intensity / (scale - 0.5)

        if self.image_grayscale is None:
            self.to_grayscale()

        # Reduce the gray level
        reduced_image = np.floor(self.image_grayscale / scale).astype(np.uint8) * scale 

        return reduced_image       

    def save_gray_levels(self) -> None:
        """
        Generates and saves the reduced gray level images.
       
        """        
        for i in range(7, 0, -1):
            scale = 2 ** i  
            self.reduced_gary_levels_images[i] = self.reduce_gray_levels(scale)
            self.save_output_image(self.reduced_gary_levels_images[i], 
                                    output_file_path = f"assignments/assignment1_Outputs/Gray_Levels_Reduced_{i}_image.png")


if __name__ == "__main__":
    
    a1 = Assignment1("assignments/Image.png")
    # 1(a) Read an image, convert it to grayscale if it isn’t already, and display the converted image
    a1.to_grayscale()

    # 1(b)(c)(d)Calculate and report the size (total number of pixels) , the maximum pixel value , mean pixel value
    a1.get_image_statistics()

    # 1(e) Change the grayscale image to binary using mean_pixel_value as threshold
    a1.threshold_by_mean()

    # 1(e)(a) Thresholding on a color image
    a1.threshold_color_image()

    # 2. Image Interpolation 
    a1.reduce_and_restore_image()
    
    # # Reducing the Number of Gray Levels in an Image from 256 to 2
    a1.save_gray_levels()

    # Reduce the Number of Gray Levels using non-integer scale. e.g. 5.7
    factor = 2.1
    scale = 2 ** factor
    reduced_image = a1.reduce_gray_levels(scale)
    a1.save_output_image(reduced_image, output_file_path = f"assignments/assignment1_Outputs/Gray_Levels_Reduced_{factor}_image.png")
