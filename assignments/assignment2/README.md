# Assignment 2: Image Processing Tasks

This folder contains solutions to Assignment 2 for the Image Processing course. The assignment focuses on image scaling by pixel replication and decimation, basic grey level transformations, and affine transformations. Each problem has been solved by writing custom code without using pre-existing libraries for certain image operations.

## Problem 1: Image Scaling by Pixel Replication 

### Task:
Write custom code capable of shrinking and zooming an image using pixel replication and decimation. The program accepts zoom/shrink factors as integer inputs, where a negative input shrinks the image, and a positive input expands it.

### Subtasks:
- **(a) Shrinking**: Shrink an image by a factor of 8 in both width and height, and display the resulting image.
- **(b) Zooming**: Zoom the shrunk image back to its original size and display it. Compare the original and zoomed images, discussing the visual differences. Additionally, perform image subtraction between the original and zoomed image, and display the result.

### Instructions to Run:
1. Provide an input image in `.png` or `.jpg` format.
2. Run the `shrink` and `zoom` methods with the required zoom/shrink factors.
3. The program will display the shrunk, zoomed, and subtracted images.

### Expected Outputs:
- The shrunk image.
- The zoomed image.
- A comparison of the original and zoomed images.
- The result of image subtraction between the original and zoomed images.

---

## Problem 2: Basic Grey Level Transformations 

### Task:
Implement basic grey level transformations on an image.

### Subtasks:
- **(a)** Read and display an image.
- **(b)** Calculate and display the negative of the image.
- **(c)** Perform contrast stretching using at least three different settings. Show the contrast-stretched images and perform image subtraction between the original image and each contrast-stretched image.

### Instructions to Run:
1. Provide an input image in `.png` or `.jpg` format.
2. Call the methods to:
   - Read and display the image.
   - Compute the negative of the image.
   - Perform contrast stretching with at least three settings.
3. The program will display the results and the corresponding image subtractions.

### Expected Outputs:
- The original and negative image.
- At least three contrast-stretched images.
- Subtraction results between the original and each contrast-stretched image.

---

## Problem 3: Image Transformations

### Task:
Research and implement affine transformations, specifically rotation, on an image. Write custom code to rotate an image by a specified angle without using any existing rotation libraries. Additionally, perform image subtraction between the original and rotated images.

### Instructions to Run:
1. Provide an input image in `.png` or `.jpg` format.
2. Call the method to rotate the image by a specified angle (in degrees).
3. The program will display the rotated image and the result of the subtraction between the original and rotated images.

### Expected Outputs:
- The rotated image.
- The subtraction result between the original and rotated image.

---

## How to Run the Code
1. Clone the repository and navigate to the `assignment2` folder.
2. Install any required dependencies (refer to the `requirements.txt` file if applicable).
3. Run the Python script with appropriate method calls for each problem.
4. Images will be displayed as outputs, and results will be saved if specified.

---

## Notes
- The operations performed in this assignment are implemented without using pre-existing libraries for shrinking, zooming, and rotating images.
- Results may vary depending on the image input and the transformation parameters.
