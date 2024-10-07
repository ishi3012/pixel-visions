# Assignment 3 - Image Processing Techniques

This assignment focuses on three key image processing techniques: **Histogram Equalization**, **Filtering**, and **Bit Plane Splicing**. The goal is to enhance images and understand the transformations applied using these techniques.

## 1. Histogram Equalization 

### Task Overview:
- **Objective:** Enhance the contrast of an image by equalizing its histogram and comparing the original and enhanced versions.
  
### Steps:
1. **Read and Display an Image:**
   - Select an image with varying intensities.
   - Display the original image.

2. **Calculate and Display Histogram:**
   - Compute the histogram of the image to observe its intensity distribution.
   
3. **Apply Histogram Equalization:**
   - Enhance the image contrast using histogram equalization.
   - Display both the uniform histogram and the enhanced image.
   
4. **Explain the Difference:**
   - Compare the histograms of the original and enhanced images and explain why they differ.

5. **Calculate Difference Between Images:**
   - Compute and display the difference between the original and enhanced image.

#### 481 Students: 
- Apply local enhancement to the image using a chosen window size.
- Experiment with different window sizes and display the results.

## 2. Filtering 

### Task Overview:
- **Objective:** Perform various filtering techniques on an image, including edge detection and blurring.

### Steps:
1. **481 Students :**
   - Implement a custom filtering function that processes the image pixel by pixel.
   - State assumptions for handling image borders.

2. **Perform Filtering Using Built-in Filters:**
   - Apply the following filters:
     - **Prewitt Filter**
     - **Sobel Filter**
     - **Point Filter**
     - **Blurring Filter**
   - Display the filtered images.

3. **481 Students:** 
   - Compare the results of your custom function with the results of library functions.
   - Calculate and display the difference between both results for each filter.

## 3. Bit Plane Splicing 

### Task Overview:
- **Objective:** Perform bit-plane splicing to analyze the intensity distribution of an image.

### Steps:
1. **Bit Plane Extraction:**
   - Perform bit-plane splicing on the image.
   - Generate and display the 8 bit planes of the image.

2. **Image Reconstruction:**
   - Reconstruct the image by successively adding bit planes, starting from the most significant bit.
   - Create 7 new images from the combination of bit planes (e.g., combining bit planes 7 and 6, 7 and 6 and 5, etc.).
   - Display the reconstructed images.

3. **Conclusion:**
   - Determine which bit plane combination provides a good visual approximation of the original image.

---

### Additional Notes:
- Make sure to document all the steps in your code and provide clear visual outputs for each stage.
- For students in the 481 section, extra tasks such as local enhancement and implementing a custom filtering function are required.

### References:
- [Bit Plane Splicing - Wikipedia](https://en.wikipedia.org/wiki/Bit_plane)


