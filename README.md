# Pixel-Visions

**Pixel-Visions** is a Python package designed for image processing, analysis, and vision tasks. This repository contains implementations of various image manipulation techniques, as well as cool projects that showcase advanced capabilities in computer vision.

## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Assignments](#assignments)
- [Projects](#projects)
- [Contributing](#contributing)
- [License](#license)

## Features
- Read and manipulate images in various formats
- Convert images to grayscale
- Perform image analysis (e.g., calculate pixel metrics)
- Implement image interpolation techniques (nearest neighbor, bilinear, bicubic)
- Reduce the number of gray levels in an image
- Various projects demonstrating advanced image processing techniques

## Installation

To install the required dependencies, create a virtual environment and run:
```bash
pip install -r requirements.txt
```

## Usage

To use the package, import it in your Python scripts or Jupyter notebooks. Below is a basic example of how to use the Assignment1 class for image manipulation:

```bash
from pixel_visions.assignment1 import Assignment1

# Create an instance of Assignment1 with the path to your image
assignment = Assignment1('path_to_image.jpg')

# Process the image
assignment.process_image()
```
## Assignments

This repository includes various assignments demonstrating fundamental image processing techniques, organized in the [assignments](assignments) directory:

- **Image Manipulation**
  - Convert images to grayscale, display, and analyze pixel metrics.
  - Thresholding techniques for binary image conversion.

- **Image Interpolation**
  - Reduce spatial resolution and restore images using different interpolation methods (nearest neighbor, bilinear, bicubic).
  - Visual comparison of restored images.

- **Gray Level Reduction**
  - Reduce the number of gray levels in an image from 256 to 2, demonstrating pixel-level manipulations.

Feel free to explore the [assignments](assignments) folder for specific implementations and examples.

## Projects

The projects/ directory contains advanced projects that utilize the core functionalities of the Pixel-Visions package, including but not limited to:

- Image Super-Resolution
- Medical Image Segmentation
- Real-Time Object Detection

## Contributing
Contributions are welcome! Please feel free to submit a pull request or open an issue for any enhancements or bug fixes.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

