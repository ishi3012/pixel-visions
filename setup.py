from setuptools import setup, find_packages

setup(
    name='pixel-visions',
    version='0.1.0',
    author='Shilpa Musale',
    author_email='ishishiv3012@gmail.com',
    description='A Python package for image processing, analysis, and vision tasks.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/your_username/pixel-visions',  # Replace with your repo URL
    packages=find_packages(),
    install_requires=[
        'numpy>=1.21.0',
        'opencv-python>=4.5.0',
        'scikit-image>=0.18.0',
        'matplotlib>=3.4.0',
        'Pillow>=8.0.0'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)

