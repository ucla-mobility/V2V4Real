# Xinshuo's Python Toolbox
A Python toolbox that contains common help functions for **stream I/O**, **mathematics**, **image & video processing** and **visualization**. All my projects depend on this toolbox.

### Usage:

*1. Clone the github repository.*
~~~shell
git clone https://github.com/xinshuoweng/Xinshuo_PyToolbox
~~~

*2. Install dependency for the toolbox.*
~~~shell
cd Xinshuo_PyToolbox
pip install -r requirements.txt
~~~

*3. Add the path to the code to your PYTHONPATH.*
~~~shell
export PYTHONPATH=${PYTHONPATH}:/home/user/workspace/code/Xinshuo_PyToolbox
~~~

### Features
- **I/O Stream**

  We provide extensive I/O functions such as loading and saving images with additional preprocessing (resizing, rotating, etc) options, files containing key-points, data array, lists and matrices. Also, functions for uploading and downloading data from google apps are available.

- **Math for Computer Vision**

  Math functions that are commonly used in computer vision are available: such as transforming (converting format, cliping, crop, rotate, enlarge) bounding box and IoU computation, transforming mask, processing key-points and heatmap. Also, some help fuctions for geometry-based vision is supported such as homogeneous representation for points, line, plane, sphere and transformation between them.

- **Image Processing**

  Our tool supports image processing functions such as color space conversion, histogram computation, normalization, cropping, padding, resizing, rotating, concatenation and also common filters. 

- **Video Processing**

  Conversion between images and video is supported in video processing tool.

- **Visualization**

  A diverse set of compositional visualization functions are supported such as visualizaing key-points, bounding boxes, lines and masks on top of images. Also, we provide some statistical visualization functions including cumulative error distribution curve, nearest neighbor, distribution, bar graph and so on.

- **Miscellaneous**

  In addition to processing functions for specific data type, functions for common data type such as list, tuple, dictionary are in miscellaneous toolbox. Also, we provide useful logging, counter and sanity check functions.
