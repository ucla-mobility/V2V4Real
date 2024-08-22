# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com

# this file includes functions about images I/O
import numpy as np
from PIL import Image

from xinshuo_miscellaneous.private import safe_path
from xinshuo_images.private import safe_image

from .file_io import mkdir_if_missing
from xinshuo_miscellaneous import is_path_exists_or_creatable, isimage, isscalar, is_path_exists
from xinshuo_images import image_rotate, image_resize, rgb2gray

def load_image(src_path, resize_factor=None, target_size=None, input_angle=0, gray=False, warning=True, debug=True):
    '''
    load an image from given path, with preprocessing of resizing and rotating, output a rgb image

    parameters:
        resize_factor:      a scalar
        target_size:        a list or tuple or numpy array with 2 elements, representing height and width
        input_angle:        a scalar, counterclockwise rotation in degree

    output:
        np_image:           an uint8 rgb numpy image
    '''
    src_path = safe_path(src_path, warning=warning, debug=debug)
    if debug: assert is_path_exists(src_path), 'image path is not correct at %s' % src_path
    if resize_factor is None and target_size is None: resize_factor = 1.0           # default not to have resizing

    with open(src_path, 'rb') as f:
        with Image.open(f) as img:
            if gray: img = img.convert('L')
            else: 
                try:
                    img = img.convert('RGB')
                except IOError:
                    print(src_path)
                    zxc
            np_image = image_rotate(img, input_angle=input_angle, warning=warning, debug=debug)
            np_image = image_resize(np_image, resize_factor=resize_factor, target_size=target_size, warning=warning, debug=debug)
    return np_image

def save_image(input_image, save_path, resize_factor=None, target_size=None, input_angle=0, warning=True, debug=True):
    '''
    load an image to a given path, with preprocessing of resizing and rotating

    parameters:
        resize_factor:      a scalar
        target_size:        a list of tuple or numpy array with 2 elements, representing height and width
        input_angle:        a scalar, counterclockwise rotation in degree
    '''
    save_path = safe_path(save_path, warning=warning, debug=debug); mkdir_if_missing(save_path)
    if debug: is_path_exists_or_creatable(save_path), 'the path is not good to save'
    np_image, _ = safe_image(input_image, warning=warning, debug=debug)
    if resize_factor is None and target_size is None: resize_factor = 1.0           # default not to have resizing

    # preprocessing the image before saving
    np_image = image_rotate(np_image, input_angle=input_angle, warning=warning, debug=debug)
    np_image = image_resize(np_image, resize_factor=resize_factor, target_size=target_size, warning=warning, debug=debug)
    
    # saving
    pil_image = Image.fromarray(np_image)
    pil_image.save(save_path)