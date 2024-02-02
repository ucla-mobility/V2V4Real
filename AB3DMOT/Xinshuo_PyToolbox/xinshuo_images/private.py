# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com

# this file includes private functions for internal use only
import numpy as np
from xinshuo_miscellaneous import ispilimage, isnpimage, isnparray, isimage_dimension, isnannparray

def safe_image(input_image, warning=True, debug=True):
	'''
	return a numpy image no matter what format the input is
	make sure the output numpy image is a copy of the input image

	parameters:
		input_image:		pil or numpy image, color or gray, float or uint

	outputs:
		np_image:			numpy image, with the same color and datatype as the input
		isnan:				return True if any nan value exists
	'''
	if ispilimage(input_image): np_image = np.array(input_image)
	elif isnpimage(input_image): np_image = input_image.copy()
	else: assert False, 'only pil and numpy images are supported, might be the case the image is float but has range of [0, 255], or might because the data is float64'

	isnan = isnannparray(np_image)
	if warning and isnan: print('nan exists in the image data')

	return np_image, isnan

def safe_batch_image(input_image, warning=True, debug=True):
	'''
	return a numpy image no matter what format the input is
	make sure the output numpy image is a copy of the input image

	parameters:
		input_image:		a numpy image, NHWC, float or uint

	outputs:
		np_image:			NHWC numpy image, with the same datatype as the input
		isnan:				return True if any nan value exists
	'''
	if debug: assert isnparray(input_image), 'the input image should be a numpy array'
	np_image = input_image.copy()

	if np_image.ndim == 2: np_image = np.expand_dims(np_image, axis=0)			# compatible with grayscale image
	if np_image.ndim == 3:		# expand HWC to NHWC batch images with batch of 1
		if debug: assert isnpimage(np_image), 'the image should be a numpy image'
		np_image = np.expand_dims(np_image, axis=0)
	elif np_image.ndim == 4:
		if debug: 
			assert np_image.shape[-1] == 3 or np_image.shape[-1] == 1, 'the image shape is not good'
			for image_index in range(np_image.shape[0]):
				assert isnpimage(np_image[image_index]), 'each individual image should be a numpy image'
	else: assert False, 'only color images are supported'

	isnan = isnannparray(np_image)
	if warning and isnan: print('nan exists in the image data')

	return np_image, isnan

def safe_image_like(input_image, warning=True, debug=True):
	'''
	return an image-like numpy no matter what format the input is, the numpy has the image shape, but value may not be in [0, 1] for float image
	make sure the output numpy image is a copy of the input image

	note:
		an image-like numpy array is an array with image-like shape, but might contain arbitrary value

	parameters:
		input_image:		pil image or image-like array, color or gray, float or uint

	outputs:
		np_image:			numpy image, with the same color and datatype as the input
		isnan:				return True if any nan value exists
	'''
	if ispilimage(input_image): np_image = np.array(input_image)
	elif isnparray(input_image):
		np_image = input_image.copy()
		assert isimage_dimension(np_image), 'the input is not an image-like numpy array'
	else: assert False, 'only pil and numpy image-like data are supported'

	isnan = isnannparray(np_image)
	if warning and isnan: print('nan exists in the image data')

	return np_image, isnan

def safe_batch_deep_image(input_image, warning=True, debug=True):
	'''
	return a batch image-like deep numpy no matter what format the input is,
	the shape of input should be N3HW or 3HW,
	make sure the output numpy image is a copy of the input image

	note:
		an image-like numpy array is an array with image-like shape, but might contain arbitrary value

	parameters:
		input_image:		image-like numpy array, N3HW or 3HW, float or uint

	outputs:
		np_image:			N3HW numpy image, with the same datatype as the input
		isnan:				return True if any nan value exists
	'''
	if debug: assert isnparray(input_image), 'the input image should be a numpy array'
	np_image = input_image.copy()

	# if np_image.ndim == 2:		# expand HW gradscale image to CHW image with one channel
		# np_image = np.expand_dims(np_image, axis=0)
	if np_image.ndim == 3:		# expand CHW to NCHW batch images with batch of 1
		if debug: assert np_image.shape[0] == 3, 'the image should be a color image'
		np_image = np.expand_dims(np_image, axis=0)
	elif np_image.ndim == 4:
		if debug: assert np_image.shape[1] == 3, 'the image should be a color image'
	else: assert False, 'only color images are supported'

	isnan = isnannparray(np_image)
	if warning and isnan: print('nan exists in the image data')

	return np_image, isnan