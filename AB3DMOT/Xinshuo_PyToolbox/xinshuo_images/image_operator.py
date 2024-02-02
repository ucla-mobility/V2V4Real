# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com

# this file includes classes of linear filters ready applied on the images
import numpy as np
# from scipy import ndimage

from .private import safe_image
from xinshuo_miscellaneous import isimsize, iscolorimage_dimension, isgrayimage_dimension, isuintimage, isfloatimage
from xinshuo_math import data_normalize

class linear_filter(object):
	def __init__(self, filter_size=None, warning=True, debug=True):
		'''
		generate a class of filter
		'''
		if debug: isimsize(filter_size), 'the filter size is not correct'
		self.debug = debug
		self.warning = warning
		self.weights = None

	def __expand_3d(self):
		'''
		broadcast the current 2D filter to 3D

		outputs:
			weights:	float32 numpy array with shape of filter_size x filter_size x 1
		'''
		if self.debug: 
			assert self.weights is not None, 'please set up a kernel/filter first'
			assert len(self.weights.shape) == 2, 'the current kernel is not a 2D kernel'

		self.weights = np.expand_dims(self.weights, axis=2)
		return self.weights

	def convolve(self, input_image):
		'''
		convolve the kernel with the input image, whatever the input image format is. If the input image 
		is a color image, the filter is expanded to a 3D shape
	
		parameters:
			input_image:		an pil or numpy, gray or color image

		outputs:
			filtered_image:		a float32 numpy image, shape is same as before
		'''
		np_image, _ = safe_image(input_image, warning=self.warning, debug=self.debug)
		if isuintimage(np_image): np_image = np_image.astype('float32') / 255.

		if self.debug: 
			assert isfloatimage(np_image), 'the input image should be a float image'
			self.weights is not None, 'the kernel is not defined yet'

		if iscolorimage_dimension(np_image): self.weights = self.__expand_3d()			# expand the filter to 3D for color image
		elif isgrayimage_dimension(np_image): np_image = np_image.reshape(np_image.shape[0], np_image.shape[1])		# squeeze the image dimension to 2
		else: assert False, 'the dimension of the image is not correct'

		filtered_image = ndimage.filters.convolve(np_image, self.weights)
		return filtered_image

	def gaussian(self):
		'''
		generate Gaussian filter 5 x 5

		outputs:
			weights:	float32 numpy array with gaussian weights
		'''
		gaussian = np.array([[1,  4,  6,  4, 1],
							 [4, 16, 24, 16, 4],
							 [6, 24, 36, 24, 6],
							 [4, 16, 24, 16, 4],
							 [1,  4,  6,  4, 1]], dtype='float32')
		
		self.weights = 1. / 256 * gaussian
		return self.weights

	def laplacian(self):
		'''
		generate Laplacian of Gaussian filter 3 x 3, edge detection

		outputs:
			weights:	float32 numpy array with laplacian weights
		'''
		laplacian = np.array([[-1, -1, -1],
							  [-1,  8, -1],
							  [-1, -1, -1]], dtype='float32')
		
		self.weights = laplacian
		return self.weights

	def sobel(self, axis='x'):
		'''
		generate sobel filter along x or y axis

		parameters:
			axis:		x or y, define the orientation of sobel filter

		outputs:
			weights:	float32 numpy array with sobel weights
		'''
		if self.debug: assert axis in ['x', 'y'], 'the axis for sobel filter is not correct'

		if axis == 'x': sobel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype='float32')
		elif axis == 'y': sobel = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype='float32')
		
		self.weights = 1. / 8 * sobel
		return self.weights.reshape((3, 3))

	def unsharp_mask(self):
		'''
		generate the filter of unsharp mask, the filter is composition of minus gaussian blur 
		plus an original image

		outputs:
			weights:	float32 numpy array with unsharp masking weights
		'''
		unsharp_mask = np.array([[1,  4,   6,   4, 1],
							 	 [4, 16,  24,  16, 4],
							 	 [6, 24, -476, 24, 6],
							 	 [4, 16,  24,  16, 4],
							 	 [1,  4,   6,   4, 1]], dtype='float32')
		
		self.weights = (-1.0) / 256 * unsharp_mask
		return self.weights