# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com

# this file includes functions of basic probability and statistics
import math, cv2, numpy as np

from .private import safe_npdata
from xinshuo_miscellaneous import isnparray, isrange, isscalar

def hist_equalization(input_data, num_bins=256, warning=True, debug=True):
	'''
	convert a N-d numpy data (or list) with random distribution to a 1-d data with equalized histogram
	e.g., for the samples from a gaussian distribution, the data points are dense in the middle, the cdf increases fast
	in the middle so that the discrete cdf is sparse in the middle, the equalized data points are interpolated from cdf such
	that the density can be the same for the middle and the rest

	parameters:
		input_data:		a list or a numpy data, could be any shape, not necessarily a 1-d data, can be integer data (uint8 image) or float data (float32 image)
		num_bins:		bigger, the histogram of equalized data points is more flat

	outputs:
		data_equalized:	equalized data with the same shape as input, it is float with [0, 1]
	'''
	np_data = safe_npdata(input_data, warning=warning, debug=debug)
	if debug: assert isnparray(np_data), 'the input data is not a numpy data'

	ori_shape = np_data.shape
	np_data = np_data.flatten()
	hist, xs = np.histogram(np_data, num_bins, density=True)	# return distribution and X's coordinates
	cdf = hist.cumsum()
	cdf = cdf / cdf[-1]			# sparse in the middle
	data_equalized = np.interp(np_data, xs[:-1], cdf)

	return data_equalized.reshape((ori_shape))

def data_normalize(input_data, method='max', data_range=None, sum=1, warning=True, debug=True):
	'''
	this function normalizes N-d data in different ways: 1) normalize the data from a range to [0, 1]; 2) normalize the data which sums to a value

	parameters:
		input_data:			a list or a numpy N-d data to normalize
		method:				max:	normalize the data from a range to [0, 1], when the range is not given, the max and min are obtained from the data
							sum:	normalize the data such that all elements are summed to a value, the default value is 1
		data_range:			None or 2-element tuple, list or array
		sum:				a scalar

	outputs:
		normalized_data:	a float32 numpy array with same shape as the input data
	'''
	np_data = safe_npdata(input_data, warning=warning, debug=debug).astype('float32')
	if debug: 
		assert isnparray(np_data), 'the input data is not a numpy data'
		assert method in ['max', 'sum'], 'the method for normalization is not correct'

	if method == 'max':
		if data_range is None: max_value, min_value = np.max(np_data), np.min(np_data)
		else:	
			if debug: assert isrange(data_range), 'data range is not correct'
			max_value, min_value = data_range[1], data_range[0]
	elif method == 'sum':
		if debug: assert isscalar(sum), 'the sum is not correct'
		max_value, min_value = np.sum(np_data) / sum, 0

	normalized_data = (np_data - min_value) / (max_value - min_value)	# normalization

	return normalized_data

def data_unnormalize(data, data_range, debug=True):
	'''
	this function unnormalizes 1-d label to normal scale based on range of data
	'''
	np_data = safe_npdata(input_data, warning=warning, debug=debug).astype('float32')
	if debug: 
		assert isnparray(np_data), 'the input data is not a numpy data'
		assert isrange(data_range), 'data range is not correct'

	max_value = data_range[1]
	min_value = data_range[0]
	unnormalized = np_data * (max_value - min_value) + min_value

	# if debug:
		# normalized = normalize_data(data=unnormalized, data_range=data_range, debug=False)
		# assert_almost_equal(data, normalized, decimal=6, err_msg='data is not correct: %f vs %f' % (data, normalized))
	return unnormalized

def identity(data, data_range=None, debug=True):
    if debug:
        print('debug mode is on during identity function. Please turn off after debuging')
        assert isnparray(data), 'data is not correct'
    return data