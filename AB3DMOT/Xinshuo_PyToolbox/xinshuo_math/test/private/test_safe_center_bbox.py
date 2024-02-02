# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com
import numpy as np, sys

import init_paths
from private import safe_center_bbox
from xinshuo_miscellaneous import CHECK_EQ_NUMPY

def test_safe_center_bbox():
	######################################## test with 4 elements
	print('test single list')
	bbox = [1, 2, 3, 4]
	good_bbox = safe_center_bbox(bbox)
	print(good_bbox)
	print(good_bbox.shape)
	assert CHECK_EQ_NUMPY(good_bbox, np.array(bbox).reshape((1, 4)))

	print('test list of list of 4 elements')
	bbox = [[1, 2, 3, 4], [5, 6, 7, 8]]
	good_bbox = safe_center_bbox(bbox)
	print(good_bbox)
	print(good_bbox.shape)
	assert CHECK_EQ_NUMPY(good_bbox, np.array(bbox).reshape((2, 4)))

	print('test (4, ) numpy array')
	bbox = np.array([1, 2, 3, 4])
	good_bbox = safe_center_bbox(bbox)
	print(bbox.shape)
	print(good_bbox.shape)
	assert CHECK_EQ_NUMPY(good_bbox, np.array(bbox).reshape((1, 4)))

	print('test (N, 4) numpy array')
	bbox = np.random.rand(10, 4)
	good_bbox = safe_center_bbox(bbox)
	print(bbox.shape)
	print(good_bbox.shape)
	assert CHECK_EQ_NUMPY(good_bbox, bbox)

	######################################## test with 2 elements
	print('test single list')
	bbox = [1, 4]
	good_bbox = safe_center_bbox(bbox)
	print(good_bbox)
	print(good_bbox.shape)
	assert CHECK_EQ_NUMPY(good_bbox, np.array(bbox).reshape((1, 2)))

	print('test list of list of 2 elements')
	bbox = [[1, 2], [5, 8]]
	good_bbox = safe_center_bbox(bbox)
	print(good_bbox)
	print(good_bbox.shape)
	assert CHECK_EQ_NUMPY(good_bbox, np.array(bbox).reshape((2, 2)))

	print('test (2, ) numpy array')
	bbox = np.array([1, 2])
	good_bbox = safe_center_bbox(bbox)
	print(bbox.shape)
	print(good_bbox.shape)
	assert CHECK_EQ_NUMPY(good_bbox, np.array(bbox).reshape((1, 2)))

	print('test (N, 2) numpy array')
	bbox = np.random.rand(10, 2)
	good_bbox = safe_center_bbox(bbox)
	print(bbox.shape)
	print(good_bbox.shape)
	assert CHECK_EQ_NUMPY(good_bbox, bbox)

	######################################## test failure cases
	print('test list of list of 3 elements')
	bbox = [[1, 2, 4], [5, 7, 8]]
	try:
		good_bbox = safe_center_bbox(bbox)
		sys.exit('\nwrong! never should be here\n\n')
	except AssertionError:
		print('the length of list should be 4 or 2')

	print('test list of 3 elements')
	bbox = [1, 2, 4]
	try:
		good_bbox = safe_center_bbox(bbox)
		sys.exit('\nwrong! never should be here\n\n')
	except AssertionError:
		print('the length of list should be 4 or 2')

	print('test numpy array with columns of 3')
	bbox = np.array([[1, 2, 4], [5, 7, 8]])
	try:
		good_bbox = safe_center_bbox(bbox)
		sys.exit('\nwrong! never should be here\n\n')
	except AssertionError:
		print('the numpy array should be columns of 4 or 2')

	print('test numpy array with 3 elements')
	bbox = np.array([1, 2, 4]).reshape(3, )
	try:
		good_bbox = safe_center_bbox(bbox)
		sys.exit('\nwrong! never should be here\n\n')
	except AssertionError:
		print('the numpy array should be columns of 4 or 2')

	print('\n\nDONE! SUCCESSFUL!!\n')
	
if __name__ == '__main__':
	test_safe_center_bbox()
