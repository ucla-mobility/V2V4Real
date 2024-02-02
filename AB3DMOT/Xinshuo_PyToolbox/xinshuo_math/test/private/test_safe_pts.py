# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com
import numpy as np, copy, sys

import init_paths
from private import safe_pts
from xinshuo_miscellaneous import CHECK_EQ_NUMPY, CHECK_EQ_LIST_ORDERED

def test_safe_pts():
	print('test single list')
	pts = [1, 4]
	good_pts = safe_pts(pts)
	print(good_pts)
	print(good_pts.shape)
	assert CHECK_EQ_NUMPY(good_pts, np.array(pts).reshape((2, 1)))
	assert CHECK_EQ_LIST_ORDERED(pts, [1, 4])

	print('test list of list of 2 elements')
	pts = [[1, 2], [5, 8]]
	good_pts = safe_pts(pts)
	print(good_pts)
	print(good_pts.shape)
	assert CHECK_EQ_NUMPY(good_pts, np.array(pts).reshape((2, 2)).transpose())
	assert CHECK_EQ_LIST_ORDERED(pts, [[1, 2], [5, 8]])

	print('test (2, ) numpy array')
	pts = np.array([1, 4])
	good_pts = safe_pts(pts)
	print(pts.shape)
	print(good_pts.shape)
	assert CHECK_EQ_NUMPY(good_pts, np.array(pts).reshape((2, 1)))
	assert CHECK_EQ_NUMPY(pts, np.array([1, 4]))

	print('test (2, N) numpy array')
	pts = np.random.rand(10, 2).transpose()
	pts_ori = copy.copy(pts)
	good_pts = safe_pts(pts)
	print(pts.shape)
	print(good_pts.shape)
	assert CHECK_EQ_NUMPY(good_pts, pts)
	assert CHECK_EQ_NUMPY(pts, pts_ori)

	######################################## test failure cases
	print('test list of 3 elements')
	pts = [1, 2, 4]
	try:
		good_pts = safe_pts(pts)
		sys.exit('\nwrong! never should be here\n\n')
	except AssertionError:
		print('the length of list should be 2')

	print('test list of list of 3 elements')
	pts = [[1, 2, 4], [5, 7, 8]]
	try:
		good_pts = safe_pts(pts)
		sys.exit('\nwrong! never should be here\n\n')
	except AssertionError:
		print('the length of list should be 2')

	print('test numpy array with 3 elements')
	pts = np.array([1, 2, 4]).reshape(3, )
	try:
		good_pts = safe_pts(pts)
		sys.exit('\nwrong! never should be here\n\n')
	except AssertionError:
		print('the numpy array should be rows of 2')

	print('test numpy array with rows of 3')
	pts = np.array([[1, 2], [5, 7], [4, 6]])
	try:
		good_pts = safe_pts(pts)
		sys.exit('\nwrong! never should be here\n\n')
	except AssertionError:
		print('the numpy array should be rows of 2')

	print('\n\nDONE! SUCCESSFUL!!\n')
	
if __name__ == '__main__':
	test_safe_pts()
