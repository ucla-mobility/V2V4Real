# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com
import numpy as np

import init_paths
from math_geometry import get_2dline_from_pts
from xinshuo_miscellaneous import CHECK_EQ_NUMPY

def test_get_2dline_from_pts():
	print('check normal line')
	pts1 = [0, 1, 1]
	pts2 = [1, 0, 1]
	line = get_2dline_from_pts(pts1, pts2)
	assert CHECK_EQ_NUMPY(line, np.array([1, 1, -1]).reshape((3, 1)))
	
	print('check vertical line')
	pts1 = [1, 1, 1]
	pts2 = [1, 12, 1]
	line = get_2dline_from_pts(pts1, pts2)
	assert CHECK_EQ_NUMPY(line, np.array([-11, 0, 11]).reshape((3, 1)))

	print('check horizontal line')
	pts1 = [20, 0, 1]
	pts2 = [1, 0, 1]
	line = get_2dline_from_pts(pts1, pts2)
	assert CHECK_EQ_NUMPY(line, np.array([0, -19, 0]).reshape((3, 1)))

	print('check a point at infinity')
	pts1 = [0, 3, 0]
	pts2 = [1, 0, 1]
	line = get_2dline_from_pts(pts1, pts2)
	assert CHECK_EQ_NUMPY(line, np.array([3, 0, -3]).reshape((3, 1)))

	print('check line at infinity')
	pts1 = [0, 3, 0]
	pts2 = [1, 0, 0]
	line = get_2dline_from_pts(pts1, pts2)
	assert CHECK_EQ_NUMPY(line, np.array([0, 0, -3]).reshape((3, 1)))

	print('\n\nDONE! SUCCESSFUL!!\n')

if __name__ == '__main__':
	test_get_2dline_from_pts()