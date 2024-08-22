# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com
import numpy as np

import init_paths
from math_geometry import get_2dline_from_pts_slope
from numpy.testing import assert_almost_equal

def test_get_2dline_from_pts_slope():
	print('check normal line')
	pts1 = [0, 1, 1]
	slope = 45
	line = get_2dline_from_pts_slope(pts1, slope)
	assert_almost_equal(line, np.array([-0.70710678, 0.70710678, -0.70710678]).reshape((3, 1)))
	
	print('check vertical line')
	pts1 = [0, 1, 1]
	slope = -90
	line = get_2dline_from_pts_slope(pts1, slope)
	assert_almost_equal(line, np.array([1, 0, 0]).reshape((3, 1)))

	print('check vertical line')
	pts1 = [0, 1, 1]
	slope = 90
	line = get_2dline_from_pts_slope(pts1, slope)
	assert_almost_equal(line, np.array([-1, 0, 0]).reshape((3, 1)))

	print('check horizontal line')
	pts1 = [0, 1, 1]
	slope = 0
	line = get_2dline_from_pts_slope(pts1, slope)
	assert_almost_equal(line, np.array([0, 1, -1]).reshape((3, 1)))
	
	print('check a point at infinity')
	pts1 = [0, 2, 0]
	slope = 45
	line = get_2dline_from_pts_slope(pts1, slope)
	assert_almost_equal(line, np.array([0, 0, -1.41421356]).reshape((3, 1)))

	print('\n\nDONE! SUCCESSFUL!!\n')

if __name__ == '__main__':
	test_get_2dline_from_pts_slope()