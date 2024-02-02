# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com
import numpy as np

import init_paths
from math_geometry import get_slope_from_pts
from numpy.testing import assert_almost_equal

def test_get_slope_from_pts():
	print('check normal line')
	pts1 = [0, 1, 1]
	pts2 = [1, 0, 1]
	slope = get_slope_from_pts(pts1, pts2)
	assert slope == -45
	
	print('check vertical line')
	pts1 = [1, 1, 1]
	pts2 = [1, 12, 1]
	slope = get_slope_from_pts(pts1, pts2)
	assert slope == 90

	print('check horizontal line')
	pts1 = [20, 0, 1]
	pts2 = [1, 0, 1]
	slope = get_slope_from_pts(pts1, pts2)
	assert slope == 0

	print('check a point at infinity')
	pts1 = [0, 3, 0]
	pts2 = [1, 0, 1]
	slope = get_slope_from_pts(pts1, pts2)
	assert slope == 90

	print('check line at infinity')
	pts1 = [0, 3, 0]
	pts2 = [1, 0, 0]
	slope = get_slope_from_pts(pts1, pts2)
	assert slope == 90

	print('\n\nDONE! SUCCESSFUL!!\n')

if __name__ == '__main__':
	test_get_slope_from_pts()