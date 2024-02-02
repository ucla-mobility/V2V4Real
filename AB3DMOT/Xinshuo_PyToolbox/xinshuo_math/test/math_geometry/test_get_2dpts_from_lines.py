# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com
import numpy as np

import init_paths
from math_geometry import get_2dpts_from_lines
from xinshuo_miscellaneous import CHECK_EQ_NUMPY

def test_get_2dpts_from_lines():
	print('check normal point')
	line1 = [1, 1, -2]
	line2 = [1, -1, 0]
	pts = get_2dpts_from_lines(line1, line2)
	assert CHECK_EQ_NUMPY(pts, np.array([-2, -2, -2]).reshape((3, 1)))
	
	print('check vertical and horizontal line')
	line1 = [1, 0, -1]
	line2 = [0, 1, -1]
	pts = get_2dpts_from_lines(line1, line2)
	assert CHECK_EQ_NUMPY(pts, np.array([1, 1, 1]).reshape((3, 1)))

	print('\n\nDONE! SUCCESSFUL!!\n')

if __name__ == '__main__':
	test_get_2dpts_from_lines()