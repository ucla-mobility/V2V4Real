# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com
import numpy as np, sys

import init_paths
from private import safe_angle
# from xinshuo_miscellaneous import CHECK_EQ_NUMPY

def test_safe_angle():
	print('test normal case')
	angle = 100
	safed = safe_angle(angle)
	assert safed == 100

	print('test normal case more than 180')
	angle = 300
	safed = safe_angle(angle)
	assert safed == -60

	print('test normal case less than -180')
	angle = -300
	safed = safe_angle(angle)
	assert safed == 60

	print('test edge case 180')
	angle = 180
	safed = safe_angle(angle)
	assert safed == 180

	print('test edge case -180')
	angle = -180
	safed = safe_angle(angle)
	assert safed == 180

	print('test edge case: single numpy')
	angle = np.array([100])
	safed = safe_angle(angle)
	assert safed == 100

	######################################## test failure cases
	print('test edge case: list')
	angle = [-180]
	try:
		safed = safe_angle(angle)
		sys.exit('\nwrong! never should be here\n\n')
	except TypeError:
		print('the length of list should be a scalar')

	print('\n\nDONE! SUCCESSFUL!!\n')

if __name__ == '__main__':
	test_safe_angle()