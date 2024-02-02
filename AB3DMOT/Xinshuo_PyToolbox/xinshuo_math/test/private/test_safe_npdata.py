# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com
import numpy as np, sys, copy

import init_paths
from private import safe_npdata
from xinshuo_miscellaneous import CHECK_EQ_NUMPY, CHECK_EQ_LIST_ORDERED

def test_safe_npdata():
	print('test list with multiple values')
	data = [1, 2, 3]
	data_bak = copy.copy(data)
	npdata = safe_npdata(data)
	assert CHECK_EQ_NUMPY(npdata, np.array(data))
	npdata += 100
	assert CHECK_EQ_LIST_ORDERED(data, data_bak)

	print('test list with single value')
	data = [1]
	npdata = safe_npdata(data)
	assert CHECK_EQ_NUMPY(npdata, np.array(data))

	print('test scalar')
	data = 10
	npdata = safe_npdata(data)
	assert CHECK_EQ_NUMPY(npdata, np.array(data))

	######################################## test failure cases
	print('test edge case: tuple')
	data = (1, 2)
	try:
		npdata = safe_npdata(data)
		sys.exit('\nwrong! never should be here\n\n')
	except TypeError:
		print('the input should never be a tuple')

	print('\n\nDONE! SUCCESSFUL!!\n')

if __name__ == '__main__':
	test_safe_npdata()