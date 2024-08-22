# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com
import numpy as np

import init_paths
from bbox_transform import bbox_TLWH2TLBR
from xinshuo_miscellaneous import CHECK_EQ_NUMPY

def test_bbox_TLWH2TLBR():
	print('check basic')
	bbox = [1, 1, 10, 10]
	clipped = bbox_TLWH2TLBR(bbox)
	print(clipped)
	assert CHECK_EQ_NUMPY(clipped, np.array([1, 1, 11, 11]).reshape((1, 4)))
	
	print('check out of boundary and 0 height')
	bbox = [-1, 3, 20, 0]
	clipped = bbox_TLWH2TLBR(bbox)
	print(clipped)
	assert CHECK_EQ_NUMPY(clipped, np.array([-1, 3, 19, 3]).reshape((1, 4)))

	print('check 0 height and width')
	bbox = [10, 30, 0, 0]
	clipped = bbox_TLWH2TLBR(bbox)
	print(clipped)
	assert CHECK_EQ_NUMPY(clipped, np.array([10, 30, 10, 30]).reshape((1, 4)))

	print('check multi bboxes')
	bbox = np.array([[10, 30, 0, 0], [-1, 3, 20, 0]])
	clipped = bbox_TLWH2TLBR(bbox)
	print(clipped)
	assert CHECK_EQ_NUMPY(clipped, np.array([[10, 30, 10, 30], [-1, 3, 19, 3]]).reshape((2, 4)))

	print('check width < 0')
	bbox = [10, 30, -1, 29]
	try:
		clipped = bbox_TLWH2TLBR(bbox)
		sys.exit('\nwrong! never should be here\n\n')
	except AssertionError:
		print('the width should be no less than 0')

	print('\n\nDONE! SUCCESSFUL!!\n')

if __name__ == '__main__':
	test_bbox_TLWH2TLBR()