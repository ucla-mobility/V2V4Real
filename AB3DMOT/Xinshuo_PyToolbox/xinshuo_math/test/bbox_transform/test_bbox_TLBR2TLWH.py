# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com
import numpy as np

import init_paths
from bbox_transform import bbox_TLBR2TLWH
from xinshuo_miscellaneous import CHECK_EQ_NUMPY

def test_bbox_TLBR2TLWH():
	print('check basic')
	bbox = [1, 1, 10, 10]
	clipped = bbox_TLBR2TLWH(bbox)
	print(clipped)
	assert CHECK_EQ_NUMPY(clipped, np.array([1, 1, 9, 9]).reshape((1, 4)))
	
	print('check out of boundary and 0 height')
	bbox = [-1, 3, 20, 3]
	clipped = bbox_TLBR2TLWH(bbox)
	print(clipped)
	assert CHECK_EQ_NUMPY(clipped, np.array([-1, 3, 21, 0]).reshape((1, 4)))

	print('check 0 height and width')
	bbox = [10, 30, 10, 30]
	clipped = bbox_TLBR2TLWH(bbox)
	print(clipped)
	assert CHECK_EQ_NUMPY(clipped, np.array([10, 30, 0, 0]).reshape((1, 4)))

	print('check multi bboxes')
	bbox = np.array([[10, 30, 10, 30], [-1, 3, 20, 3]])
	clipped = bbox_TLBR2TLWH(bbox)
	print(clipped)
	assert CHECK_EQ_NUMPY(clipped, np.array([[10, 30, 0, 0], [-1, 3, 21, 0]]).reshape((2, 4)))

	print('check x2 < x1')
	bbox = [10, 30, 9, 29]
	try:
		clipped = bbox_TLBR2TLWH(bbox)
		sys.exit('\nwrong! never should be here\n\n')
	except AssertionError:
		print('the bottom right point coordinate should be no less than the top left one')

	print('\n\nDONE! SUCCESSFUL!!\n')

if __name__ == '__main__':
	test_bbox_TLBR2TLWH()