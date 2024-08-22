# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com
import numpy as np

import init_paths
from bbox_transform import clip_bboxes_TLWH
from xinshuo_miscellaneous import CHECK_EQ_NUMPY

def test_clip_bboxes_TLWH():
	print('check basic')
	bbox = [1, 1, 3, 3]
	clipped = clip_bboxes_TLWH(bbox, 5, 5)
	print(clipped)
	assert CHECK_EQ_NUMPY(clipped, np.array([1, 1, 3, 3]).reshape((1, 4)))
	
	print('check top left intersected')
	bbox = [-1, -1, 3, 3]
	clipped = clip_bboxes_TLWH(bbox, 5, 5)
	print(clipped)
	assert CHECK_EQ_NUMPY(clipped, np.array([0, 0, 2, 2]).reshape((1, 4)))

	print('check bottom right intersected')
	bbox = [2, 3, 10, 30]
	clipped = clip_bboxes_TLWH(bbox, 5, 5)
	print(clipped)
	assert CHECK_EQ_NUMPY(clipped, np.array([2, 3, 3, 2]).reshape((1, 4)))

	print('check left intersected')
	bbox = [-1, -2, 3, 30]
	clipped = clip_bboxes_TLWH(bbox, 5, 5)
	print(clipped)
	assert CHECK_EQ_NUMPY(clipped, np.array([0, 0, 2, 5]).reshape((1, 4)))

	print('check right intersected')
	bbox = [2, -3, 10, 30]
	clipped = clip_bboxes_TLWH(bbox, 5, 5)
	print(clipped)
	assert CHECK_EQ_NUMPY(clipped, np.array([2, 0, 3, 5]).reshape((1, 4)))

	print('check all intersected')
	bbox = [-2, -3, 10, 30]
	clipped = clip_bboxes_TLWH(bbox, 5, 5)
	print(clipped)
	assert CHECK_EQ_NUMPY(clipped, np.array([0, 0, 5, 5]).reshape((1, 4)))

	print('check bottom right outside')
	bbox = [10, 10, 10, 30]
	clipped = clip_bboxes_TLWH(bbox, 5, 5)
	print(clipped)
	assert CHECK_EQ_NUMPY(clipped, np.array([5, 5, 0, 0]).reshape((1, 4)))

	print('check top left outside')
	bbox = [-10, -10, 5, 5]
	clipped = clip_bboxes_TLWH(bbox, 5, 5)
	print(clipped)
	assert CHECK_EQ_NUMPY(clipped, np.array([0, 0, 0, 0]).reshape((1, 4)))

	print('check same height')
	bbox = [-2, 3, 10, 0]
	clipped = clip_bboxes_TLWH(bbox, 5, 5)
	print(clipped)
	assert CHECK_EQ_NUMPY(clipped, np.array([0, 3, 5, 0]).reshape((1, 4)))

	print('check same width')
	bbox = [2, -3, 0, 20]
	clipped = clip_bboxes_TLWH(bbox, 5, 5)
	print(clipped)
	assert CHECK_EQ_NUMPY(clipped, np.array([2, 0, 0, 5]).reshape((1, 4)))

	print('check all same')
	bbox = [2, 3, 0, 0]
	clipped = clip_bboxes_TLWH(bbox, 5, 5)
	print(clipped)
	assert CHECK_EQ_NUMPY(clipped, np.array([2, 3, 0, 0]).reshape((1, 4)))

	print('check for multiple bboxes')
	bbox = np.array([[2, 3, 0, 0], [2, -3, 0, 20], [-2, 3, 10, 0]])
	clipped = clip_bboxes_TLWH(bbox, 5, 5)
	print(clipped)
	assert CHECK_EQ_NUMPY(clipped, np.array([[2, 3, 0, 0], [2, 0, 0, 5], [0, 3, 5, 0]]).reshape((3, 4)))

	print('check for failure case')
	bbox = [-10, 30, -1, -1]
	try:
		clipped = clip_bboxes_TLWH(bbox, 5, 5)
		sys.exit('\nwrong! never should be here\n\n')
	except AssertionError:
		print('the bottom right point coordinate should be larger than bottom left one')

	print('\n\nDONE! SUCCESSFUL!!\n')

if __name__ == '__main__':
	test_clip_bboxes_TLWH()