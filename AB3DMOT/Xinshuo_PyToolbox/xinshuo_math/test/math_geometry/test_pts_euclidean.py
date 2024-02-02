# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com
import numpy as np

import init_paths
from math_geometry import pts_euclidean
from xinshuo_miscellaneous import CHECK_EQ_LIST_ORDERED, CHECK_EQ_NUMPY

def test_pts_euclidean():
	print('check single list')
	pts1 = [0, 1]
	pts2 = [3, 5]
	dist, dist_list = pts_euclidean(pts1, pts2)
	assert CHECK_EQ_LIST_ORDERED(pts1, [0, 1])
	assert CHECK_EQ_LIST_ORDERED(pts2, [3, 5])
	assert dist == 5
	assert dist_list == [5]
	
	print('check list of list of 2 elements')
	pts1 = [[0, 1], [1, 1], [1, 4], [0, 0]]
	pts2 = [[3, 5], [4, 5], [6, 16], [1, 0]]
	dist, dist_list = pts_euclidean(pts1, pts2)
	assert CHECK_EQ_LIST_ORDERED(pts1, [[0, 1], [1, 1], [1, 4], [0, 0]])
	assert CHECK_EQ_LIST_ORDERED(pts2, [[3, 5], [4, 5], [6, 16], [1, 0]])
	assert dist == 6
	assert dist_list == [5, 5, 13, 1]

	print('check numpy array with 2 elements')
	pts1 = np.array([0, 1])
	pts2 = np.array([3, 5])
	dist, dist_list = pts_euclidean(pts1, pts2)
	assert CHECK_EQ_NUMPY(pts1, np.array([0, 1]))
	assert CHECK_EQ_NUMPY(pts2, np.array([3, 5]))
	assert dist == 5
	assert dist_list == [5]
	
	print('check numpy array with rows of 2 elements')
	pts1 = np.array([[0, 1], [1, 1], [1, 4], [0, 0]]).transpose()
	pts2 = np.array([[3, 5], [4, 5], [6, 16], [1, 0]]).transpose()
	print(pts1.shape)
	dist, dist_list = pts_euclidean(pts1, pts2)
	assert CHECK_EQ_NUMPY(pts1, np.array([[0, 1], [1, 1], [1, 4], [0, 0]]).transpose())
	assert CHECK_EQ_NUMPY(pts2, np.array([[3, 5], [4, 5], [6, 16], [1, 0]]).transpose())
	assert dist == 6
	assert dist_list == [5, 5, 13, 1]

	######################################## test edge cases
	print('check numpy array with rows of 2 elements')
	pts1 = np.random.rand(2, 0)
	pts2 = np.random.rand(2, 0)
	dist, dist_list = pts_euclidean(pts1, pts2)
	assert dist == 0
	assert dist_list == []	

	print('\n\nDONE! SUCCESSFUL!!\n')

if __name__ == '__main__':
	test_pts_euclidean()