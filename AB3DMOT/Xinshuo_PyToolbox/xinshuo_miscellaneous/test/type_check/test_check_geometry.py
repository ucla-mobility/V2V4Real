# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com
import os, sys, numpy as np, pytest

import init_paths
from type_check import is2dpts, is3dpts, is2dptsarray, is3dptsarray, is2dptsarray_occlusion, is2dptsarray_confidence

def test_is2dpts():
	pts = [1, 3]
	assert is2dpts(pts)
	pts = (2, 4)
	assert is2dpts(pts)
	pts = np.array([3, 6])
	assert is2dpts(pts)
	pts = [1, 3, 100]
	assert not is2dpts(pts)
	pts = (2, 4, 3)
	assert not is2dpts(pts)
	pts = np.array([3])
	assert not is2dpts(pts)

def test_is2dptsarray():
	pts = np.array([[1, 3], [2, 4]])
	assert is2dptsarray(pts)
	pts = np.array([[1, 3, 5], [2, 4, 3]])
	assert is2dptsarray(pts)
	pts = np.random.rand(2, 0)
	assert is2dptsarray(pts)
	pts = np.random.rand(3, 0)
	assert not is2dptsarray(pts)
	pts = [[1, 3], [2, 4]]
	assert not is2dptsarray(pts)
	pts = np.random.rand(1, 0)
	assert not is2dptsarray(pts)

def test_is3dpts():
	pts = [1, 3, 2]
	assert is3dpts(pts)
	pts = (2, 4, 1)
	assert is3dpts(pts)
	pts = np.array([3, 3, 6])
	assert is3dpts(pts)
	pts = [1, 3]
	assert not is3dpts(pts)
	pts = (3)
	assert not is3dpts(pts)
	pts = np.array([3])
	assert not is3dpts(pts)

def test_is3dptsarray():
	pts = np.array([[1, 3], [2, 4], [1, 2]])
	assert is3dptsarray(pts)
	pts = np.random.rand(3, 10)
	assert is3dptsarray(pts)
	pts = np.random.rand(3, 0)
	assert is3dptsarray(pts)
	pts = np.random.rand(2, 0)
	assert not is3dptsarray(pts)
	pts = [[1, 3], [2, 4]]
	assert not is3dptsarray(pts)
	pts = np.random.rand(1, 0)
	assert not is3dptsarray(pts)

def test_is2dptsarray_occlusion():
	pts = np.array([[1, 3, 0], [2, 4, -1], [1, 2, 1]])
	assert not is2dptsarray_occlusion(pts)
	pts = np.array([[1, 3, 1], [2, 4, 3], [1, 0, -1]])
	assert is2dptsarray_occlusion(pts)
	pts = np.random.rand(3, 0)
	assert is2dptsarray_occlusion(pts)
	pts = np.random.rand(2, 0)
	assert not is2dptsarray_occlusion(pts)
	pts = np.random.rand(3, 10)
	pts[-1, :] = -1
	assert is2dptsarray_occlusion(pts)
	pts = np.random.rand(3, 10)
	assert not is2dptsarray_occlusion(pts)

def test_is2dptsarray_confidence():
	pts = np.array([[1, 3, 0], [2, 4, -1], [1, 2, 1]])
	assert not is2dptsarray_confidence(pts)
	pts = np.array([[1, 3, 1], [2, 4, 3], [0.5, 0, 1]])
	assert is2dptsarray_confidence(pts)
	pts = np.random.rand(3, 0)
	assert is2dptsarray_confidence(pts)
	pts = np.random.rand(2, 0)
	assert not is2dptsarray_confidence(pts)
	pts = np.random.rand(3, 10)
	assert is2dptsarray_confidence(pts)
	pts = np.random.rand(3, 10)
	pts[-1, :] = -1
	assert not is2dptsarray_confidence(pts)


if __name__ == '__main__':
	pytest.main([__file__])