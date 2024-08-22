# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com
import os, sys, numpy as np, pytest
from PIL import Image

import init_paths
from type_check import isimsize, isimage_dimension, iscolorimage_dimension, isgrayimage_dimension, isuintimage, isfloatimage, isnpimage, ispilimage, isimage

def test_isimsize():
	input_test = np.zeros((100, 100), dtype='uint8')
	input_test = input_test.shape
	assert isimsize(input_test)
	input_test = [100, 200]
	assert isimsize(input_test)
	input_test = (100, 200)
	assert isimsize(input_test)
	input_test = np.array([100, 200])
	assert isimsize(input_test)

	input_test = np.zeros((100, 100, 3), dtype='float32')
	input_test = input_test.shape
	assert isimsize(input_test) is False
	input_test = [100, 200, 3]
	assert isimsize(input_test) is False
	input_test = (100, 200, 3)
	assert isimsize(input_test) is False

def test_ispilimage():
	input_test = Image.fromarray(np.zeros((100, 100, 3), dtype='uint8'))
	assert ispilimage(input_test)
	input_test = Image.fromarray(np.zeros((100, 100), dtype='uint8'))
	assert ispilimage(input_test)

	input_test = np.zeros((100, 100), dtype='uint8')
	assert ispilimage(input_test) is False
	input_test = np.zeros((100, 100), dtype='float32')
	assert ispilimage(input_test) is False

def test_iscolorimage_dimension():
	input_test = np.zeros((100, 100, 4), dtype='uint8')
	assert iscolorimage_dimension(input_test)
	input_test = np.zeros((100, 100, 3), dtype='uint8')
	assert iscolorimage_dimension(input_test)
	input_test = np.zeros((100, 100, 4), dtype='float32')
	assert iscolorimage_dimension(input_test)
	input_test = np.zeros((100, 100, 3), dtype='float64')
	assert iscolorimage_dimension(input_test)
	input_test = Image.fromarray(np.zeros((100, 100, 3), dtype='uint8'))
	assert iscolorimage_dimension(input_test)

	input_test = Image.fromarray(np.zeros((100, 100), dtype='uint8'))
	assert iscolorimage_dimension(input_test) is False
	input_test = np.zeros((100, 100), dtype='float32')
	assert iscolorimage_dimension(input_test) is False
	input_test = np.zeros((100, 100, 1), dtype='uint8')
	assert iscolorimage_dimension(input_test) is False
	input_test = np.zeros((100, 100, 2), dtype='uint8')
	assert iscolorimage_dimension(input_test) is False

def test_isgrayimage_dimension():
	input_test = np.zeros((100, 100, 1), dtype='uint8')
	assert isgrayimage_dimension(input_test)
	input_test = np.zeros((100, 100), dtype='uint8')
	assert isgrayimage_dimension(input_test)
	input_test = np.zeros((100, 100, 1), dtype='float32')
	assert isgrayimage_dimension(input_test)
	input_test = np.zeros((100, 100), dtype='float64')
	assert isgrayimage_dimension(input_test)
	input_test = Image.fromarray(np.zeros((100, 100), dtype='uint8'))
	assert isgrayimage_dimension(input_test)

	input_test = Image.fromarray(np.zeros((100, 100, 3), dtype='uint8'))
	assert isgrayimage_dimension(input_test) is False
	input_test = np.zeros((100, 100, 3), dtype='float32')
	assert isgrayimage_dimension(input_test) is False
	input_test = np.zeros((100, 100, 4), dtype='uint8')
	assert isgrayimage_dimension(input_test) is False
	input_test = np.zeros((100, 100, 2), dtype='uint8')
	assert isgrayimage_dimension(input_test) is False

def test_isimage_dimension():
	input_test = np.zeros((100, 100, 1), dtype='uint8')
	assert isimage_dimension(input_test)
	input_test = np.zeros((100, 100), dtype='uint8')
	assert isimage_dimension(input_test)
	input_test = np.zeros((100, 100, 3), dtype='float32')
	assert isimage_dimension(input_test)
	input_test = np.zeros((100, 100, 4), dtype='float64')
	assert isimage_dimension(input_test)
	input_test = Image.fromarray(np.zeros((100, 100), dtype='uint8'))
	assert isimage_dimension(input_test)
	input_test = Image.fromarray(np.zeros((100, 100, 3), dtype='uint8'))
	assert isimage_dimension(input_test)

	input_test = np.zeros((100, 100, 3, 1), dtype='float32')
	assert isimage_dimension(input_test) is False
	input_test = np.zeros((100, 100, 2), dtype='uint8')
	assert isimage_dimension(input_test) is False

def test_isuintimage():
	input_test = np.random.rand(100, 100).astype('uint8')
	assert isuintimage(input_test)
	input_test = np.zeros((100, 100, 3), dtype='uint8')
	assert isuintimage(input_test)
	input_test = np.zeros((100, 100, 1), dtype='uint8')
	assert isuintimage(input_test)
	input_test = np.zeros((100, 100, 4), dtype='uint8')
	assert isuintimage(input_test)
	input_test = Image.fromarray(np.zeros((100, 100, 3), dtype='uint8'))
	assert isimage_dimension(input_test)
	input_test = (np.random.rand(100, 100) * 255.).astype('uint8') - 255
	assert isuintimage(input_test)
	input_test = (np.random.rand(100, 100) * 255.).astype('uint8') + 255
	assert isuintimage(input_test)

	input_test = np.zeros((100, 100), dtype='float32')
	assert isuintimage(input_test) is False
	input_test = np.zeros((100, 100), dtype='float64')
	assert isuintimage(input_test) is False

def test_isfloatimage():
	input_test = np.zeros((100, 100), dtype='float32')
	assert isfloatimage(input_test)
	input_test = np.zeros((100, 100, 3), dtype='float32')
	assert isfloatimage(input_test)
	input_test = np.zeros((100, 100, 1), dtype='float32')
	assert isfloatimage(input_test)
	input_test = np.ones((100, 100, 1), dtype='float32')
	assert isfloatimage(input_test)
	input_test = np.zeros((100, 100, 4), dtype='float32')
	assert isfloatimage(input_test)
	
	input_test = np.zeros((100, 100), dtype='uint8')
	assert isfloatimage(input_test) is False
	input_test = np.zeros((100, 100), dtype='float64')
	assert isfloatimage(input_test) is False
	input_test = np.ones((100, 100, 3), dtype='float32')
	input_test[0, 0, 0] += 1e-5
	assert isfloatimage(input_test) is False
	input_test = np.zeros((100, 100, 3), dtype='float32')
	input_test[0, 0, 0] -= 1e-5
	assert isfloatimage(input_test) is False

def test_isnpimage():
	input_test = np.zeros((100, 100), dtype='uint8')
	assert isnpimage(input_test)
	input_test = np.zeros((100, 100, 3), dtype='float32')
	assert isnpimage(input_test)
	input_test = np.zeros((100, 100, 4), dtype='float32')
	assert isnpimage(input_test)

	input_test = Image.fromarray(np.zeros((100, 100, 3), dtype='uint8'))
	assert isnpimage(input_test) is False
	input_test = Image.fromarray(np.zeros((100, 100), dtype='uint8'))
	assert isnpimage(input_test) is False
	input_test = np.ones((100, 100, 3), dtype='float32')
	input_test[0, 0, 0] += 1e-5
	assert isnpimage(input_test) is False
	input_test = np.zeros((100, 100, 3), dtype='float32')
	input_test[0, 0, 0] -= 1e-5
	assert isnpimage(input_test) is False

def test_isimage():
	input_test = np.zeros((100, 100), dtype='uint8')
	assert isimage(input_test)
	input_test = np.zeros((100, 100, 3), dtype='float32')
	assert isimage(input_test)
	input_test = np.zeros((100, 100, 4), dtype='float32')
	assert isimage(input_test)
	input_test = Image.fromarray(np.zeros((100, 100, 3), dtype='uint8'))
	assert isimage(input_test)
	input_test = Image.fromarray(np.zeros((100, 100), dtype='uint8'))
	assert isimage(input_test)

	input_test = np.zeros((100, 100, 4), dtype='float64')
	assert isimage(input_test) is False
	input_test = np.zeros((100, 100, 2), dtype='float32')
	assert isimage(input_test) is False
	input_test = np.ones((100, 100, 3), dtype='float32')
	input_test[0, 0, 0] += 1e-5
	assert isnpimage(input_test) is False
	input_test = np.zeros((100, 100, 3), dtype='float32')
	input_test[0, 0, 0] -= 1e-5
	assert isnpimage(input_test) is False
	
if __name__ == '__main__':
	pytest.main([__file__])