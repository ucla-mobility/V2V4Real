# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com
import numpy as np, pytest

import init_paths
from type_check import isinteger, isfloat, ispositiveinteger, isnonnegativeinteger, isuintnparray, isfloatnparray, isnannparray

def test_isinteger():
	input_test = 1
	assert isinteger(input_test)
	input_test = (1)			# tuple with length 1 is not a tuple
	assert isinteger(input_test)
	input_test = True
	assert isinteger(input_test)

	input_test = 1.5
	assert isinteger(input_test) is False
	input_test = 1e-10
	assert isinteger(input_test) is False
	input_test = {'a':1}
	assert isinteger(input_test) is False
	input_test = [1]
	assert isinteger(input_test) is False
	input_test = (1, 2)
	assert isinteger(input_test) is False
	input_test = 'sss'
	assert isinteger(input_test) is False
	input_test = np.array([1]).astype('bool')
	assert isinteger(input_test) is False

def test_isfloat():
	input_test = (1.0)			# tuple with length 1 is not a tuple
	assert isfloat(input_test)
	input_test = 1.5
	assert isfloat(input_test)

	input_test = True
	assert isfloat(input_test) is False
	input_test = 1
	assert isfloat(input_test) is False
	input_test = {'a':1}
	assert isfloat(input_test) is False
	input_test = [1]
	assert isfloat(input_test) is False
	input_test = (1, 2)
	assert isfloat(input_test) is False
	input_test = 'sss'
	assert isfloat(input_test) is False
	input_test = np.array([1]).astype('bool')
	assert isfloat(input_test) is False

def test_ispositiveinteger():
	input_test = 1
	assert ispositiveinteger(input_test)
	input_test = (1)			# tuple with length 1 is not a tuple
	assert ispositiveinteger(input_test)
	input_test = True
	assert ispositiveinteger(input_test)

	input_test = 0
	assert ispositiveinteger(input_test) is False
	input_test = -1
	assert ispositiveinteger(input_test) is False
	input_test = 1.5
	assert ispositiveinteger(input_test) is False
	input_test = 1e-10
	assert ispositiveinteger(input_test) is False
	input_test = {'a':1}
	assert ispositiveinteger(input_test) is False
	input_test = [1]
	assert ispositiveinteger(input_test) is False
	input_test = (1, 2)
	assert ispositiveinteger(input_test) is False
	input_test = 'sss'
	assert ispositiveinteger(input_test) is False
	input_test = np.array([1]).astype('bool')
	assert ispositiveinteger(input_test) is False

def test_isnonnegativeinteger():
	input_test = 1
	assert isnonnegativeinteger(input_test)
	input_test = (1)			# tuple with length 1 is not a tuple
	assert isnonnegativeinteger(input_test)
	input_test = True
	assert isnonnegativeinteger(input_test)
	input_test = 0
	assert isnonnegativeinteger(input_test)

	input_test = -1
	assert isnonnegativeinteger(input_test) is False
	input_test = 1.5
	assert isnonnegativeinteger(input_test) is False
	input_test = 1e-10
	assert isnonnegativeinteger(input_test) is False
	input_test = {'a':1}
	assert isnonnegativeinteger(input_test) is False
	input_test = [1]
	assert isnonnegativeinteger(input_test) is False
	input_test = (1, 2)
	assert isnonnegativeinteger(input_test) is False
	input_test = 'sss'
	assert isnonnegativeinteger(input_test) is False
	input_test = np.array([1]).astype('bool')
	assert isnonnegativeinteger(input_test) is False

def test_isuintnparray():
	input_test = np.array([]).astype('uint8')
	assert isuintnparray(input_test)
	input_test = np.asarray(1).astype('uint8')
	assert isuintnparray(input_test)

	input_test = np.array([]).astype('uint16')
	assert isuintnparray(input_test) is False
	input_test = np.asarray(1).astype('float32')
	assert isuintnparray(input_test) is False
	input_test = 123
	assert isuintnparray(input_test) is False
	input_test = False
	assert isuintnparray(input_test) is False
	input_test = dict()
	assert isuintnparray(input_test) is False
	input_test = 'ss'
	assert isuintnparray(input_test) is False
	input_test = []
	assert isuintnparray(input_test) is False
	input_test = ('syt')
	assert isuintnparray(input_test) is False

def test_isfloatnparray():
	input_test = np.array([]).astype('float32')
	assert isfloatnparray(input_test)
	input_test = np.asarray(1).astype('float32')
	assert isfloatnparray(input_test)

	input_test = np.array([]).astype('uint8')
	assert isfloatnparray(input_test) is False
	input_test = np.asarray(1).astype('float64')
	assert isfloatnparray(input_test) is False
	input_test = 123
	assert isfloatnparray(input_test) is False
	input_test = False
	assert isfloatnparray(input_test) is False
	input_test = dict()
	assert isfloatnparray(input_test) is False
	input_test = 'ss'
	assert isfloatnparray(input_test) is False
	input_test = []
	assert isfloatnparray(input_test) is False
	input_test = ('syt')
	assert isfloatnparray(input_test) is False

def test_isnannparray():
	input_test = np.asarray([1, float('nan')])
	assert isnannparray(input_test)

	input_test = np.asarray([1, float('inf')])
	assert isnannparray(input_test) is False
	input_test = np.array([])
	assert isnannparray(input_test) is False
	input_test = np.asarray(1)
	assert isnannparray(input_test) is False
	input_test = 123
	assert isnannparray(input_test) is False
	input_test = False
	assert isnannparray(input_test) is False
	input_test = dict()
	assert isnannparray(input_test) is False
	input_test = 'ss'
	assert isnannparray(input_test) is False
	input_test = []
	assert isnannparray(input_test) is False
	input_test = ('syt')
	assert isnannparray(input_test) is False

if __name__ == '__main__':
	pytest.main([__file__])