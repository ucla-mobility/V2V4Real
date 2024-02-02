# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com
import os, sys, numpy as np, pytest

import init_paths
from type_check import isstring, islist, islogical, isscalar, isnparray, istuple, isfunction, isdict, isext, isrange

def test_isstring():
	input_test = ''
	assert isstring(input_test)
	input_test = './'
	assert isstring(input_test)
	input_test = 'test'
	assert isstring(input_test)
	input_test = 'test.txt'
	assert isstring(input_test)
	input_test = ('syt')			# tuple with length 1 is not a tuple
	assert isstring(input_test)
	input_test = str(11)			
	assert isstring(input_test)

	input_test = 123
	assert isstring(input_test) is False
	input_test = False
	assert isstring(input_test) is False
	input_test = dict()
	assert isstring(input_test) is False
	input_test = ['ss']
	assert isstring(input_test) is False
	input_test = np.array(('sss'))
	assert isstring(input_test) is False
	input_test = ('syt', 'ss')
	assert isstring(input_test) is False

def test_islist():
	input_test = []
	assert islist(input_test)
	input_test = ['']
	assert islist(input_test)
	input_test = [1]
	assert islist(input_test)
	input_test = [1, 2, 3]
	assert islist(input_test)
	input_test = [[], []]
	assert islist(input_test)
	input_test = list()
	assert islist(input_test)

	input_test = 123
	assert islist(input_test) is False
	input_test = False
	assert islist(input_test) is False
	input_test = dict()
	assert islist(input_test) is False
	input_test = 'ss'
	assert islist(input_test) is False
	input_test = np.array(('sss'))
	assert islist(input_test) is False
	input_test = ('syt')
	assert islist(input_test) is False

def test_islogical():
	input_test = True
	assert islogical(input_test)
	input_test = bool(1)
	assert islogical(input_test)
	input_test = bool(0)
	assert islogical(input_test)
	input_test = bool(-1)
	assert islogical(input_test)

	input_test = 1
	assert islogical(input_test) is False
	input_test = [1]
	assert islogical(input_test) is False
	input_test = dict()
	assert islogical(input_test) is False
	input_test = 'True'
	assert islogical(input_test) is False
	input_test = np.array([1]).astype('bool')
	assert islogical(input_test) is False
	input_test = (1)
	assert islogical(input_test) is False

def test_isnparray():
	input_test = np.array([])
	assert isnparray(input_test)
	input_test = np.asarray(1)
	assert isnparray(input_test)

	input_test = 123
	assert isnparray(input_test) is False
	input_test = False
	assert isnparray(input_test) is False
	input_test = dict()
	assert isnparray(input_test) is False
	input_test = 'ss'
	assert isnparray(input_test) is False
	input_test = []
	assert isnparray(input_test) is False
	input_test = ('syt')
	assert isnparray(input_test) is False

def test_istuple():
	input_test = ()
	assert istuple(input_test)
	input_test = (1, 2)
	assert istuple(input_test)

	input_test = (1)		# tuple with length 1 is not a tuple
	assert istuple(input_test) is False
	input_test = 1
	assert istuple(input_test) is False
	input_test = [1]
	assert istuple(input_test) is False
	input_test = dict()
	assert istuple(input_test) is False
	input_test = 'True'
	assert istuple(input_test) is False
	input_test = np.array([1]).astype('bool')
	assert istuple(input_test) is False
	input_test = True
	assert istuple(input_test) is False

def test_isfunction():
	input_test = test_istuple
	assert isfunction(input_test)

	input_test = 1
	assert isfunction(input_test) is False
	input_test = [1]
	assert isfunction(input_test) is False
	input_test = dict()
	assert isfunction(input_test) is False
	input_test = 'True'
	assert isfunction(input_test) is False
	input_test = np.array([1]).astype('bool')
	assert isfunction(input_test) is False
	input_test = True
	assert isfunction(input_test) is False

def test_isdict():
	input_test = dict()
	assert isdict(input_test)
	input_test = {}
	assert isdict(input_test)
	input_test = {'a':1, 'b':2}
	assert isdict(input_test)

	input_test = 1
	assert isdict(input_test) is False
	input_test = [1]
	assert isdict(input_test) is False
	input_test = (1, 2)
	assert isdict(input_test) is False
	input_test = 'True'
	assert isdict(input_test) is False
	input_test = np.array([1]).astype('bool')
	assert isdict(input_test) is False
	input_test = True
	assert isdict(input_test) is False

def test_isext():
	input_test = '.jpg'
	assert isext(input_test)
	
	input_test = '.'
	assert isext(input_test) is False
	input_test = 'aaa.jpg'
	assert isext(input_test) is False
	input_test = '.jpg.ext'
	assert isext(input_test) is False
	input_test = 1
	assert isext(input_test) is False
	input_test = [1]
	assert isext(input_test) is False
	input_test = (1, 2)
	assert isext(input_test) is False
	input_test = '.'
	assert isext(input_test) is False
	input_test = np.array([1]).astype('bool')
	assert isext(input_test) is False
	input_test = True
	assert isext(input_test) is False

def test_isrange():
	input_test = [0, 1]
	assert isrange(input_test)
	input_test = (0, 1)
	assert isrange(input_test)
	input_test = np.array([0, 1])
	assert isrange(input_test)

	input_test = 1
	assert isrange(input_test) is False
	input_test = [1]
	assert isrange(input_test) is False
	input_test = (1, 2, 3)
	assert isrange(input_test) is False
	input_test = 'True'
	assert isrange(input_test) is False
	input_test = np.array([1]).astype('bool')
	assert isrange(input_test) is False
	input_test = True
	assert isrange(input_test) is False
	input_test = dict()
	assert isrange(input_test) is False

def test_isscalar():
	input_test = 1
	assert isscalar(input_test)
	input_test = (1)			# tuple with length 1 is not a tuple
	assert isscalar(input_test)
	input_test = True
	assert isscalar(input_test)

	input_test = {'a':1}
	assert isscalar(input_test) is False
	input_test = [1]
	assert isscalar(input_test) is False
	input_test = (1, 2)
	assert isscalar(input_test) is False
	input_test = 'sss'
	assert isscalar(input_test) is False
	input_test = np.array([1]).astype('bool')
	assert isscalar(input_test) is False

if __name__ == '__main__':
	pytest.main([__file__])