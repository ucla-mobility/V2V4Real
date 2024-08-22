# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com
import pytest

import init_paths
from type_check import islistoflist, islistofdict, islistofscalar, islistofpositiveinteger, islistofnonnegativeinteger

def test_islistoflist():
	input_test = [[], []]
	assert islistoflist(input_test)
	input_test = [[1], [2], ['a']]
	assert islistoflist(input_test)
	input_test = [[[1], [3]], [2], ['a']]
	assert islistoflist(input_test)

	input_test = []
	assert islistoflist(input_test) is False
	input_test = ['']
	assert islistoflist(input_test) is False
	input_test = [1]
	assert islistoflist(input_test) is False
	input_test = [1, 2, 3]
	assert islistoflist(input_test) is False
	input_test = list()
	assert islistoflist(input_test) is False
	input_test = 123
	assert islistoflist(input_test) is False

def test_islistofdict():
	input_test = [dict(), dict()]
	assert islistofdict(input_test)
	input_test = [{}, {}, {'a':1}]
	assert islistofdict(input_test)
	input_test = [dict(b=[{'a':1}, {'b':3}]), dict(), {'a':2}]
	assert islistofdict(input_test)

	input_test = []
	assert islistofdict(input_test) is False
	input_test = [[], []]
	assert islistofdict(input_test) is False
	input_test = ['']
	assert islistofdict(input_test) is False
	input_test = [1]
	assert islistofdict(input_test) is False
	input_test = [1, 2, 3]
	assert islistofdict(input_test) is False
	input_test = list()
	assert islistofdict(input_test) is False
	input_test = 123
	assert islistofdict(input_test) is False

def test_islistofscalar():
	input_test = [1]
	assert islistofscalar(input_test)
	input_test = [1, 2, 3]
	assert islistofscalar(input_test)

	input_test = [1, 2, [1, 2, 3]]
	assert islistofscalar(input_test) is False
	input_test = [dict(), dict()]
	assert islistofscalar(input_test) is False
	input_test = [{}, {}, {'a':1}]
	assert islistofscalar(input_test) is False
	input_test = [dict(b=[{'a':1}, {'b':3}]), dict(), {'a':2}]
	assert islistofscalar(input_test) is False
	input_test = []
	assert islistofscalar(input_test) is False
	input_test = [[], []]
	assert islistofscalar(input_test) is False
	input_test = ['']
	assert islistofscalar(input_test) is False
	input_test = list()
	assert islistofscalar(input_test) is False
	input_test = 123
	assert islistofscalar(input_test) is False

def test_islistofpositiveinteger():
	input_test = [1]
	assert islistofpositiveinteger(input_test)
	input_test = [1, 2, 3]
	assert islistofpositiveinteger(input_test)

	input_test = [1, 2, 3.5]
	assert islistofpositiveinteger(input_test) is False
	input_test = [1, 2, 0]
	assert islistofpositiveinteger(input_test) is False
	input_test = [1, -1, 1]
	assert islistofpositiveinteger(input_test) is False
	input_test = [1, 2, [1, 2, 3]]
	assert islistofpositiveinteger(input_test) is False
	input_test = [dict(), dict()]
	assert islistofpositiveinteger(input_test) is False
	input_test = [{}, {}, {'a':1}]
	assert islistofpositiveinteger(input_test) is False
	input_test = [dict(b=[{'a':1}, {'b':3}]), dict(), {'a':2}]
	assert islistofpositiveinteger(input_test) is False
	input_test = []
	assert islistofpositiveinteger(input_test) is False
	input_test = [[], []]
	assert islistofpositiveinteger(input_test) is False
	input_test = ['']
	assert islistofpositiveinteger(input_test) is False
	input_test = list()
	assert islistofpositiveinteger(input_test) is False
	input_test = 123
	assert islistofpositiveinteger(input_test) is False

def test_islistofnonnegativeinteger():
	input_test = [1]
	assert islistofnonnegativeinteger(input_test)
	input_test = [1, 2, 3]
	assert islistofnonnegativeinteger(input_test)
	input_test = [1, 2, 0]
	assert islistofnonnegativeinteger(input_test)

	input_test = [1, 2, 3.5]
	assert islistofnonnegativeinteger(input_test) is False
	input_test = [1, -1, 1]
	assert islistofnonnegativeinteger(input_test) is False
	input_test = [1, 2, [1, 2, 3]]
	assert islistofnonnegativeinteger(input_test) is False
	input_test = [dict(), dict()]
	assert islistofnonnegativeinteger(input_test) is False
	input_test = [{}, {}, {'a':1}]
	assert islistofnonnegativeinteger(input_test) is False
	input_test = [dict(b=[{'a':1}, {'b':3}]), dict(), {'a':2}]
	assert islistofnonnegativeinteger(input_test) is False
	input_test = []
	assert islistofnonnegativeinteger(input_test) is False
	input_test = [[], []]
	assert islistofnonnegativeinteger(input_test) is False
	input_test = ['']
	assert islistofnonnegativeinteger(input_test) is False
	input_test = list()
	assert islistofnonnegativeinteger(input_test) is False
	input_test = 123
	assert islistofnonnegativeinteger(input_test) is False

if __name__ == '__main__':
	pytest.main([__file__])