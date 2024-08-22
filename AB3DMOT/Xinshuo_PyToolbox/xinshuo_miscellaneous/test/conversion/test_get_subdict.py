# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com
import os, sys
import numpy as np
import pytest

import __init__paths__
from check import *
from conversions import *

def test_get_subdict():
	dictionary = dict()
	for i in range(10):
		dictionary[str(i)] = i
	subdict = get_subdict(dictionary, 5)

	dictionary_test = dict()
	for i in range(5):
		dictionary_test[str(i)] = i
	CHECK_EQ_DICT(dictionary_test, subdict)

if __name__ == '__main__':
	pytest.main([__file__])