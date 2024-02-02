# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com
import numpy as np

import init_paths
from conversion import reverse_list
from xinshuo_miscellaneous import CHECK_EQ_LIST_ORDERED

def test_reverse_list():
	print('check basic')
	list1 = [1, 2, 3, 4]
	reversed_list = reverse_list(list1)
	assert CHECK_EQ_LIST_ORDERED(reversed_list, [4, 3, 2, 1])

	print('check basic with repetitive item')
	list1 = [1, 2, 2, 3, 2, 4]
	reversed_list = reverse_list(list1)
	assert CHECK_EQ_LIST_ORDERED(reversed_list, [4, 2, 3, 2, 2, 1])

	print('check basic with empty')
	list1 = []
	reversed_list = reverse_list(list1)
	assert CHECK_EQ_LIST_ORDERED(reversed_list, [])

	print('\n\nDONE! SUCCESSFUL!!\n')

if __name__ == '__main__':
	test_reverse_list()