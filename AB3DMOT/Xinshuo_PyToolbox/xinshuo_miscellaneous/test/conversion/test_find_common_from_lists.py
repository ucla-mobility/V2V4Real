# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com
import numpy as np

import init_paths
from conversion import find_unique_common_from_lists
from xinshuo_miscellaneous import CHECK_EQ_LIST_ORDERED

def test_find_common_from_lists():
	print('check basic')
	list1 = [1, 2, 3, 4]
	list2 = [2, 4, 5]
	list_common = find_unique_common_from_lists(list1, list2)
	assert CHECK_EQ_LIST_ORDERED(list_common, [2, 4])

	print('check basic with empty')
	list1 = [1, 2, 3, 4]
	list2 = []
	list_common = find_unique_common_from_lists(list1, list2)
	assert CHECK_EQ_LIST_ORDERED(list_common, [])

	print('check basic with all empty')
	list1 = []
	list2 = []
	list_common = find_unique_common_from_lists(list1, list2)
	assert CHECK_EQ_LIST_ORDERED(list_common, [])

	print('check basic with different order')
	list1 = [1, 2, 3, 4, 2]
	list2 = [2, 4, 2, 3, 1]
	list_common = find_unique_common_from_lists(list1, list2)
	assert CHECK_EQ_LIST_ORDERED(list_common, [1, 2, 3, 4])

	print('\n\nDONE! SUCCESSFUL!!\n')

if __name__ == '__main__':
	test_find_common_from_lists()