# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com
import numpy as np

import init_paths
from conversion import remove_list_from_list
from xinshuo_miscellaneous import CHECK_EQ_LIST_ORDERED

def test_remove_list_from_list():
	print('check basic')
	list1 = [1, 2, 3, 4]
	list2 = [2, 4]
	list_remain, list_removed = remove_list_from_list(list1, list2)
	assert CHECK_EQ_LIST_ORDERED(list_remain, [1, 3])
	assert CHECK_EQ_LIST_ORDERED(list_removed, [2, 4])

	print('check basic with additional item')
	list1 = [1, 2, 3, 4]
	list2 = [2, 4, 5]
	list_remain, list_removed = remove_list_from_list(list1, list2)
	assert CHECK_EQ_LIST_ORDERED(list_remain, [1, 3])
	assert CHECK_EQ_LIST_ORDERED(list_removed, [2, 4])

	print('check basic with repetitive item')
	list1 = [1, 2, 3, 4, 2]
	list2 = [2, 4]
	list_remain, list_removed = remove_list_from_list(list1, list2)
	assert CHECK_EQ_LIST_ORDERED(list_remain, [1, 3, 2])
	assert CHECK_EQ_LIST_ORDERED(list_removed, [2, 4])

	print('check basic with repetitive item')
	list1 = [1, 2, 3, 4, 2]
	list2 = [2, 4, 2]
	list_remain, list_removed = remove_list_from_list(list1, list2)
	assert CHECK_EQ_LIST_ORDERED(list_remain, [1, 3])
	assert CHECK_EQ_LIST_ORDERED(list_removed, [2, 4, 2])


	print('\n\nDONE! SUCCESSFUL!!\n')

if __name__ == '__main__':
	test_remove_list_from_list()