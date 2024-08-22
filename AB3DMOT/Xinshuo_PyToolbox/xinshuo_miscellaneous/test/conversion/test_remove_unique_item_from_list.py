# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com
import numpy as np

import init_paths
from conversion import remove_unique_item_from_list
from xinshuo_miscellaneous import CHECK_EQ_LIST_ORDERED

def test_remove_unique_item_from_list():
	print('check basic')
	list1 = [1, 2, 3, 4]
	item = 2
	list_remain, count = remove_unique_item_from_list(list1, item)
	assert CHECK_EQ_LIST_ORDERED(list_remain, [1, 3, 4])
	assert count == 1

	print('check empty list')
	list1 = []
	item = 2
	list_remain, count = remove_unique_item_from_list(list1, item)
	assert CHECK_EQ_LIST_ORDERED(list_remain, [])
	assert count == 0

	print('check multiple instances to remove')
	list1 = [1, 2, 3, 2, 4, 2]
	item = 2
	list_remain, count = remove_unique_item_from_list(list1, item)
	assert CHECK_EQ_LIST_ORDERED(list_remain, [1, 3, 4])
	assert count == 3

	print('check item not in the list')
	list1 = [1, 2, 3, 2, 4, 2]
	item = 'a'
	list_remain, count = remove_unique_item_from_list(list1, item)
	assert CHECK_EQ_LIST_ORDERED(list_remain, [1, 2, 3, 2, 4, 2])
	assert count == 0

	print('\n\nDONE! SUCCESSFUL!!\n')

if __name__ == '__main__':
	test_remove_unique_item_from_list()