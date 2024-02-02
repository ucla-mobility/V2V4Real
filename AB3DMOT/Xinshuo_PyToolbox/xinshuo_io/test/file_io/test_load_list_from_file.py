# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com
import numpy as np

import init_paths
from file_io import load_list_from_file

def test_load_list_from_file():
	image_path = '../lena.png'

	print('basic')
    path = 'test1.txt'
    datalist, num_elem = load_list_from_file(path)
    assert datalist[0] == '/home/xinshuow/test'
    assert datalist[1] == '/home/xinshuow/toy'
    assert num_elem == 2

	print('\n\nDONE! SUCCESSFUL!!\n')
	
if __name__ == '__main__':
	test_load_list_from_file()