# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com
import numpy as np

import init_paths
from file_io import load_list_from_folder

def test_load_list_from_folder():
	print('basic')
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../test')
    datalist, num_elem = load_list_from_folder(folder_path=path, ext_filter='txt')   
    assert datalist[0] == os.path.abspath('test.txt')
    assert datalist[1] == os.path.abspath('test1.txt')
    assert num_elem == 2

    # datalist, num_elem = load_list_from_folder(folder_path=path)
    # assert num_elem == 8


	print('\n\nDONE! SUCCESSFUL!!\n')
	
if __name__ == '__main__':
	test_load_list_from_folder()