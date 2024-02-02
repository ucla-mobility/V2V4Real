# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com
import numpy as np

import init_paths
from file_io import mkdir_if_missing

def test_mkdir_if_missing():
	print('test repetitive')
	path = './mnt/dome/adhoc_0.5x/abd'
	mkdir_if_missing(path)

	print('test repetitive')
	path = './'
	mkdir_if_missing(path)

	print('test basic')
	path = 'test_folder'
	mkdir_if_missing(path)

	print('test recursive folder')
	path = 'test_folder1/test3/test4'
	mkdir_if_missing(path)

	print('test recursive file')
	path = 'test_folder1/test2/test3/te.txt'
	mkdir_if_missing(path)

	print('test edge case')
	try:
		path = 2
		mkdir_if_missing(path)
		sys.exit('\nwrong! never should be here\n\n')
	except AssertionError:
		print('the input should be a string')

	print('\n\nDONE! SUCCESSFUL!!\n')
	
if __name__ == '__main__':
	test_mkdir_if_missing()