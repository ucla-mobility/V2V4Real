# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com
import init_paths, pytest

from type_check import is_path_exists, is_path_valid, is_path_creatable, is_path_exists_or_creatable, isfolder, isfile

def test_is_path_valid():
	path = './'
	assert is_path_valid(path)
	path = 'test'
	assert is_path_valid(path)
	path = 'test.txt'
	assert is_path_valid(path)

	path = ''
	assert is_path_valid(path) is False
	path = 123
	assert is_path_valid(path) is False

def test_is_path_creatable():
	path = './'
	assert is_path_creatable(path)
	path = 'test'
	assert is_path_creatable(path)
	path = 'test.txt'
	assert is_path_creatable(path)
	path = '/home/xinshuo/aaa.txt'
	assert is_path_creatable(path)
	path = '/home/xinshuo/aaa/aaa'
	assert is_path_creatable(path)

	path = 123
	assert is_path_creatable(path) is False
	path = '/usr'
	assert is_path_creatable(path) is False
	path = ''
	assert is_path_creatable(path) is False
	path = '/mnt/sdc1/xinshuow/dataset/mnist/mnist/train/images/image.jpg'
	assert is_path_creatable(path) is False
	path = '/mnt/sdc1/xinshuow/dataset/mnist/mnist/train/images/image1/image2/image3'
	assert is_path_creatable(path) is False

def test_is_path_exists():
	path = './'
	assert is_path_exists(path)
	path = '../../test'
	assert is_path_exists(path)
	
	path = ''
	assert is_path_exists(path) is False
	path = 'test'
	assert is_path_exists(path) is False
	path = 123
	assert is_path_exists(path) is False
	path = 'test.txt'
	assert is_path_exists(path) is False
	path = '/mnt/sdc1/xinshuow/dataset/mnist/mnist/train/images/image0000001.jpg'
	assert is_path_exists(path) is False

def test_is_path_exists_or_creatable():
	path = './'
	assert is_path_exists_or_creatable(path)
	path = 'test'
	assert is_path_exists_or_creatable(path)
	path = 'test.txt'
	assert is_path_exists_or_creatable(path)
	path = '../test'
	assert is_path_exists_or_creatable(path)
	
	path = '/mnt/sdc1/xinshuow/dataset/mnist/mnist/train/images/image0000001.jpg'
	assert is_path_exists_or_creatable(path) is False
	path = ''
	assert is_path_exists_or_creatable(path) is False
	path = 123
	assert is_path_exists_or_creatable(path) is False

def test_isfolder():
	path = './'
	assert isfolder(path)
	path = 'test/'
	assert isfolder(path)
	path = 'test'
	assert isfolder(path)
	path = '/home/xinshuo/test'
	assert isfolder(path)
	path = '.'
	assert isfolder(path)

	path = ''
	assert isfolder(path) is False
	path = 123
	assert isfolder(path) is False
	path = 'test.txt'
	assert isfolder(path) is False
	path = '/home/xinshuo/test.txt'
	assert isfolder(path) is False

def test_isfile():
	path = 'test.txt'
	assert isfile(path)
	path = '/home/xinshuo/test.txt'
	assert isfile(path)

	path = ''
	assert isfile(path) is False
	path = 123
	assert isfile(path) is False
	path = './'
	assert isfile(path) is False
	path = 'test/'
	assert isfile(path) is False
	path = 'test'
	assert isfile(path) is False
	path = '/home/xinshuo/test'
	assert isfile(path) is False
	path = '.'
	assert isfile(path) is False

if __name__ == '__main__':
	pytest.main([__file__])