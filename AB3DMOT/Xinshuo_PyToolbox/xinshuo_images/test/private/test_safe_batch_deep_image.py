# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com
from PIL import Image
import numpy as np

import init_paths
from private import safe_batch_deep_image
from xinshuo_miscellaneous import CHECK_EQ_NUMPY

def test_safe_batch_deep_image():
	print('test 3HW')
	img = np.random.rand(3, 100, 100)
	img_bak = img.copy()
	batch_image, isnan = safe_batch_deep_image(img)
	assert CHECK_EQ_NUMPY(batch_image, img.reshape((1, 3, 100, 100))), 'the original image should be equal to the copy'
	batch_image += 1
	assert not CHECK_EQ_NUMPY(batch_image, img.reshape((1, 3, 100, 100))), 'the original image should be equal to the copy'
	assert CHECK_EQ_NUMPY(img_bak, img), 'the original image should be equal to the backup version'
	assert not isnan

	print('test N3HW')
	img = np.random.rand(12, 3, 100, 100)
	img_bak = img.copy()
	batch_image, isnan = safe_batch_deep_image(img)
	assert CHECK_EQ_NUMPY(batch_image, img), 'the original image should be equal to the copy'
	batch_image += 1
	assert not CHECK_EQ_NUMPY(batch_image, img), 'the original image should be equal to the copy'
	assert CHECK_EQ_NUMPY(img_bak, img), 'the original image should be equal to the backup version'
	assert not isnan

	######################################## test failure cases
	print('test N1HW')
	img = np.random.rand(12, 1, 100, 100)
	try:
		batch_image, isnan = safe_batch_deep_image(img)
		sys.exit('\nwrong! never should be here\n\n')
	except AssertionError:
		print('the batch image has wrong shape')

	print('test 1HW')
	img = np.random.rand(1, 100, 100)
	try:
		batch_image, isnan = safe_batch_deep_image(img)
		sys.exit('\nwrong! never should be here\n\n')
	except AssertionError:
		print('the batch image has wrong shape')

	print('\n\nDONE! SUCCESSFUL!!\n')
	
if __name__ == '__main__':
	test_safe_batch_deep_image()