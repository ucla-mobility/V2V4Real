# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com
from PIL import Image
import numpy as np
from numpy.testing import assert_almost_equal

import init_paths
from image_processing import image_mean
from xinshuo_miscellaneous import CHECK_EQ_NUMPY

def test_image_mean():
	print('test HWC with gray image')
	img = (np.random.rand(4, 4, 1) * 255.).astype('uint8')
	img_bak = img.copy()
	mean_img = image_mean(img)
	assert mean_img.shape == (4, 4, 1)
	assert CHECK_EQ_NUMPY(mean_img, img), 'the original image should be equal to the copy'
	mean_img += 1
	assert not CHECK_EQ_NUMPY(mean_img, img), 'the original image should be equal to the copy'
	assert CHECK_EQ_NUMPY(img_bak, img), 'the original image should be equal to the backup version'

	print('test NHWC with color images')
	img = (np.random.rand(12, 4, 4, 3) * 255.).astype('uint8')
	img_bak = img.copy()
	mean_img = image_mean(img)
	assert mean_img.shape == (4, 4, 3)
	assert CHECK_EQ_NUMPY(mean_img, np.mean(img, axis=0).astype('uint8')), 'the original image should be equal to the copy'
	mean_img += 1
	assert not CHECK_EQ_NUMPY(mean_img, np.mean(img, axis=0).astype('uint8')), 'the original image should be equal to the copy'
	assert CHECK_EQ_NUMPY(img_bak, img), 'the original image should be equal to the backup version'

	print('\n\nDONE! SUCCESSFUL!!\n')
	
if __name__ == '__main__':
	test_image_mean()