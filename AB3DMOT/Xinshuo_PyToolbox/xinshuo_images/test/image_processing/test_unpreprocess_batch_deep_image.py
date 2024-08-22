# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com
from PIL import Image
import numpy as np
from numpy.testing import assert_almost_equal

import init_paths
from image_processing import preprocess_batch_deep_image, unpreprocess_batch_deep_image, bgr2rgb, chw2hwc
from xinshuo_miscellaneous import CHECK_EQ_NUMPY

def test_unpreprocess_batch_deep_image():
	print('test HW3, no rgb2bgr')
	img = (np.random.rand(100, 200, 3) * 255.).astype('uint8')
	img_bak = img.copy()
	batch_image = preprocess_batch_deep_image(img, rgb2bgr=False)
	img_inv = unpreprocess_batch_deep_image(batch_image, bgr2rgb=False)
	assert img_inv.shape == (1, 100, 200, 3)
	assert CHECK_EQ_NUMPY(img, img_inv[0]), 'the original image should be equal to the copy'
	assert CHECK_EQ_NUMPY(img_bak, img), 'the original image should be equal to the backup version'

	print('test NHW3, no rgb2bgr')
	img = (np.random.rand(12, 100, 200, 3) * 255.).astype('uint8')
	img_bak = img.copy()
	batch_image = preprocess_batch_deep_image(img, rgb2bgr=False)
	img_inv = unpreprocess_batch_deep_image(batch_image, bgr2rgb=False)
	assert img_inv.shape == (12, 100, 200, 3)
	assert CHECK_EQ_NUMPY(img, img_inv), 'the original image should be equal to the copy'
	assert CHECK_EQ_NUMPY(img_bak, img), 'the original image should be equal to the backup version'

	print('test HW3, with rgb2bgr')
	img = (np.random.rand(100, 200, 3) * 255.).astype('uint8')
	img_bak = img.copy()
	batch_image = preprocess_batch_deep_image(img)
	img_inv = unpreprocess_batch_deep_image(batch_image)
	assert img_inv.shape == (1, 100, 200, 3)
	assert CHECK_EQ_NUMPY(img, img_inv[0]), 'the original image should be equal to the copy'
	assert CHECK_EQ_NUMPY(img_bak, img), 'the original image should be equal to the backup version'

	print('test HW3, with rgb2bgr, pixel mean and std')
	img = (np.random.rand(12, 100, 200, 3) * 255.).astype('uint8')
	pixel_mean = [0.3, 0.4, 0.5]
	pixel_std = [0.3, 0.5, 0.7]
	batch_image = preprocess_batch_deep_image(img, pixel_mean=pixel_mean, pixel_std=pixel_std)
	img_inv = unpreprocess_batch_deep_image(batch_image, pixel_mean=pixel_mean, pixel_std=pixel_std)
	assert img_inv.shape == (12, 100, 200, 3)
	img_diff = img_inv - img
	assert not (np.logical_and(img_diff != 0, np.absolute(img_diff) != 255)).any()

	print('test HW3, with rgb2bgr, single pixel mean and std')
	img = (np.random.rand(12, 100, 200, 3) * 255.).astype('uint8')
	pixel_mean = 0.4
	pixel_std = 0.5
	batch_image = preprocess_batch_deep_image(img, pixel_mean=pixel_mean, pixel_std=pixel_std)
	img_inv = unpreprocess_batch_deep_image(batch_image, pixel_mean=pixel_mean, pixel_std=pixel_std)
	assert img_inv.shape == (12, 100, 200, 3)
	img_diff = img_inv - img
	assert not (np.logical_and(img_diff != 0, np.absolute(img_diff) != 255)).any()

	print('\n\nDONE! SUCCESSFUL!!\n')
	
if __name__ == '__main__':
	test_unpreprocess_batch_deep_image()