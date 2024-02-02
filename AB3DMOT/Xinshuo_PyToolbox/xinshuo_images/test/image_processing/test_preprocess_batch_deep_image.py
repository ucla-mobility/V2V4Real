# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com
from PIL import Image
import numpy as np
from numpy.testing import assert_almost_equal

import init_paths
from image_processing import preprocess_batch_deep_image, bgr2rgb, chw2hwc
from xinshuo_miscellaneous import CHECK_EQ_NUMPY

def test_preprocess_batch_deep_image():
	print('test HW3, no rgb2bgr')
	img = np.random.rand(100, 200, 3).astype('float32')
	img_bak = img.copy()
	batch_image = preprocess_batch_deep_image(img, rgb2bgr=False)
	assert batch_image.shape == (1, 3, 100, 200)
	assert CHECK_EQ_NUMPY(batch_image, np.transpose(img, (2, 0, 1)).reshape((1, 3, 100, 200))), 'the original image should be equal to the copy'
	batch_image += 1
	assert not CHECK_EQ_NUMPY(batch_image, np.transpose(img, (2, 0, 1)).reshape((1, 3, 100, 200))), 'the original image should be equal to the copy'
	assert CHECK_EQ_NUMPY(img_bak, img), 'the original image should be equal to the backup version'

	print('test NHW3, no rgb2bgr')
	img = np.random.rand(12, 100, 100, 3).astype('float32')
	img_bak = img.copy()
	batch_image = preprocess_batch_deep_image(img, rgb2bgr=False)
	assert CHECK_EQ_NUMPY(batch_image, np.transpose(img, (0, 3, 1, 2))), 'the original image should be equal to the copy'
	batch_image += 1
	assert not CHECK_EQ_NUMPY(batch_image, np.transpose(img, (0, 3, 1, 2))), 'the original image should be equal to the copy'
	assert CHECK_EQ_NUMPY(img_bak, img), 'the original image should be equal to the backup version'

	print('test HW3, with rgb2bgr')
	img = np.random.rand(100, 200, 3).astype('float32')
	img_bak = img.copy()
	batch_image = preprocess_batch_deep_image(img)
	assert batch_image.shape == (1, 3, 100, 200)
	assert CHECK_EQ_NUMPY(bgr2rgb(chw2hwc(batch_image[0])), img), 'the original image should be equal to the copy'
	assert CHECK_EQ_NUMPY(img_bak, img), 'the original image should be equal to the backup version'

	print('test HW3, with rgb2bgr, pixel mean and std')
	img = np.random.rand(2, 4, 3).astype('float32')
	img[img < 0.5] = 0.5
	pixel_mean = [0.3, 0.4, 0.5]
	pixel_std = [0.9, 0.99, 0.999]
	batch_image = preprocess_batch_deep_image(img, pixel_mean=pixel_mean, pixel_std=pixel_std)
	assert batch_image.shape == (1, 3, 2, 4)
	assert_almost_equal((chw2hwc(batch_image[0])), (bgr2rgb(img) - np.array(pixel_mean)) / np.array(pixel_std), err_msg='the original image should be equal to the copy')

	print('test HW3, with rgb2bgr, single pixel mean')
	img = np.random.rand(2, 4, 3).astype('float32')
	img[img < 0.5] = 0.5
	pixel_mean = 0.5
	pixel_std = [0.9, 0.99, 0.999]
	batch_image = preprocess_batch_deep_image(img, pixel_mean=pixel_mean, pixel_std=pixel_std)
	assert batch_image.shape == (1, 3, 2, 4)
	assert_almost_equal((chw2hwc(batch_image[0])), (bgr2rgb(img) - np.array(pixel_mean)) / np.array(pixel_std), err_msg='the original image should be equal to the copy')

	print('test HW3, with rgb2bgr, single pixel std')
	img = np.random.rand(2, 4, 3).astype('float32')
	img[img < 0.5] = 0.5
	pixel_mean = 0.5
	pixel_std = 0.9
	batch_image = preprocess_batch_deep_image(img, pixel_mean=pixel_mean, pixel_std=pixel_std)
	assert batch_image.shape == (1, 3, 2, 4)
	assert_almost_equal((chw2hwc(batch_image[0])), (bgr2rgb(img) - np.array(pixel_mean)) / np.array(pixel_std), err_msg='the original image should be equal to the copy')

	print('\n\nDONE! SUCCESSFUL!!\n')
	
if __name__ == '__main__':
	test_preprocess_batch_deep_image()