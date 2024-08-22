# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com
from PIL import Image
import numpy as np

import init_paths
from private import safe_image
from xinshuo_miscellaneous import CHECK_EQ_NUMPY
from xinshuo_visualization import visualize_image

def test_safe_image():
	image_path = '../lena.jpg'

	# test when the input image is pil image
	img_pil = Image.open(image_path)
	img_bak = np.array(img_pil).copy()
	copy_image, isnan = safe_image(img_pil)
	assert CHECK_EQ_NUMPY(copy_image, np.asarray(img_pil)), 'the original image should be equal to the copy'
	copy_image += 1
	assert not CHECK_EQ_NUMPY(copy_image, np.asarray(img_pil)), 'the original image should be equal to the copy'
	assert CHECK_EQ_NUMPY(img_bak, np.asarray(img_pil)), 'the original image should be equal to the backup version'
	assert not isnan

	# test when the input image is a numpy image
	img_numpy = np.asarray(img_pil)			# read only
	img_bak = img_numpy.copy()
	copy_image, isnan = safe_image(img_numpy)
	assert CHECK_EQ_NUMPY(copy_image, img_numpy), 'the original image should be equal to the copy'
	copy_image += 1
	assert not CHECK_EQ_NUMPY(copy_image, img_numpy), 'the original image should be equal to the copy'
	assert CHECK_EQ_NUMPY(img_bak, img_numpy), 'the original image should be equal to the backup version'
	assert not isnan
	
	print('\n\nDONE! SUCCESSFUL!!\n')
	
if __name__ == '__main__':
	test_safe_image()
