# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com
from PIL import Image
import numpy as np

import init_paths
from image_processing import hwc2chw
from xinshuo_visualization import visualize_image

def test_hwc2chw():
	print('test same channel image, should be the same')
	image_path = '../lena.jpg'
	img = np.array(Image.open(image_path).convert('RGB'))
	visualize_image(img[:, :, 0], vis=True)
	img_shape = img.shape
	chw_img = hwc2chw(img)
	visualize_image(chw_img[0, :, :], vis=True)
	print(chw_img.shape)
	print(img_shape)
	assert chw_img.shape[0] == img_shape[2] and chw_img.shape[1] == img_shape[0] and chw_img.shape[2] == img_shape[1]

	print('test different channel image, should not be the same')
	image_path = '../lena.jpg'
	img = np.array(Image.open(image_path).convert('RGB'))
	visualize_image(img[:, :, 0], vis=True)
	img_shape = img.shape
	chw_img = hwc2chw(img)
	visualize_image(chw_img[1, :, :], vis=True)
	print(chw_img.shape)
	print(img_shape)
	assert chw_img.shape[0] == img_shape[2] and chw_img.shape[1] == img_shape[0] and chw_img.shape[2] == img_shape[1]

	print('\n\nDONE! SUCCESSFUL!!\n')

if __name__ == '__main__':
	test_hwc2chw()