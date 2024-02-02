# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com
from PIL import Image
import numpy as np

import init_paths
from image_processing import chw2hwc, hwc2chw
from xinshuo_visualization import visualize_image

def test_chw2hwc():
	print('test input image from hwc to chw to hwc')
	image_path = '../lena.png'
	img = np.array(Image.open(image_path).convert('RGB'))
	visualize_image(img[:, :, 0], vis=True)
	img_shape = img.shape
	chw_img = hwc2chw(img)
	visualize_image(chw_img[0, :, :], vis=True)
	hwc_img = chw2hwc(chw_img)
	visualize_image(hwc_img[:, :, 0], vis=True)
	assert hwc_img.shape == img_shape

	print('\n\nDONE! SUCCESSFUL!!\n')

if __name__ == '__main__':
	test_chw2hwc()