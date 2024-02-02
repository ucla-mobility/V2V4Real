# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com
from PIL import Image
import numpy as np

import init_paths
from image_operator import linear_filter
from xinshuo_visualization import visualize_image

def test_gaussian_filter():
	image_path = '../lena.png'

	print('testing for grayscale image with Gaussian filter')
	img = Image.open(image_path).convert('L')
	filter = linear_filter()
	gaussian_kernel = filter.gaussian()
	filtered_img = filter.convolve(img)
	visualize_image(img, vis=True)
	visualize_image(filtered_img, vis=True)

	print('testing for color image with Gaussian filter')
	img = Image.open(image_path).convert('RGB')
	filter = linear_filter()
	gaussian_kernel = filter.gaussian()
	filtered_img = filter.convolve(img)
	visualize_image(img, vis=True)
	visualize_image(filtered_img, vis=True)

	print('\n\nDONE! SUCCESSFUL!!\n')
	
if __name__ == '__main__':
	test_gaussian_filter()