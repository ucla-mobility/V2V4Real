# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com
from PIL import Image
import numpy as np

import init_paths
from image_operator import linear_filter
from image_processing import image_normalize
from xinshuo_visualization import visualize_image

def test_laplacian_filter():
	image_path = '../lena.png'

	print('testing for grayscale image with laplacian filter')
	img = Image.open(image_path).convert('L')
	filter = linear_filter()
	laplacian_kernel = filter.laplacian()
	filtered_img = filter.convolve(img)
	visualize_image(img, vis=True)
	visualize_image(image_normalize(filtered_img), vis=True)

	print('testing for color image with laplacian filter')
	img = Image.open(image_path).convert('RGB')
	filter = linear_filter()
	laplacian_kernel = filter.laplacian()
	filtered_img = filter.convolve(img)
	visualize_image(img, vis=True)
	visualize_image(image_normalize(filtered_img), vis=True)

	print('\n\nDONE! SUCCESSFUL!!\n')
	
if __name__ == '__main__':
	test_laplacian_filter()