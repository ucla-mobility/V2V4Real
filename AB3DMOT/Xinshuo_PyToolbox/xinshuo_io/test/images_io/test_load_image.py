# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com
import numpy as np

import init_paths
from images_io import load_image
from xinshuo_visualization import visualize_image

def test_load_image():
	image_path = '../lena.png'

	print('basic')
	img = load_image(image_path)
	assert img.shape == (512, 512, 3)	

	print('testing for resizing')
	img = load_image(image_path, resize_factor=2.0)
	assert img.shape == (1024, 1024, 3)

	print('testing for resizing')
	img = load_image(image_path, target_size=[1033, 1033])
	assert img.shape == (1033, 1033, 3)

	print('testing for rotation')
	img = load_image(image_path, input_angle=45)
	visualize_image(img, vis=True)
	assert img.shape == (726, 726, 3)

	print('testing for rotation')
	img = load_image(image_path, input_angle=450)
	visualize_image(img, vis=True)

	print('\n\nDONE! SUCCESSFUL!!\n')
	
if __name__ == '__main__':
	test_load_image()