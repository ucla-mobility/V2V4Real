# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com
import numpy as np

import init_paths
from images_io import save_image, load_image
from file_io import mkdir_if_missing

def test_save_image():
	image_path = '../lena.png'
	img = load_image(image_path)
	assert img.shape == (512, 512, 3)	
	mkdir_if_missing('./tmp/')

	print('basic')
	save_image(img, save_path='./tmp/basic.png')

	print('testing for resizing')
	save_image(img, save_path='./tmp/resizing.png', resize_factor=2.0)

	print('testing for resizing')
	save_image(img, save_path='./tmp/target_size.png', target_size=[1033, 1033])

	print('testing for rotation')
	save_image(img, save_path='./tmp/rotating.png', input_angle=45)

	print('testing for rotation')
	save_image(img, save_path='./tmp/out_rotating.png', input_angle=450)

	print('\n\nDONE! SUCCESSFUL!!\n')
	
if __name__ == '__main__':
	test_save_image()