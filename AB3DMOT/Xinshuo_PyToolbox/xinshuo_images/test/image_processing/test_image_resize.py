# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com
from PIL import Image
import numpy as np

import init_paths
from image_processing import image_resize
from xinshuo_visualization import visualize_image

def test_image_resize():
	print('test input image as numpy uint8 image with resize_factor')
	img = (np.random.rand(400, 300, 1) * 255.).astype('uint8')
	resized_img = image_resize(img, resize_factor=0.5)
	assert resized_img.shape == (200, 150)

	image_path = '../lena.png'
	img = Image.open(image_path).convert('RGB')
	im_height, im_width = img.size
	visualize_image(img, vis=True)

	print('test input image as pil image')
	resized_img = image_resize(img, resize_factor=0.3)
	assert resized_img.shape == (int(round(im_height * 0.3)), int(round(im_width * 0.3)), 3)

	print('test input image as numpy float32 image with target_size')
	resized_img = image_resize(img, target_size=(1000, 1000))
	assert resized_img.shape == (1000, 1000, 3)
	visualize_image(resized_img, vis=True)

	print('test input image as numpy float32 image with target_size and bilinear')
	resized_img = image_resize(img, target_size=(1000, 1000), interp='bilinear')
	assert resized_img.shape == (1000, 1000, 3)
	visualize_image(resized_img, vis=True)

	######################################## test failure cases
	print('test random interp')
	try:
		resized_img = image_resize(img, target_size=(800, 600), interp='random')
		sys.exit('\nwrong! never should be here\n\n')
	except AssertionError:
		print('the interp is not correct')

	print('test both resize_factor and target_size')
	try:
		resized_img = image_resize(img, resize_factor=0.4, target_size=(800, 600), interp='random')
		sys.exit('\nwrong! never should be here\n\n')
	except AssertionError:
		print('the resize_factor and target_size coexist')

	print('\n\nDONE! SUCCESSFUL!!\n')

if __name__ == '__main__':
	test_image_resize()