# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com
from PIL import Image
import numpy as np

import init_paths
from image_processing import image_pad_around
from xinshuo_visualization import visualize_image

def test_image_pad_around():
	image_path = '../lena.jpg'
	img = Image.open(image_path)

	print('test 2d matrix')
	np_data = (np.random.rand(3, 3) * 255.).astype('uint8')
	pad_rect = [0, 2, 1, 1]
	img_padded = image_pad_around(np_data, pad_rect=pad_rect, pad_value=10)
	print(np_data)
	print(img_padded)
	assert img_padded.shape == (6, 4), 'the padded image does not have a good shape'

	print('test 2d matrix with negative pad_rect')
	np_data = (np.random.rand(3, 3) * 255.).astype('uint8')
	pad_rect = [-1, 2, 1, 1]
	try:
		img_padded = image_pad_around(np_data, pad_rect=pad_rect, pad_value=10)
	except AssertionError:
		print('the pad rect should not include negative integers')

	print('test 2d matrix with float pad_rect')
	np_data = (np.random.rand(3, 3) * 255.).astype('uint8')
	pad_rect = [0, 0.5, 1, 1]
	try:
		img_padded = image_pad_around(np_data, pad_rect=pad_rect, pad_value=10)
	except AssertionError:
		print('the pad rect should not include floating number')

	visualize_image(img, vis=True)
	img_padded = image_pad_around(img, [50, 100, 150, 200], pad_value=150)
	visualize_image(img_padded, vis=True)

	img_padded = image_pad_around(img.convert('L'), [50, 100, 150, 200], pad_value=10)
	visualize_image(img_padded, vis=True)

	print('\n\nDONE! SUCCESSFUL!!\n')

if __name__ == '__main__':
	test_image_pad_around()
