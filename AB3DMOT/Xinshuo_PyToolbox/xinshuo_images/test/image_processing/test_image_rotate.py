# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com
from PIL import Image
import numpy as np

import init_paths
from image_processing import image_rotate
from xinshuo_visualization import visualize_image

def test_image_rotate():
	print('test input image as numpy uint8 image')
	image_path = '../lena.png'
	img = np.array(Image.open(image_path).convert('RGB')).astype('uint8')
	visualize_image(img, vis=True)
	rotated_img = image_rotate(img, input_angle=45)
	visualize_image(rotated_img, vis=True)
	
	print('test input image with a out of general range rotation angle')
	visualize_image(img, vis=True)
	rotated_img = image_rotate(img, input_angle=720)
	visualize_image(rotated_img, vis=True)

	# ######################################## test failure cases
	print('test two input angles')
	try:
		rotated_img = image_rotate(img, input_angle=[45, 90])
		sys.exit('\nwrong! never should be here\n\n')
	except AssertionError:
		print('the rotation angle should be a scalar')
	except TypeError:
		print('the rotation angle should be a scalar')

	print('\n\nDONE! SUCCESSFUL!!\n')

if __name__ == '__main__':
	test_image_rotate()