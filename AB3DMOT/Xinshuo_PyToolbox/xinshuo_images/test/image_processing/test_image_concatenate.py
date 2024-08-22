# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com
from PIL import Image
import numpy as np

import init_paths
from image_processing import image_concatenate
from xinshuo_visualization import visualize_image

def test_image_concatenate():
	image_path = '../lena.png'
	image = np.array(Image.open(image_path).convert('RGB'))

	print('test a single image')
	target_size = [600, 600]
	image_concatenated = image_concatenate(image, target_size=target_size)
	visualize_image(image_concatenated, vis=True)

	print('test multiple images')
	image_list = [image, image, image, image, image, image, image]
	image_all = np.stack(image_list, axis=0)
	target_size = [1200, 1800]
	image_concatenated = image_concatenate(image_all, target_size=target_size)
	visualize_image(image_concatenated, vis=True)

	print('test multiple images with grid size')
	image_list = [image, image, image, image, image, image, image]
	image_all = np.stack(image_list, axis=0)
	target_size = [600, 3600]
	image_concatenated = image_concatenate(image_all, target_size=target_size, grid_size=[1, 7])
	visualize_image(image_concatenated, vis=True)

	print('test multiple images with grid size')
	image_list = [image, image, image, image, image, image, image]
	image_all = np.stack(image_list, axis=0)
	target_size = [1800, 300]
	image_concatenated = image_concatenate(image_all, target_size=target_size, grid_size=[7, 1])
	visualize_image(image_concatenated, vis=True)

	print('test multiple images with edge factor')
	image_list = [image, image, image, image, image, image, image]
	image_all = np.stack(image_list, axis=0)
	target_size = [1200, 1800]
	image_concatenated = image_concatenate(image_all, target_size=target_size, edge_factor=0.5)
	visualize_image(image_concatenated, vis=True)

	print('test multiple grayscale images')
	image_grayscale = np.array(Image.open(image_path).convert('L'))
	image_list = [image_grayscale, image_grayscale]
	image_all = np.expand_dims(np.stack(image_list, axis=0), axis=3)		# NHW1
	target_size = [1200, 1800]
	image_concatenated = image_concatenate(image_all, target_size=target_size)
	visualize_image(image_concatenated, vis=True)

	print('\n\nDONE! SUCCESSFUL!!\n')
	
if __name__ == '__main__':
	test_image_concatenate()