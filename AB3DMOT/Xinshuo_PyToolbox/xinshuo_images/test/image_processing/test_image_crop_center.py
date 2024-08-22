# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com
from PIL import Image
import numpy as np

import init_paths
from image_processing import image_crop_center
from xinshuo_visualization import visualize_image
from xinshuo_miscellaneous import CHECK_EQ_NUMPY

def test_crop_center():
	image_path = '../lena.jpg'
	img = Image.open(image_path).convert('RGB')

	#################################### test with 4 elements in center_rect #########################################
	print('test 2d matrix intersected on the top left')
	np_data = (np.random.rand(5, 5) * 255.).astype('uint8')
	center_rect = [1, 2, 4, 6]
	img_padded, crop_bbox, crop_bbox_clipped = image_crop_center(np_data, center_rect=center_rect, pad_value=10)
	print(np_data)
	print(img_padded)
	print(img_padded.shape)
	assert img_padded.shape == (6, 4), 'the padded image does not have a good shape'
	assert CHECK_EQ_NUMPY(crop_bbox, np.array([[-1, -1, 4, 6]]))
	assert CHECK_EQ_NUMPY(crop_bbox_clipped, np.array([[0, 0, 3, 5]]))

	print('test 2d matrix intersected on the bottom right')
	np_data = (np.random.rand(5, 5) * 255.).astype('uint8')
	center_rect = [3, 3, 5, 6]
	img_padded, crop_bbox, crop_bbox_clipped = image_crop_center(np_data, center_rect=center_rect, pad_value=10)
	print(np_data)
	print(img_padded)
	assert img_padded.shape == (6, 5), 'the padded image does not have a good shape'
	assert CHECK_EQ_NUMPY(crop_bbox, np.array([[1, 0, 5, 6]]))
	assert CHECK_EQ_NUMPY(crop_bbox_clipped, np.array([[1, 0, 4, 5]]))

	print('test with color image, clipped on the left')
	center_rect = [0, 50, 100, 100]
	img_cropped, crop_bbox, crop_bbox_clipped = image_crop_center(img, center_rect=center_rect, pad_value=100)
	assert CHECK_EQ_NUMPY(crop_bbox, np.array([[-50, 0, 100, 100]]))
	assert CHECK_EQ_NUMPY(crop_bbox_clipped, np.array([[0, 0, 50, 100]]))
	visualize_image(img, vis=True)
	visualize_image(img_cropped, vis=True)

	#################################### test with 2 elements in center_rect #########################################
	print('test 2d matrix - basic')
	np_data = (np.random.rand(5, 5) * 255.).astype('uint8')
	center_rect = [3, 3]
	img_padded, crop_bbox, crop_bbox_clipped = image_crop_center(np_data, center_rect=center_rect, pad_value=10)
	print(np_data)
	print(img_padded)
	print(img_padded.shape)
	assert img_padded.shape == (3, 3), 'the padded image does not have a good shape'
	assert CHECK_EQ_NUMPY(crop_bbox, np.array([[1, 1, 3, 3]]))
	assert CHECK_EQ_NUMPY(crop_bbox_clipped, np.array([[1, 1, 3, 3]]))

	print('test 2d matrix - boundary')
	np_data = (np.random.rand(5, 5) * 255.).astype('uint8')
	center_rect = [5, 5]
	img_padded, crop_bbox, crop_bbox_clipped = image_crop_center(np_data, center_rect=center_rect, pad_value=10)
	print(np_data)
	print(img_padded)
	assert img_padded.shape == (5, 5), 'the padded image does not have a good shape'
	assert CHECK_EQ_NUMPY(crop_bbox, np.array([[0, 0, 5, 5]]))
	assert CHECK_EQ_NUMPY(crop_bbox_clipped, np.array([[0, 0, 5, 5]]))

	print('test 2d matrix - intersected')
	np_data = (np.random.rand(5, 5) * 255.).astype('uint8')
	center_rect = [7, 7]
	img_padded, crop_bbox, crop_bbox_clipped = image_crop_center(np_data, center_rect=center_rect, pad_value=10)
	print(np_data)
	print(img_padded)
	assert img_padded.shape == (7, 7), 'the padded image does not have a good shape'
	assert CHECK_EQ_NUMPY(crop_bbox, np.array([[-1, -1, 7, 7]]))
	assert CHECK_EQ_NUMPY(crop_bbox_clipped, np.array([[0, 0, 5, 5]]))

	print('test with color image, clipped on the center')
	center_rect = [100, 100]
	img_cropped, crop_bbox, crop_bbox_clipped = image_crop_center(img, center_rect=center_rect, pad_value=100)
	visualize_image(img, vis=True)
	visualize_image(img_cropped, vis=True)

	print('test with color image, interesected')
	center_rect = [600, 600]
	img_cropped, crop_bbox, crop_bbox_clipped = image_crop_center(img, center_rect=center_rect, pad_value=100)
	visualize_image(img, vis=True)
	visualize_image(img_cropped, vis=True)

	print('test with grayscale image')
	center_rect = [100, 100]
	img_cropped, crop_bbox, crop_bbox_clipped = image_crop_center(img.convert('L'), center_rect=center_rect, pad_value=100)
	visualize_image(img.convert('L'), vis=True)
	visualize_image(img_cropped, vis=True)

	print('\n\nDONE! SUCCESSFUL!!\n')

if __name__ == '__main__':
	test_image_crop_center()