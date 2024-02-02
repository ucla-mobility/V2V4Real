# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com
from PIL import Image
import numpy as np

import init_paths
from math_conversion import generate_gaussian_heatmap
from xinshuo_visualization import visualize_image
from xinshuo_miscellaneous import CHECK_EQ_NUMPY

def test_generate_gaussian_heatmap():
	print('test single point')
	input_pts = [300, 400, 1]
	image_size = [800, 600]
	std = 10
	heatmap, mask, _ = generate_gaussian_heatmap(input_pts, image_size=image_size, std=std)
	assert heatmap.shape == (800, 600, 2)
	assert mask.shape == (1, 1, 2)
	assert CHECK_EQ_NUMPY(mask, np.array([[[1, 1]]]))
	visualize_image(heatmap[:, :, 0], vis=True)

	print('test two points')
	input_pts = [[300, 400, 1], [400, 400, 1]]
	image_size = [800, 600]
	std = 10
	heatmap, mask, _ = generate_gaussian_heatmap(input_pts, image_size=image_size, std=std)
	assert heatmap.shape == (800, 600, 3)
	assert mask.shape == (1, 1, 3)
	assert CHECK_EQ_NUMPY(mask, np.array([[[1, 1, 1]]]))
	visualize_image(heatmap[:, :, -1], vis=True)
	visualize_image(heatmap[:, :, 0], vis=True)
	visualize_image(heatmap[:, :, 1], vis=True)

	print('test two points with invalid one')
	input_pts = [[300, 400, 1], [400, 400, -1]]
	image_size = [800, 600]
	std = 10
	heatmap, mask, _ = generate_gaussian_heatmap(input_pts, image_size=image_size, std=std)
	assert heatmap.shape == (800, 600, 3)
	assert mask.shape == (1, 1, 3)
	assert CHECK_EQ_NUMPY(mask, np.array([[[1, 0, 1]]]))
	visualize_image(heatmap[:, :, -1], vis=True)
	visualize_image(heatmap[:, :, 0], vis=True)
	visualize_image(heatmap[:, :, 1], vis=True)

	print('test two points with invisible one')
	input_pts = [[300, 400, 1], [-400, -400, 0]]
	image_size = [800, 600]
	std = 10
	heatmap, mask, mask_visible = generate_gaussian_heatmap(input_pts, image_size=image_size, std=std)
	assert heatmap.shape == (800, 600, 3)
	assert mask.shape == (1, 1, 3)
	assert CHECK_EQ_NUMPY(mask, np.array([[[1, 1, 1]]]))
	assert CHECK_EQ_NUMPY(mask_visible, np.array([[[1, 0, 1]]]))
	visualize_image(heatmap[:, :, -1], vis=True)
	visualize_image(heatmap[:, :, 0], vis=True)
	visualize_image(heatmap[:, :, 1], vis=True)

	print('test two points with invisible and invalid')
	input_pts = [[300, 400, -1], [-400, -400, 0]]
	image_size = [800, 600]
	std = 10
	heatmap, mask, mask_visible = generate_gaussian_heatmap(input_pts, image_size=image_size, std=std)
	assert heatmap.shape == (800, 600, 3)
	assert mask.shape == (1, 1, 3)
	assert CHECK_EQ_NUMPY(mask, np.array([[[0, 1, 1]]]))
	assert CHECK_EQ_NUMPY(mask_visible, np.array([[[0, 0, 1]]]))
	visualize_image(heatmap[:, :, -1], vis=True)
	visualize_image(heatmap[:, :, 0], vis=True)
	visualize_image(heatmap[:, :, 1], vis=True)

	print('\n\nDONE! SUCCESSFUL!!\n')
	
if __name__ == '__main__':
	test_generate_gaussian_heatmap()