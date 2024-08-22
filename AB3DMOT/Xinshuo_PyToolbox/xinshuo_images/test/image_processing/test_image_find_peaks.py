# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com
from PIL import Image
import numpy as np

import init_paths
from image_processing import image_find_peaks
from xinshuo_math import generate_gaussian_heatmap
from xinshuo_visualization import visualize_image, visualize_image_with_pts
from xinshuo_miscellaneous import CHECK_EQ_NUMPY

def test_image_find_peaks():
	print('test two points')
	input_pts = [[300, 400, 1], [300, 500, 1]]
	image_size = [800, 600]
	std = 50
	heatmap, _, _ = generate_gaussian_heatmap(input_pts, image_size=image_size, std=std)
	img = 1 - heatmap[:, :, -1]
	visualize_image(img, vis=True)
	peak_array, peak_global = image_find_peaks(img)
	CHECK_EQ_NUMPY(peak_array, np.array(input_pts).transpose())
	visualize_image_with_pts(img, peak_array, vis=True)
	CHECK_EQ_NUMPY(peak_global, np.array([300, 400, 1]).reshape((3, 1)))

	print('test five points')
	input_pts = [[300, 350, 1], [300, 400, 1], [200, 350, 1], [280, 320, 1], [330, 270, 1]]
	image_size = [800, 600]
	std = 50
	heatmap, _, _ = generate_gaussian_heatmap(input_pts, image_size=image_size, std=std)
	img = 1 - heatmap[:, :, -1]
	visualize_image(img, vis=True)
	peak_array, peak_global = image_find_peaks(img)
	CHECK_EQ_NUMPY(peak_array, np.array(input_pts).transpose())
	visualize_image_with_pts(img, peak_array, vis=True)

	print('\n\nDONE! SUCCESSFUL!!\n')
	
if __name__ == '__main__':
	test_image_find_peaks()