# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com
import numpy as np
from PIL import Image

import init_paths
from xinshuo_visualization import visualize_image_with_pts_bbox

def test_visualize_image_with_pts_bbox():
	image_path = '../lena.png'

	print('testing basic')
	img = Image.open(image_path).convert('RGB')
	pts_array = [[200, 300, 1], [400, 400, 1]]
	window_size = 20
	visualize_image_with_pts_bbox(img, pts_array, window_size=window_size, vis=True)	

	print('testing advanced')
	img = Image.open(image_path).convert('RGB')
	pts_array = [[200, 300, 1], [400, 400, 0.2]]
	window_size = 20
	visualize_image_with_pts_bbox(img, pts_array, window_size=window_size, vis=True)	

	print('\n\nDONE! SUCCESSFUL!!\n')
	
if __name__ == '__main__':
	test_visualize_image_with_pts_bbox()