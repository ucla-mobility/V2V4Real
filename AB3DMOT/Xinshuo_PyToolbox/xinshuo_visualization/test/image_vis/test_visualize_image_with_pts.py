# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com
import numpy as np
from PIL import Image

import init_paths
from xinshuo_visualization import visualize_image_with_pts

def test_visualize_image_with_pts():
	image_path = '../lena.png'

	print('testing basic')
	img = Image.open(image_path).convert('L')
	pts_array = [[200, 300], [400, 400]]
	visualize_image_with_pts(img, pts_array, vis=True)	

	print('testing basic')
	img = Image.open(image_path).convert('L')
	pts_array = [[200, 300, 1], [400, 400, 1]]
	visualize_image_with_pts(img, pts_array, vis=True)	

	print('testing color index')
	img = Image.open(image_path).convert('L')
	pts_array = [[200, 300, 1], [400, 400, 1]]
	visualize_image_with_pts(img, pts_array, color_index=1, vis=True)	

	print('testing pts size')
	img = Image.open(image_path).convert('L')
	pts_array = [[200, 300, 1], [400, 400, 1]]
	visualize_image_with_pts(img, pts_array, pts_size=100, vis=True)	

	print('testing vis threshold')
	img = Image.open(image_path).convert('L')
	pts_array = [[200, 300, 0.4], [400, 400, 0.2]]
	visualize_image_with_pts(img, pts_array, vis=True)	

	print('testing vis threshold')
	img = Image.open(image_path).convert('L')
	pts_array = [[200, 300, 0.4], [400, 400, 0.8]]
	visualize_image_with_pts(img, pts_array, vis_threshold=0.7, vis=True)	

	print('testing vis threshold')
	img = Image.open(image_path).convert('L')
	pts_array = [[200, 300, 1], [400, 400, 0]]
	visualize_image_with_pts(img, pts_array, vis_threshold=0.7, vis=True)	

	print('testing labels')
	img = Image.open(image_path).convert('L')
	pts_array = [[200, 300, 0.4], [400, 400, 0.8]]
	visualize_image_with_pts(img, pts_array, label=True, vis=True)	

	print('testing label list')
	img = Image.open(image_path).convert('L')
	pts_array = [[200, 300, 0.4], [400, 400, 0.8]]
	visualize_image_with_pts(img, pts_array, label_list=['2', '6'], vis=True)	

	print('testing label size')
	img = Image.open(image_path).convert('L')
	pts_array = [[200, 300, 0.4], [400, 400, 0.8]]
	visualize_image_with_pts(img, pts_array, label_list=['2', '6'], label_size=100, vis=True)	

	print('testing a dict of pts')
	img = Image.open(image_path).convert('L')
	pts_array1 = [[200, 300, 0.4], [400, 400, 0.8]]
	pts_array2 = [[100, 100, 0.4], [50, 50, 0.2], [150, 150, 0.6]]
	pts_array = {'pts1': pts_array1, 'pts2': pts_array2}
	visualize_image_with_pts(img, pts_array, label_list=['3', '6', '9', '12'], vis=True)	

	print('\n\nDONE! SUCCESSFUL!!\n')
	
if __name__ == '__main__':
	test_visualize_image_with_pts()