# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com
import numpy as np
from PIL import Image

import init_paths
from xinshuo_visualization import visualize_image_with_bbox

def test_visualize_image_with_bbox():
	image_path = '../lena.png'

	print('testing basic')
	img = Image.open(image_path).convert('RGB')
	bbox = [[200, 300, 300, 400], [0, 0, 400, 400]]
	visualize_image_with_bbox(img, bbox, vis=True)	

	print('testing linewidth')
	img = Image.open(image_path).convert('RGB')
	bbox = [[200, 300, 300, 400], [0, 0, 400, 400]]
	visualize_image_with_bbox(img, bbox, vis=True, linewidth=10)	

	print('testing edge color')
	img = Image.open(image_path).convert('RGB')
	bbox = [[200, 300, 300, 400], [0, 0, 400, 400]]
	visualize_image_with_bbox(img, bbox, vis=True, linewidth=10, edge_color_index=0)	

	print('\n\nDONE! SUCCESSFUL!!\n')
	
if __name__ == '__main__':
	test_visualize_image_with_bbox()