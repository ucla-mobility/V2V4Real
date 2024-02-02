# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com
import numpy as np

import init_paths
from bbox_transform import get_center_crop_bbox
from xinshuo_miscellaneous import CHECK_EQ_NUMPY

def test_get_center_crop_bbox():
	print('check basic')
	bbox = [1, 1, 10, 10]
	crop_bbox = get_center_crop_bbox(bbox)
	print(bbox)
	print(crop_bbox)
	assert CHECK_EQ_NUMPY(crop_bbox, np.array([-4, -4, 10, 10]).reshape((1, 4)))
	
	print('\n\nDONE! SUCCESSFUL!!\n')
	
if __name__ == '__main__':
	test_get_center_crop_bbox()