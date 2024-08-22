# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com
from PIL import Image
import numpy as np
from numpy.testing import assert_almost_equal

import init_paths
from image_processing import image_draw_mask, image_resize
from xinshuo_miscellaneous import CHECK_EQ_NUMPY
from xinshuo_visualization import visualize_image
from xinshuo_io import save_image

def test_image_draw_mask():
	# mask = '../rainbow.jpg'
	mask = '/home/xinshuo/Dropbox/test7.png'
	mask = Image.open(mask).convert('RGB')
	# mask = image_resize(mask, target_size=(500, 500))
	mask = image_resize(mask, target_size=(1148, 749))
	visualize_image(mask, vis=True)
	# image_path = '../lena.png'
	image_path = '/home/xinshuo/test8.png'

	print('test with pil image')
	img = Image.open(image_path).convert('RGB')
	img = image_resize(img, target_size=(1148, 749))
	print(img.shape)
	print(mask.shape)
	masked_img = image_draw_mask(img, mask, transparency=0.5)
	# visualize_image(img, vis=True)
	# visualize_image(masked_img, vis=True)
	save_image(masked_img, save_path='7716_new.png')

	print('test with numpy image with different transparency')
	img = np.array(Image.open(image_path).convert('RGB')).astype('float32') / 255.
	img_bak = img.copy()
	masked_img = image_draw_mask(image_resize(img, target_size=(500, 500)), mask, transparency=0.9)
	visualize_image(img, vis=True)
	visualize_image(masked_img, vis=True)
	masked_img += 1
	assert CHECK_EQ_NUMPY(img_bak, img), 'the original image should be equal to the backup version'

	print('\n\nDONE! SUCCESSFUL!!\n')
	
if __name__ == '__main__':
	test_image_draw_mask()