# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com
from PIL import Image
import numpy as np

import init_paths
from image_processing import rgb2bgr, bgr2rgb
from xinshuo_visualization import visualize_image

def test_bgr2rgb():
	print('test input image as jpg')
	image_path = '../lena.jpg'
	img = Image.open(image_path).convert('RGB')
	visualize_image(img, vis=True)
	bgr_img = rgb2bgr(img)
	visualize_image(bgr_img, vis=True)
	rgb_img = bgr2rgb(bgr_img)
	visualize_image(rgb_img, vis=True)

	print('test input image as png')
	image_path = '../lena.png'
	img = Image.open(image_path).convert('RGB')
	visualize_image(img, vis=True)
	bgr_img = rgb2bgr(img)
	visualize_image(bgr_img, vis=True)
	rgb_img = bgr2rgb(bgr_img)
	visualize_image(rgb_img, vis=True)

	print('\n\nDONE! SUCCESSFUL!!\n')

if __name__ == '__main__':
	test_bgr2rgb()