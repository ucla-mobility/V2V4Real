# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com
from PIL import Image
import numpy as np

import init_paths
from image_processing import hsv2rgb, rgb2hsv
from xinshuo_visualization import visualize_image

def test_hsv2rgb():
	print('test input image as jpg')
	image_path = '../lena.jpg'
	img = Image.open(image_path).convert('RGB')
	visualize_image(img, vis=True)
	hsv_img = rgb2hsv(img)
	visualize_image(hsv_img, vis=True)
	rgb_img = hsv2rgb(hsv_img)
	visualize_image(rgb_img, vis=True)

	print('test input image as png')
	image_path = '../lena.png'
	img = Image.open(image_path).convert('RGB')
	visualize_image(img, vis=True)
	hsv_img = rgb2hsv(img)
	visualize_image(hsv_img, vis=True)
	rgb_img = hsv2rgb(hsv_img)
	visualize_image(rgb_img, vis=True)

	print('\n\nDONE! SUCCESSFUL!!\n')

if __name__ == '__main__':
	test_hsv2rgb()
