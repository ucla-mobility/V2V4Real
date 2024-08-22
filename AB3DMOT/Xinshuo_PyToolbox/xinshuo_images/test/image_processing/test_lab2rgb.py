# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com
from PIL import Image
import numpy as np

import init_paths
from image_processing import lab2rgb, rgb2lab
from xinshuo_visualization import visualize_image

def test_lab2rgb():
	print('test input image as jpg')
	image_path = '../lena.jpg'
	img = Image.open(image_path).convert('RGB')
	visualize_image(img, vis=True)
	lab_img = rgb2lab(img)
	visualize_image(lab_img, vis=True)
	rgb_img = lab2rgb(lab_img)
	visualize_image(rgb_img, vis=True)

	print('test input image as png')
	image_path = '../lena.png'
	img = Image.open(image_path).convert('RGB')
	visualize_image(img, vis=True)
	lab_img = rgb2lab(img)
	visualize_image(lab_img, vis=True)
	rgb_img = lab2rgb(lab_img)
	visualize_image(rgb_img, vis=True)

	print('\n\nDONE! SUCCESSFUL!!\n')

if __name__ == '__main__':
	test_lab2rgb()
