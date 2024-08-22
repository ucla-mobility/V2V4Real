# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com
import numpy as np
from PIL import Image

import init_paths
from image_processing import image_hist_equalization_hsv, image_hist_equalization
from xinshuo_visualization import visualize_image

def test_image_hist_equalization():
	image_path = '../lena.png'

	print('testing for grayscale pil image')
	img = Image.open(image_path).convert('L')
	visualize_image(img, vis=True)	
	data_equalized = image_hist_equalization_hsv(img)
	visualize_image(data_equalized, vis=True)

	print('testing for grayscale numpy image')
	img = np.array(Image.open(image_path).convert('L'))
	visualize_image(img, vis=True)	
	data_equalized = image_hist_equalization(img)
	visualize_image(data_equalized, vis=True)

	print('testing for color pil image')
	img = Image.open(image_path).convert('RGB')
	visualize_image(img, vis=True)	
	data_equalized = image_hist_equalization_hsv(img)
	visualize_image(data_equalized, vis=True)

	print('testing for color numpy image')
	img = np.array(Image.open(image_path).convert('RGB'))
	visualize_image(img, vis=True)	
	data_equalized = image_hist_equalization(img)
	visualize_image(data_equalized, vis=True)

	print('\n\nDONE! SUCCESSFUL!!\n')
	
if __name__ == '__main__':
	test_image_hist_equalization()