# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com
from PIL import Image
import numpy as np

import init_paths
from image_processing import gray2rgb
from xinshuo_miscellaneous import isgrayimage_dimension, iscolorimage_dimension
from xinshuo_visualization import visualize_image

def test_gray2rgb():
	image_path = '../lena.jpg'
	img = Image.open(image_path).convert('L')
	img = np.array(img)
	print('input grayscale image has dimension of: '),
	print(img.shape)
	assert isgrayimage_dimension(img), 'the input image is not a gray image'
	visualize_image(img, vis=True)

	img_rgb = gray2rgb(img, with_color=True)
	print('converted rgb image has dimension of: '),
	print(img_rgb.shape)
	assert iscolorimage_dimension(img_rgb), 'the converted image is not a color image'
	visualize_image(img_rgb, vis=True)

	# test when input image is float image
	test_floatimage = (img.astype('float32')) / 255.
	img_rgb = gray2rgb(test_floatimage, with_color=True)
	assert iscolorimage_dimension(img_rgb), 'the converted image is not a color image'
	visualize_image(img_rgb, vis=True)

	# test when input image is PIL image
	test_pil_format_image = Image.fromarray(img)
	img_rgb = gray2rgb(test_pil_format_image, with_color=True)
	assert iscolorimage_dimension(img_rgb), 'the converted image is not a color image'

	print('\n\nDONE! SUCCESSFUL!!\n')
	
if __name__ == '__main__':
	test_gray2rgb()
