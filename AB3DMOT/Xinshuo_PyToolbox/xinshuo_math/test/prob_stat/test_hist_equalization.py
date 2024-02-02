# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com
import numpy as np
from PIL import Image

import init_paths
from prob_stat import hist_equalization
from xinshuo_visualization import visualize_distribution, visualize_image

def test_hist_equalization():
	print('testing for gaussian distribution')
	random_data = np.random.normal(0.5, 0.1, 10000)
	visualize_distribution(random_data, vis=True)
	num_bins = 100
	data_equalized = hist_equalization(random_data, num_bins=num_bins)
	visualize_distribution(data_equalized, vis=True)
	
	print('testing for image data')
	image_path = 'lena.jpg'
	img = np.array(Image.open(image_path).convert('L'))
	visualize_image(img, vis=True)	
	num_bins = 256
	data_equalized = hist_equalization(img, num_bins=num_bins)
	visualize_image(data_equalized, vis=True)

	print('\n\nDONE! SUCCESSFUL!!\n')
if __name__ == '__main__':
	test_hist_equalization()